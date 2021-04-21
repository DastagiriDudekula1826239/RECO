import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

class GnnSocialRecommendationSystem(nn.Module):
    """ Class for deep graph neural networks social recommendation system."""

    def __init__(self, user_item_updater, user_user_updater, item_updater):

        """
        Base class to model and predict the recommendations.
        :param user_item_updater: user latent factor representation (h^I_u) from user-item graph
        :param user_user_updater: user latent factor representation (h^S_u) from social graph
        :param item_updater: item latent factor representation (h^V_i) from user-item graph
        """
        super(GnnSocialRecommendationSystem, self).__init__()

        self.user_item_updater = user_item_updater
        self.user_user_updater = user_user_updater
        self.item_updater = item_updater
        self.embeddig_dim = user_user_updater.embeddig_dim

        self.w_ur1 = nn.Linear(self.embeddig_dim, self.embeddig_dim)
        self.w_ur2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)

        self.w_uir1 = nn.Linear(self.embeddig_dim, self.embeddig_dim)
        self.w_uir2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)

        self.w_u1 = nn.Linear(self.embeddig_dim * 2, self.embeddig_dim)
        self.w_u2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)

        self.w_vr1 = nn.Linear(self.embeddig_dim, self.embeddig_dim)
        self.w_vr2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)

        self.w_uv1 = nn.Linear(self.embeddig_dim * 2, self.embeddig_dim)
        self.w_uv2 = nn.Linear(self.embeddig_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)

        self.bn5 = nn.BatchNorm1d(self.embeddig_dim, momentum=0.5)
        self.bn6 = nn.BatchNorm1d(self.embeddig_dim, momentum=0.5)

        self.bn1 = nn.BatchNorm1d(self.embeddig_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embeddig_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embeddig_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)

        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):

        """
        Prediction takes place by collaborative concatenation of user latents from user-item, social graphs with item latent of user-item graph.
        P^u_i = sigma(sigma(h^S_u * h^I_u) * h^V_i)
        :param nodes_u: user samples
        :param nodes_v: item samples
        :return: prdictive scores to recommend
        """

        #ui_latent = self.user_item_updater(nodes_u)
        u_latent = self.user_user_updater(nodes_u)     #TODO: try by to(device)
        v_latent = self.item_updater(nodes_v)

        """
        ui_latent_repr = F.relu(self.bn5(self.w_uir1(ui_latent)))
        ui_latent_repr = F.dropout(ui_latent_repr, training=self.training)
        ui_latent_repr = self.w_uir2(ui_latent_repr)
        """

        uu_latent_repr = F.relu(self.bn1(self.w_ur1(u_latent)))
        uu_latent_repr = F.dropout(uu_latent_repr, training=self.training)
        uu_latent_repr = self.w_ur2(uu_latent_repr)

        """
        u_latent_factor_repr = torch.cat((uu_latent_repr, ui_latent_repr), 1)
        u_latent_factor_repr = F.relu(self.bn6(self.w_u1(u_latent_factor_repr)))
        u_latent_factor_repr = F.dropout(u_latent_factor_repr, training=self.training)
        u_latent_factor_repr = self.w_u2(u_latent_factor_repr)
        """

        item_latent_repr = F.relu(self.bn2(self.w_vr1(v_latent)))
        item_latent_repr = F.dropout(item_latent_repr, training=self.training)
        item_latent_repr = self.w_vr2(item_latent_repr)

        predScores = torch.cat((uu_latent_repr, item_latent_repr), 1)
        predScores = F.relu(self.bn3(self.w_uv1(predScores)))
        predScores = F.dropout(predScores, training=self.training)
        predScores = F.relu(self.bn4(self.w_uv2(predScores)))
        predScores = F.dropout(predScores, training=self.training)
        predictionScores = self.w_uv3(predScores)

        nodes_u = nodes_v = ui_latent = u_latent = v_latent = ui_latent_repr = uu_latent_repr = u_latent_factor_repr = item_latent_repr = None
        del [nodes_u, nodes_v, ui_latent, u_latent, v_latent, ui_latent_repr, uu_latent_repr, u_latent_factor_repr, item_latent_repr]

        return predictionScores.squeeze()

    def loss(self, nodes_u, nodes_v, targetLabels):

        """
        :param nodes_u: user samples
        :param nodes_v: item samples
        :param targetLabels: rating scores
        :return: MSELoss and predicted recommendation scores
        """
        predictionScores = self.forward(nodes_u, nodes_v)

        nodes_u = nodes_v = None
        del [nodes_u, nodes_v]

        return self.criterion(predictionScores, targetLabels), predictionScores

    def fit(self, train_DataLoader, optimizer, device, best_rmse, best_mae):

        """
        Training function.
        :param train_DataLoader: chuckwise updated train set
        :param optimizer: RMSprop() optimizer
        :param device: device
        :param best_rmse: least rmse
        :param best_mae: least mae
        :return: rmse and mae values of train data
        """
        self.train()
        train_preds = []
        train_targets = []
        running_loss = 0.0

        for i, data in enumerate(train_DataLoader, 0):
            users, items, targets = data
            optimizer.zero_grad()
            loss, predScores = self.loss(users.to(device), items.to(device), targets.to(device))
            train_preds.append(list(predScores.data.cpu().numpy()))
            train_targets.append(list(targets.data.cpu().numpy()))
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('loss: %.3f, The best rmse/mae: %.6f / %.6f' % (running_loss / 100, best_rmse, best_mae))
                running_loss = 0.0

            users = items = targets = None
            del [users, items, targets]

        train_preds = np.array(sum(train_preds, []))
        train_targets = np.array(sum(train_targets, []))
        train_rmse = sqrt(mean_squared_error(train_preds, train_targets))
        train_mae = mean_absolute_error(train_preds, train_targets)

        train_DataLoader = train_preds = train_targets = None
        del [train_DataLoader, train_preds, train_targets]

        return train_rmse, train_mae

    def evaluate(self, device, dataLoader):

        """
        Test function.
        :param device: device
        :param dataLoader: test or validation chuckwise sampled data set
        :return: rmse and mae values
        """
        self.eval()
        recommendations = []
        true_targets = []
        with torch.no_grad():
            for users, items, targets in dataLoader:
                users, items, targets = users.to(device), items.to(device), targets.to(device)
                rating_scores = self.forward(users, items)
                recommendations.append(list(rating_scores.data.cpu().numpy()))
                true_targets.append(list(targets.data.cpu().numpy()))

                users = items = targets = None
                del [users, items, targets]

        recommendations = np.array(sum(recommendations, []))
        true_targets = np.array(sum(true_targets, []))
        rmse = sqrt(mean_squared_error(recommendations, true_targets))
        mae = mean_absolute_error(recommendations, true_targets)

        dataLoader = recommendations = true_targets = None
        del [dataLoader, recommendations, true_targets]

        return rmse, mae

    def drawcurve(self, tr, v, ts, id, epoch, name):

        """
        Plot function to plot the rmse and mae curves of train, test and validation samples.
        :param tr: train values
        :param v: validation values
        :param ts:test values
        :param id: id
        :param epoch: epoch
        :param name: either rmse or mae
        :return: saves plot in configured location.
        """
        star = mpath.Path.unit_regular_star(6)
        circle = mpath.Path.unit_circle()
        # concatenate the circle with an internal cutout of the star
        verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
        codes = np.concatenate([circle.codes, star.codes])
        cut_star = mpath.Path(verts, codes)

        tr = np.array(tr).flatten()
        v = np.array(v).flatten()
        ts = np.array(ts).flatten()

        plt.figure(id)
        plt.plot(tr, marker=cut_star, markersize=10, color='blue')
        plt.plot(v, marker=cut_star, markersize=10, color='green')
        plt.plot(ts, marker=cut_star, markersize=10, color='red')
        plt.savefig(properties.getProperty('RESEARCH_PLOTS_SAVE_HERE') + '%s_after_%d_epoch.png' % (name, epoch), dpi=300)
        # plt.draw()
        # plt.pause(0.001)
        return 0
