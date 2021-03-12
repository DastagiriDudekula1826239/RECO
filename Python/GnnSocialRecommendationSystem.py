import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from tesi.DataWrangler import DataWrangler

from tesi.UserItemUpdater import UserItemUpdater
from tesi.UserItemAggregator import UserItemAggregator
from tesi.UserUserUpdater import UserUserUpdater
from tesi.UserUserAggregator import UserUserAggregator

from tesi.UserItemOpinionConsider import UserItemOpinionConsider

from pyjavaproperties import Properties
properties = Properties()
properties.load(open('/home/giri/global.properties'))

"""

Title: RECO (deep graph neural networks social recommendation engine)
Course: Thesis
Professor: Simone Scardapane
Student: Dastagiri Dudekula 1826239
University: Sapienza University of Rome
Dept: M.Sc. in Data Science

Project Keywords: Recommendation system, graph neural networks (pytorch)
"""


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


def main():
    time_start = int(round(time.time() * 1000))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    # Dianping data loading

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_DIANPING'), properties.getProperty('USER_ITEM_GRAPH_DIANPING'))

    user_to_users_social_adjacency = data.getSocialGraph()
    user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getBipartiteGraphs()

    trainUsers, trainItems, trainLables = data.dataframeToLists(properties.getProperty('TRAIN_DIANPING'), columns=['userId','itemId', 'rating'], sep=',')
    validationUsers, validationItems, validationLables = data.dataframeToLists(properties.getProperty('VALIDATION_DIANPING'), columns=['userId','itemId', 'rating'], sep=',')
    testUsers, testItems, testLables = data.dataframeToLists(properties.getProperty('TEST_DIANPING'), columns=['userId','itemId', 'rating'], sep=',')
    """

    # Epinions data loading

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_EPINIONS'), properties.getProperty('USER_ITEM_GRAPH_EPINIONS'))

    user_to_users_social_adjacency, user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getEpinions()

    trainUsers, trainItems, trainLables = data.dataframeToLists(properties.getProperty('TRAIN_EPINIONS'), columns=['userId','itemId', 'ratingValue'], sep=',')
    validationUsers, validationItems, validationLables = data.dataframeToLists(properties.getProperty('VALIDATION_EPINIONS'), columns=['userId','itemId', 'ratingValue'], sep=',')
    testUsers, testItems, testLables = data.dataframeToLists(properties.getProperty('TEST_EPINIONS'), columns=['userId','itemId', 'ratingValue'], sep=',')


    """
    # Ciao data loading

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_CIAO'), properties.getProperty('USER_ITEM_GRAPH_CIAO'))
    user_to_users_social_adjacency, user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getCiaoData()

    trainUsers, trainItems, trainLables = data.dataframeToLists(properties.getProperty('TRAIN_CIAO'), columns=['userId','itemId', 'rating'], sep=',')
    validationUsers, validationItems, validationLables = data.dataframeToLists(properties.getProperty('VALIDATION_CIAO'), columns=['userId','itemId', 'rating'], sep=',')
    testUsers, testItems, testLables = data.dataframeToLists(properties.getProperty('TEST_CIAO'), columns=['userId','itemId', 'rating'], sep=',')
    """

    """
    # Free data loading

    user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history, trainUsers, trainItems, trainLables, testUsers, testItems, testLables, user_to_users_social_adjacency, ratings_list = pickle.load(open(properties.getProperty('SOCIAL_RECO_DATASET'), 'rb'))

    # Validation dataset hardcoded by taking 10% from train dataset
    validationUsers = trainUsers[12891:len(trainUsers)]
    validationItems = trainItems[12891:len(trainItems)]
    validationLables = trainLables[12891:len(trainLables)]
    
    """

    train_Dataset = torch.utils.data.TensorDataset(torch.LongTensor(trainUsers), torch.LongTensor(trainItems), torch.FloatTensor(trainLables))
    validation_Dataset = torch.utils.data.TensorDataset(torch.LongTensor(validationUsers), torch.LongTensor(validationItems), torch.FloatTensor(validationLables))
    test_Dataset = torch.utils.data.TensorDataset(torch.LongTensor(testUsers), torch.LongTensor(testItems), torch.FloatTensor(testLables))

    """
    Used torchDataloader to process chunkwise data loading(to ease the handling of batch sampling)
    """
    train_DataLoader = torch.utils.data.DataLoader(train_Dataset, batch_size = int(properties.getProperty('CHUNK')), shuffle = True)
    validation_DataLoader = torch.utils.data.DataLoader(validation_Dataset, batch_size = int(properties.getProperty('CHUNK')), shuffle = True)
    test_DataLoader = torch.utils.data.DataLoader(test_Dataset, batch_size = int(properties.getProperty('CHUNK')), shuffle = True)

    embeddig_dim = int(properties.getProperty('EMBEDD_DIM'))

    user_embeddings = nn.Embedding(50000, embeddig_dim).to(device, dtype=torch.half, non_blocking=True)
    item__embeddings = nn.Embedding(140000, embeddig_dim).to(device)
    #Dianping : int(properties.getProperty('RATINGS_COUNT_DIANPING'))
    #Epinions : int(properties.getProperty('RATINGS_COUNT_EPINIONS'))
    #Ciao : int(properties.getProperty('RATINGS_COUNT_CIAO'))
    #for testing data : ratings_list.__len__()
    rating__embeddings = nn.Embedding(6, embeddig_dim).to(device)

    userFeatsUVFlag = properties.getProperty('LEARN_USER_FEATURES_FROM_UV_GRAPH')

    """
    to learn the user latents from user-item network and social network 
    """
    opinionT = UserItemOpinionConsider(user_to_items_history, user_ratings_to_items_history, user_embeddings, item__embeddings, rating__embeddings, embeddig_dim, cuda=device, user_latent_flag=True)

    user_user_aggregator = UserUserAggregator(lambda nodes: opinionT(nodes).t(), user_embeddings, embeddig_dim, cuda=device)
    user_user_updater = UserUserUpdater(lambda nodes: opinionT(nodes).t(), user_to_users_social_adjacency, user_user_aggregator, embeddig_dim, userFeatsUVFlag, cuda=device)

    """
    to read the item latent for user-item network
    """
    opinionF = UserItemOpinionConsider(item_by_users_history, item_ratings_by_users_history, user_embeddings, item__embeddings, rating__embeddings, embeddig_dim, cuda=device, user_latent_flag=False)

    """
    base model
    """
    reco = GnnSocialRecommendationSystem(opinionT, user_user_updater, opinionF).to(device)
    optimizer = torch.optim.RMSprop(reco.parameters(), lr=float(properties.getProperty('LEARNING_RATE')), alpha=0.9)   # TODO; try look for some other

    train_rmse_list, valid_rmse_list, train_mae_list, valid_mae_list, test_rmse_list, test_mae_list = ([] for _ in range(6))

    best_rmse = np.array(float(properties.getProperty('BEST_RMSE')))
    best_mae = np.array(float(properties.getProperty('BEST_MAE')))
    max_calls = int(properties.getProperty('CALLS_TO_STOP'))

    for epoch in range(1, int(properties.getProperty('EPOCHS')) + 1):
        step_time = int(round(time.time() * 1000))
        print("Epoch %d/%d :: training..." % (epoch, 75))

        # Train
        tr_rmse, tr_mae = reco.fit(train_DataLoader, optimizer, device, best_rmse, best_mae)
        # Validation
        valid_rmse, valid_mae = reco.evaluate(device, validation_DataLoader)
        # Test
        ts_rmse, ts_mae = reco.evaluate(device, test_DataLoader)

        train_rmse_list.append(tr_rmse)
        train_mae_list.append(tr_mae)
        valid_rmse_list.append(valid_rmse)
        valid_mae_list.append(valid_mae)
        test_rmse_list.append(ts_rmse)
        test_mae_list.append(ts_mae)

        print(" Train:: rmse: %.4f, mae:%.4f " % (np.array(tr_rmse), np.array(tr_mae)))
        print(" Valid:: rmse: %.4f, mae:%.4f " % (np.array(valid_rmse), np.array(valid_mae)))
        print(" Test::: rmse: %.4f, mae:%.4f " % (np.array(ts_rmse), np.array(ts_mae)))

        print("%d secs" % (((int(round(time.time() * 1000))) - step_time) / 1000))

        reco.drawcurve(train_rmse_list, valid_rmse_list, test_rmse_list, 1, epoch, "RMSE")
        reco.drawcurve(train_mae_list, valid_mae_list, test_mae_list, 2, epoch, "MAE")

        """
        to stop the model processing (basic)
        """
        if best_rmse > valid_rmse:    # TODO: check for best approach for early stop concept (advanced)
            best_rmse = valid_rmse
            best_mae = valid_mae
            calls = 0
        else:
            calls += 1
        if calls > max_calls:
            break
    print("Total Time taken to complete the recommendation: %d secs" % (((int(round(time.time() * 1000))) - time_start) / 1000))


if __name__ == "__main__":
    main()



"""
reading parameters from local configuration file /home/giri/global.properties


SOCIAL_RECO_DATASET=/home/giri/DemoData/Research/socialRECO_dataset.pickle
CHUNK=256
EMBEDD_DIM=64
LEARNING_RATE=0.001
EPOCHS=75
LEARN_USER_FEATURES_FROM_UV_GRAPH=True
RESEARCH_PLOTS_SAVE_HERE=/home/giri/Thesis/research/
BEST_RMSE=9999.0
BEST_MAE=9999.0
CALLS_TO_STOP=6

"""