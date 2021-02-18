import torch
import torch.nn as nn
import torch.nn.functional as F


class UserItemAggregator(nn.Module):

    def __init__(self, user_embeddings, item__embeddings, rating__embeddings, embeddig_dim, cuda="cpu", user_latent_flag=True):

        """
        user-item Aggregator class to peroform the user aggregation or item aggregation based on call
        :param user_embeddings: user initial free embeddings
        :param item__embeddings: item initial free embeddings
        :param rating__embeddings: rating scores initial free embeddings
        :param embeddig_dim:  embedding dimension
        :param cuda: device
        :param user_latent_flag: flag to choose user aggregation or item
        """
        super(UserItemAggregator, self).__init__()

        self.user_latent_flag = user_latent_flag
        self.user_embeddings = user_embeddings
        self.item__embeddings = item__embeddings
        self.rating__embeddings = rating__embeddings
        self.device = cuda
        self.embeddig_dim = embeddig_dim

        self.w1 = nn.Linear(self.embeddig_dim * 2, self.embeddig_dim)
        self.w2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)

        self.w_a1 = nn.Linear(self.embeddig_dim * 2, self.embeddig_dim)
        self.w_a2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)
        self.w_a3 = nn.Linear(self.embeddig_dim, 1)

    def forward(self, nodes, uv_adjacency, ratings):

        """
        forward function where the user/item attentive aggregation perform, based on the user_latent_flag.
        If True, the aggregation will done for user to build the user latent, otherwise item aggregation.
        :param nodes:  user or item nodes to aggregate
        :param uv_adjacency: user or item adjacency nodes
        :param ratings: corresponding rating scores
        :return: user or item aggregated attentive embeddings
        """
        aggregated_matrix = torch.empty(len(uv_adjacency), self.embeddig_dim, dtype=torch.float).to(self.device)

        for i in range(len(uv_adjacency)):
            users_or_items = uv_adjacency[i]
            rating_scores = ratings[i]

            if self.user_latent_flag == True:
                uv_updated_embeddings = self.item__embeddings.weight[users_or_items]
                uv_updated_reprs = self.user_embeddings.weight[nodes[i]]
            else:
                uv_updated_embeddings = self.user_embeddings.weight[users_or_items]
                uv_updated_reprs = self.item__embeddings.weight[nodes[i]]

            rating_updated_embeddings = self.rating__embeddings.weight[rating_scores]
            uv_r = torch.cat((uv_updated_embeddings, rating_updated_embeddings), 1)
            uv_r = F.relu(self.w1(uv_r))
            uv_r = F.relu(self.w2(uv_r))

            # Attention mechanism
            uv_updated_reprs = uv_updated_reprs.repeat(len(users_or_items), 1)
            uv_r_update = torch.cat((uv_r, uv_updated_reprs), 1)
            uv_r_update = F.relu(self.w_a1(uv_r_update))
            uv_r_update = F.dropout(uv_r_update, training=self.training)
            uv_r_update = F.relu(self.w_a2(uv_r_update))
            uv_r_update = F.dropout(uv_r_update, training=self.training)
            uv_r_update = self.w_a3(uv_r_update)
            attentive_weighted_mx = F.softmax(uv_r_update, dim=0)

            aggregated_matrix[i] = torch.mm(uv_r.t(), attentive_weighted_mx).t()

        return aggregated_matrix
