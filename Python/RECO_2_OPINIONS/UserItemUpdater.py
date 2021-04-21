import torch
import torch.nn as nn
import torch.nn.functional as F


class UserItemUpdater(nn.Module):

    def __init__(self, user_item_history, rating_history, aggregator, uv_embeddings, embeddig_dim):

        """
        updater class to get the convolved encodes of user latent and item latent from user-item graph
        :param user_item_history: user to items / item to users interactoins
        :param rating_history: rating sccores batch
        :param aggregator: userItemAggregator object to do aggregation
        :param uv_embeddings: user or item initial free embeddings
        :param embeddig_dim: embedding dimension
        """
        super(UserItemUpdater, self).__init__()

        self.uv_embedding_features = uv_embeddings
        self.uv_history = user_item_history
        self.ratings_history = rating_history
        self.aggregator = aggregator
        self.embeddig_dim = embeddig_dim

        self.w1 = nn.Linear(2 * self.embeddig_dim, self.embeddig_dim)

    def forward(self, nodes):

        """
        the neighbor aggregations concatenated with self embeddings of nodes to form the final latent factor representation
        :param nodes: user or item nodes to encode
        :return: convolved attentive user latent or item latent factor
        """
        uv_adjacency = []
        ratings = []
        for node in nodes:
            uv_adjacency.append(self.uv_history[int(node)])
            ratings.append(self.ratings_history[int(node)])

        uv_attentive_features = self.aggregator.forward(nodes, uv_adjacency, ratings)
        self_feats = self.uv_embedding_features.weight[nodes]

        convolved = torch.cat([self_feats, uv_attentive_features], dim=1)
        convolved = F.relu(self.w1(convolved))

        return convolved