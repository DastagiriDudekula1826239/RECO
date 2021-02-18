import torch
import torch.nn as nn
import torch.nn.functional as F


class UserUserAggregator(nn.Module):

    def __init__(self, uv_updated_embeddings, user_embeddings, embeddig_dim, cuda="cpu"):

        """
        user-user aggregator class to perform the aggregation of user with interacted nodes.
        :param uv_updated_embeddings: user latent features from user-item graph(for the better understanding of user opinions on item)
        :param user_embeddings: user initial free embeddings
        :param embeddig_dim: embedding dimension
        :param cuda: device
        """
        super(UserUserAggregator, self).__init__()

        self.uv_updated_features = uv_updated_embeddings
        self.device = cuda
        self.user_embeddings = user_embeddings
        self.embeddig_dim = embeddig_dim

        self.w_a1 = nn.Linear(self.embeddig_dim * 2, self.embeddig_dim)
        self.w_a2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)
        self.w_a3 = nn.Linear(self.embeddig_dim, 1)

    def forward(self, nodes, neighbours, userFeatsUVFlag="False", last_layer=False):

        """
        The parent users aggregated with neighboring users attentively, to learn better user latent factor representation.
        (by the assumption that; user's preferences may depend directly/indectly by peers)
        :param nodes: user parent nodes
        :param neighbours: user neighboring nodes for social graph
        :param last_layer: flag to choose aggragation for last layer or not
        :param userFeatsUVFlag: flag to choose the neighbor embedding features from user latents of user-item graph
        :return: user attentive aggregated embedding matrices
        """
        aggregated_matrix = torch.empty(len(nodes), self.embeddig_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            neighbours_adj = neighbours[i]

            if last_layer:
                user_embedding_reprs = self.user_embeddings.weight[nodes[i]]    # here no need to read from user latents of U-I, because it will done in self featuring area in updater class
                attentive_weighted_mx = self.attenctionMechanism(neighbours_adj, user_embedding_reprs, len(neighbours_adj))
                attentive_weighted_mx = torch.mm(neighbours_adj.t(), attentive_weighted_mx).t()
            else:
                if userFeatsUVFlag.strip().lower() in ("yes", "true", "t", "1"):
                    neighs_embedding_features = self.uv_updated_features(torch.LongTensor(list(neighbours_adj)).to(self.device))
                    neighs_embedding_features = torch.t(neighs_embedding_features)
                else:
                    neighs_embedding_features = self.user_embeddings.weight[list(neighbours_adj)]

                user_embedding_reprs = self.user_embeddings.weight[nodes[i]]

                attentive_weighted_mx = self.attenctionMechanism(neighs_embedding_features, user_embedding_reprs, len(neighbours_adj))
                attentive_weighted_mx = torch.mm(neighs_embedding_features.t(), attentive_weighted_mx).t()

            aggregated_matrix[i] = attentive_weighted_mx

        return aggregated_matrix

    def attenctionMechanism(self, neighs_embedding_features, user_embedding_reprs, count):

        """
        Attenction mechanism to maintain the unique relationships.
        :param neighs_embedding_features: neighbor embeddings
        :param user_embedding_reprs: user embeddings
        :param count: number of neighbors
        :return: weighted attenction
        """
        user_embedding_reprs = user_embedding_reprs.repeat(count, 1)
        u_u_update = torch.cat((neighs_embedding_features, user_embedding_reprs), 1)
        u_u_update = F.relu(self.w_a1(u_u_update))
        u_u_update = F.dropout(u_u_update, training=self.training)
        u_u_update = F.relu(self.w_a2(u_u_update))
        u_u_update = F.dropout(u_u_update, training=self.training)
        u_u_update = self.w_a3(u_u_update)
        attentive_weighted_mx = F.softmax(u_u_update, dim=0)

        return attentive_weighted_mx
