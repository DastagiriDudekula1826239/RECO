import torch
import torch.nn as nn
import torch.nn.functional as F


class UserUserUpdater(nn.Module):

    def __init__(self, uv_updated_embeddings, user_to_users_social_adjacency, aggregator, user_embeddings, embeddig_dim, userFeatsUVFlag="False", cuda="cpu"):

        """
        updater class to get the convolved encodes of user latents from user-user graph
        :param uv_updated_embeddings: user latent features from user-item graph(for the better understanding of user opinions on item)
        :param user_to_users_social_adjacency: social graph
        :param aggregator: user-user aggregator classs object
        :param embeddig_dim: embedding dimension
        :param userFeatsUVFlag: flag to choose the neighbor embedding features from user latents of user-item graph
        :param cuda: device
        """
        super(UserUserUpdater, self).__init__()

        self.uv_updated_features = uv_updated_embeddings
        self.user_to_users_social_adjacency = user_to_users_social_adjacency
        self.aggregator = aggregator
        self.user_embeddings = user_embeddings
        self.embeddig_dim = embeddig_dim
        self.userFeatsUVFlag = userFeatsUVFlag
        self.device = cuda

        self.w1 = nn.Linear(2 * self.embeddig_dim, self.embeddig_dim)
        self.w2 = nn.Linear(self.embeddig_dim, self.embeddig_dim)

        self.w_cnvlvd = nn.Linear(2 * self.embeddig_dim, self.embeddig_dim)

    def forward(self, nodes):

        """
        2-hop attentive embedding propogation to learn the user latent factor representation.
        Which means, to learn the user latent, convolution will take next two successive children levels (without changing the importance of no two relations(parent-child): attention).
        self feature are considered from user latents of user-item network, for better feedback opinions.
        :param nodes: user nodes
        :return: convolved attentive user latent factor representations
        """
        sec_level_conlved = []

        for node in nodes:

            first_neighs = list(self.user_to_users_social_adjacency[int(node)])

            sec_neighs = []
            for neigh_node in first_neighs:
                sec_neighs.append(self.user_to_users_social_adjacency[int(neigh_node)])

            sec_neighs_aggregate_to_first_neighs_feats = self.aggregator.forward(first_neighs, sec_neighs, self.userFeatsUVFlag, False)

            # self_feats_first = self.uv_updated_features(torch.LongTensor(first_neighs).cpu().numpy()).to(self.device)
            self_feats_first = self.user_embeddings.weight[first_neighs]
            self_feats_first = self_feats_first

            first_neighs_sec_neighs_feats = torch.cat([self_feats_first, sec_neighs_aggregate_to_first_neighs_feats], dim=1)

            first_neighs_sec_neighs_feats = F.relu(self.w1(first_neighs_sec_neighs_feats))
            first_neighs_sec_neighs_feats = F.relu(self.w2(first_neighs_sec_neighs_feats))

            sec_level_conlved.append(first_neighs_sec_neighs_feats)

        parentnodes_convolved_with_sec_level_convolves = self.aggregator.forward(nodes, sec_level_conlved, self.userFeatsUVFlag, True)

        nodes_self_features = self.uv_updated_features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        nodes_self_features = nodes_self_features.t()                                                         #TODO

        convolved = torch.cat([nodes_self_features, parentnodes_convolved_with_sec_level_convolves], dim=1)
        convolved = F.relu(self.w_cnvlvd(convolved))

        return convolved