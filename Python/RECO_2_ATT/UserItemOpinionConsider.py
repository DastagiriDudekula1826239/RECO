import torch
import torch.nn as nn
import torch.nn.functional as F

from two_hop_att.UserItemUpdater import UserItemUpdater
from two_hop_att.UserItemAggregator import UserItemAggregator


class UserItemOpinionConsider(nn.Module):

    def __init__(self, user_item_history, user_item_ratings_history, user_embeddings, item__embeddings, rating__embeddings, embeddig_dim, cuda="cpu", user_latent_flag=True):

        """

        :param user_to_items_history:
        :param user_ratings_to_items_history:
        :param user_embeddings:
        :param item__embeddings:
        :param rating__embeddings:
        :param embeddig_dim:
        :param cuda:
        :param user_latent_flag:
        """
        super(UserItemOpinionConsider, self).__init__()

        self.device = cuda

        self.uv_aggregator = UserItemAggregator(user_embeddings, item__embeddings, rating__embeddings, embeddig_dim, self.device, user_latent_flag)

        if user_latent_flag:
            self.uv_updated_embeddings = UserItemUpdater(user_item_history, user_item_ratings_history, self.uv_aggregator, user_embeddings, embeddig_dim)
        else:
            self.uv_updated_embeddings = UserItemUpdater(user_item_history, user_item_ratings_history, self.uv_aggregator, item__embeddings, embeddig_dim)

    def forward(self, nodes):

        return self.uv_updated_embeddings.forward(nodes)