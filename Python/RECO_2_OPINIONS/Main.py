import torch
import torch.nn as nn
import torch.utils.data

import os
import time
import pickle
import numpy as np
from two_hop_opinions.UserUserUpdater import UserUserUpdater
from two_hop_opinions.UserUserAggregator import UserUserAggregator

from two_hop_opinions.UserItemOpinionConsider import UserItemOpinionConsider

from pyjavaproperties import Properties

from two_hop_opinions.GnnSocialRecommendationSystem import GnnSocialRecommendationSystem

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

def main():
    time_start = int(round(time.time() * 1000))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    # Dianping data loading

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_DIANPING'), properties.getProperty('USER_ITEM_GRAPH_DIANPING'))

    user_to_users_social_adjacency = data.getSocialGraph()
    user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getBipartiteGraphs(properties.getProperty('TEST_DIANPING'))

    trainUsers, trainItems, trainLables = data.dataframeToLists(properties.getProperty('TRAIN_DIANPING'), columns=['userId','itemId', 'rating'], sep=',')
    validationUsers, validationItems, validationLables = data.dataframeToLists(properties.getProperty('VALIDATION_DIANPING'), columns=['userId','itemId', 'rating'], sep=',')
    testUsers, testItems, testLables = data.dataframeToLists(properties.getProperty('TEST_DIANPING'), columns=['userId','itemId', 'rating'], sep=',')

    """

    """
    # Epinions data loading

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_EPINIONS'), properties.getProperty('USER_ITEM_GRAPH_EPINIONS'))

    user_to_users_social_adjacency, user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history, epinionsDF = data.getEpinionsSocialUVGraph(properties.getProperty('USERS_WITH_NO_ITEMS'))

    train, test, validation = data.trainTestValidationEpinions(epinionsDF)

    trainUsers, trainItems, trainLables = data.dataframeToListsEpinions(train)
    validationUsers, validationItems, validationLables = data.dataframeToListsEpinions(validation)
    testUsers, testItems, testLables = data.dataframeToListsEpinions(test)
    """

    """
    # Ciao data loading

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_CIAO'), properties.getProperty('USER_ITEM_GRAPH_CIAO'))

    user_to_users_social_adjacency = data.getCiaoSocialGraph()
    user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getCiaoBipartiteGraph(properties.getProperty('TEST_CIAO'))

    #train, test, validation = data.trainTestValidationCiao(ciaoDF)

    trainUsers, trainItems, trainLables = data.dataframeToLists(properties.getProperty('TRAIN_CIAO'), columns=['userId', 'itemId', 'rating'], sep=',')
    validationUsers, validationItems, validationLables = data.dataframeToLists(properties.getProperty('VALIDATION_CIAO'), columns=['userId', 'itemId', 'rating'], sep=',')
    testUsers, testItems, testLables = data.dataframeToLists(properties.getProperty('TEST_CIAO'), columns=['userId', 'itemId', 'rating'], sep=',')


    """

    # Free data loading

    user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history, trainUsers, trainItems, trainLables, testUsers, testItems, testLables, user_to_users_social_adjacency, ratings_list = pickle.load(
        open(properties.getProperty('SOCIAL_RECO_DATASET'), 'rb'))

    # Validation dataset hardcoded by taking 10% from train dataset
    validationUsers = trainUsers[12891:len(trainUsers)]
    validationItems = trainItems[12891:len(trainItems)]
    validationLables = trainLables[12891:len(trainLables)]

    train_Dataset = torch.utils.data.TensorDataset(torch.LongTensor(trainUsers), torch.LongTensor(trainItems),
                                                   torch.FloatTensor(trainLables))
    validation_Dataset = torch.utils.data.TensorDataset(torch.LongTensor(validationUsers),
                                                        torch.LongTensor(validationItems),
                                                        torch.FloatTensor(validationLables))
    test_Dataset = torch.utils.data.TensorDataset(torch.LongTensor(testUsers), torch.LongTensor(testItems),
                                                  torch.FloatTensor(testLables))

    """
    Used torchDataloader to process chunkwise data loading(to ease the handling of batch sampling)
    """
    train_DataLoader = torch.utils.data.DataLoader(train_Dataset, batch_size=int(properties.getProperty('CHUNK')),
                                                   shuffle=True)
    validation_DataLoader = torch.utils.data.DataLoader(validation_Dataset,
                                                        batch_size=int(properties.getProperty('CHUNK')), shuffle=True)
    test_DataLoader = torch.utils.data.DataLoader(test_Dataset, batch_size=int(properties.getProperty('CHUNK')),
                                                  shuffle=True)

    embeddig_dim = int(properties.getProperty('EMBEDD_DIM'))

    user_embeddings = nn.Embedding(user_to_items_history.__len__(), embeddig_dim).to(
        device)  # , dtype=torch.half, non_blocking=True
    item__embeddings = nn.Embedding(item_by_users_history.__len__(), embeddig_dim).to(device)
    # Dianping : int(properties.getProperty('RATINGS_COUNT_DIANPING'))
    # Epinions : int(properties.getProperty('RATINGS_COUNT_EPINIONS'))
    # Ciao : int(properties.getProperty('RATINGS_COUNT_CIAO'))
    # for testing data : ratings_list.__len__()
    rating__embeddings = nn.Embedding(ratings_list.__len__(), embeddig_dim).to(device)

    userFeatsUVFlag = properties.getProperty('LEARN_USER_FEATURES_FROM_UV_GRAPH')

    """
    to learn the user latents from user-item network and social network 
    """
    opinionT = UserItemOpinionConsider(user_to_items_history, user_ratings_to_items_history, user_embeddings,
                                       item__embeddings, rating__embeddings, embeddig_dim, cuda=device,
                                       user_latent_flag=True)

    user_user_aggregator = UserUserAggregator(lambda nodes: opinionT(nodes).t(), user_embeddings, embeddig_dim,
                                              cuda=device)
    user_user_updater = UserUserUpdater(lambda nodes: opinionT(nodes).t(), user_to_users_social_adjacency,
                                        user_user_aggregator, embeddig_dim, userFeatsUVFlag, cuda=device)

    """
    to read the item latent for user-item network
    """
    opinionF = UserItemOpinionConsider(item_by_users_history, item_ratings_by_users_history, user_embeddings,
                                       item__embeddings, rating__embeddings, embeddig_dim, cuda=device,
                                       user_latent_flag=False)

    """
    base model
    """
    reco = GnnSocialRecommendationSystem(opinionT, user_user_updater, opinionF).to(device)
    optimizer = torch.optim.RMSprop(reco.parameters(), lr=float(properties.getProperty('LEARNING_RATE')),
                                    alpha=0.9)  # TODO; try look for some other

    train_rmse_list, valid_rmse_list, train_mae_list, valid_mae_list, test_rmse_list, test_mae_list = ([] for _ in
                                                                                                       range(6))

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
        if best_rmse > valid_rmse:  # TODO: check for best approach for early stop concept (advanced)
            best_rmse = valid_rmse
            best_mae = valid_mae
            calls = 0
        else:
            calls += 1
        if calls > max_calls:
            break
    print("Total Time taken to complete the recommendation: %d secs" % (
                ((int(round(time.time() * 1000))) - time_start) / 1000))


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