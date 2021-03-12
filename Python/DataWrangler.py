import pandas as pd
import numpy as np


class DataWrangler():

    def __init__(self, socialFilePath, uvFilePath):

        self.socialFilePath = socialFilePath
        self.uvFilePath = uvFilePath

    def getSocialGraph(self):

        rawSocialGraph = open(self.socialFilePath, 'r')
        socialLines = rawSocialGraph.readlines()

        socialG = {}

        for line in socialLines:
            ls = line.strip().split("|")
            key = int(ls[0].strip())
            vals = [int(x.strip()) for x in ls[1].strip().split(' ')]
            if key in socialG:
                socialG[key].update(set(vals))
            else:
                socialG[key] = set(vals)
            ls = None
            vals = None
            del ls, vals

        rawSocialGraph = None
        socialLines = None
        del rawSocialGraph, socialLines

        return socialG

    def getBipartiteGraphs(self):

        rawUVGraph = open(self.uvFilePath, 'r')
        uvLines = rawUVGraph.readlines()

        userItemsGraph = {}
        userItemsRatings = {}

        itemUsersGraph = {}
        itemUsersRatings = {}

        for line in uvLines:
            ls = line.strip().split("|")
            uId = int(ls[0].strip())
            vId = int(ls[1].strip())
            r = int(ls[2].strip())

            if uId in userItemsGraph:
                userItemsGraph[uId].append(vId)
                userItemsRatings[uId].append(r)
            else:
                userItemsGraph[uId] = list([vId])
                userItemsRatings[uId] = list([r])

            if vId in itemUsersGraph:
                itemUsersGraph[vId].append(uId)
                itemUsersRatings[vId].append(r)
            else:
                itemUsersGraph[vId] = list([uId])
                itemUsersRatings[vId] = list([r])

            ls = None
            vals = None
            del ls, vals

        rawUVGraph = None
        uvLines = None
        del rawUVGraph, uvLines

        return userItemsGraph, userItemsRatings, itemUsersGraph, itemUsersRatings

    def dataframeToLists(self, filePath, columns=['userId','itemId', 'rating'], sep=','):

        """
        :param filePath:
        :return:
        """
        df = pd.read_csv(filePath, sep=sep)

        usersList = df[columns[0]].tolist()
        itemsList = df[columns[1]].tolist()
        ratingsList = df[columns[2]].tolist()

        df = None
        del df

        return usersList, itemsList, ratingsList


    def trainTestValDatasetsPreparationDianping(self):

        df = pd.read_csv('/home/giri/DemoData/Dianping/csvVersion/rating.csv')
        df = df.sample(frac=1).reset_index(drop=True)

        df = df.drop(["date"], 1)

        tr = df[:1934700]
        ts = df[1934700:]
        val = tr.iloc[np.random.choice(tr.index, 193470, replace=False)]

        #print(len(tr))
        #print(len(ts))
        #print(len(val))

        tr.to_csv("/home/giri/DemoData/Dianping/csvVersion/train_90.csv", index=False)
        val.to_csv("/home/giri/DemoData/Dianping/csvVersion/validation_10.csv", index=False)
        ts.to_csv("/home/giri/DemoData/Dianping/csvVersion/test_10.csv", index=False)

        df = None
        tr = None
        ts = None
        val = None
        del df, tr, ts, val

        return 0

    def getCiaoData(self):

        social = pd.read_csv(self.socialFilePath)
        ui = pd.read_csv(self.uvFilePath)

        socialG = {}

        for index, row in social.iterrows():
            uId = row['userId']
            fId = row['friendId']

            if uId in socialG:
                socialG[uId].update(set([fId]))
            else:
                socialG[uId] = set([fId])

            if fId in socialG:
                socialG[fId].update(set([uId]))
            else:
                socialG[fId] = set([uId])

        print(len(socialG))

        userItemsGraph = {}
        userItemsRatings = {}

        itemUsersGraph = {}
        itemUsersRatings = {}

        for index, row in ui.iterrows():
            uId = row['userId']
            vId = row['itemId']
            r = row['rating']

            if uId in userItemsGraph:
                userItemsGraph[uId].append(vId)
                userItemsRatings[uId].append(r)
            else:
                userItemsGraph[uId] = list([vId])
                userItemsRatings[uId] = list([r])

            if vId in itemUsersGraph:
                itemUsersGraph[vId].append(uId)
                itemUsersRatings[vId].append(r)
            else:
                itemUsersGraph[vId] = list([uId])
                itemUsersRatings[vId] = list([r])

        print(len(userItemsGraph))

        #print(len(list(set(userItemsGraph.keys()).difference(set(socialG.keys())))))
        #print(list(set(userItemsGraph.keys()).difference(set(socialG.keys()))))

        #print(len(itemUsersGraph))

        return socialG, userItemsGraph, userItemsRatings, itemUsersGraph, itemUsersRatings

    def getEpinions(self):

        noItemLink1 = pd.read_csv('/home/giri/DemoData/Epinions/NoItemLinks_Final_.csv')

        # noItemLink2 = pd.read_csv('/home/giri/DemoData/Epinions/NoItemLinks.csv')

        rawSocialGraph = open(self.socialFilePath, 'r')
        socialLines = rawSocialGraph.readlines()

        noitemLinksList = set(noItemLink1['userId'].tolist())

        socialG = {}

        for line in socialLines:
            ls = line.strip().split(' ')
            key = int(ls[0].strip())
            val = int(ls[1].strip())

            if (key in noitemLinksList) or (val in noitemLinksList):
                pass
            else:
                if key in socialG:
                    socialG[key].update(set([val]))
                else:
                    socialG[key] = set([val])

                if val in socialG:
                    socialG[val].update(set([key]))
                else:
                    socialG[val] = set([key])

            ls = None
            val = None
            del ls, val

        rawSocialGraph = None
        socialLines = None
        del rawSocialGraph, socialLines

        """
        sKeys = list(socialG.keys())

        sDF = pd.DataFrame(sKeys, columns=['userId'])
        sDF.to_csv('/home/giri/DemoData/Epinions/socialUsers.csv')

        """

        # print("Social: ",len(sKeys))

        rawUVGraph = open(self.uvFilePath, 'r')
        uvLines = rawUVGraph.readlines()

        userItemsGraph = {}
        userItemsRatings = {}

        itemUsersGraph = {}
        itemUsersRatings = {}

        for line in uvLines:
            ls = line.strip().split(' ')
            uId = int(ls[0].strip())
            vId = int(ls[1].strip())
            r = int(ls[2].strip())

            """
            if uId in userItemsGraph:
                userItemsGraph[uId].append(vId)
                userItemsRatings[uId].append(r)
            else:
                userItemsGraph[uId] = list([vId])
                userItemsRatings[uId] = list([r])

            if vId in itemUsersGraph:
                itemUsersGraph[vId].append(uId)
                itemUsersRatings[vId].append(r)
            else:
                itemUsersGraph[vId] = list([uId])
                itemUsersRatings[vId] = list([r])
            """
            if uId not in socialG:
                pass
            elif uId in noitemLinksList:
                pass
            else:
                if uId in userItemsGraph:
                    userItemsGraph[uId].append(vId)
                    userItemsRatings[uId].append(r)
                else:
                    userItemsGraph[uId] = list([vId])
                    userItemsRatings[uId] = list([r])

                if vId in itemUsersGraph:
                    itemUsersGraph[vId].append(uId)
                    itemUsersRatings[vId].append(r)
                else:
                    itemUsersGraph[vId] = list([uId])
                    itemUsersRatings[vId] = list([r])

            ls = None
            vals = None
            del ls, vals

        rawUVGraph = None
        uvLines = None
        del rawUVGraph, uvLines

        # bKeys = list(userItemsGraph.keys())

        # print(len(list(set(sKeys).difference(set(bKeys)))))
        # print(list(set(sKeys).difference(set(bKeys))))
        """

        nlist = list(set(sKeys).difference(set(bKeys)))

        f = list(noitemLinksList) + nlist

        fin = pd.DataFrame(f, columns=['userId'])
        fin.to_csv('/home/giri/DemoData/Epinions/NoItemLinks_Final_.csv')
        """
        # print(userItemsGraph)
        # print("UI: ",len(bKeys))
        # print(userItemsRatings)
        # print("Items: ",len(itemUsersGraph))
        # print(itemUsersRatings)

        return socialG, userItemsGraph, userItemsRatings, itemUsersGraph, itemUsersRatings

    def trainTestValDatasetsPreparationEpinions(self):

        df = pd.read_csv('/home/giri/DemoData/Epinions/csvVersion/ratings_data.csv', sep=' ')

        noItemLinks = pd.read_csv('/home/giri/DemoData/Epinions/NoItemLinks_Final_.csv')
        socialUserlinks = pd.read_csv('/home/giri/DemoData/Epinions/socialUsers.csv')

        noItemL = noItemLinks['userId'].tolist()
        sL = socialUserlinks['userId'].tolist()

        df = df[df['userId'].isin(noItemL) == False]
        df = df[df['userId'].isin(sL) == True]

        df = df.sample(frac=1).reset_index(drop=True)

        size = int(len(df) * 0.90)
        train, test = df[0:size], df[size:len(df)]
        validation = train.iloc[np.random.choice(train.index, int(len(train) * 0.10), replace=False)]

        train.to_csv("/home/giri/DemoData/Epinions/csvVersion/train_90.csv", index=False)
        validation.to_csv("/home/giri/DemoData/Epinions/csvVersion/validation_10.csv", index=False)
        test.to_csv("/home/giri/DemoData/Epinions/csvVersion/test_10.csv", index=False)

        df = None
        train = None
        test = None
        validation = None
        del df, train, test, validation

        return 0



def trainTestValDatasetsPreparationCiao():

    df = pd.read_csv('/home/giri/DemoData/ciao/wholeTrain/train_90_10.csv')
    df = df.sample(frac=1).reset_index(drop=True)

    tr = df[:9000]
    ts = df[9000:]
    val = tr.iloc[np.random.choice(tr.index, 1000, replace=False)]

    # print(len(tr))
    # print(len(ts))
    # print(len(val))

    tr.to_csv("/home/giri/DemoData/ciao/train_90.csv", index=False)
    val.to_csv("/home/giri/DemoData/ciao/validation_10.csv", index=False)
    ts.to_csv("/home/giri/DemoData/ciao/test_10.csv", index=False)

    df = None
    tr = None
    ts = None
    val = None
    del df, tr, ts, val

    return 0



#if __name__ == "__main__":

    """

    from pyjavaproperties import Properties

    properties = Properties()
    properties.load(open('/home/giri/global.properties'))

    data = DataWrangler(properties.getProperty('SOCIAL_GRAPH_EPINIONS'),
                        properties.getProperty('USER_ITEM_GRAPH_EPINIONS'))

    user_to_users_social_adjacency, user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getEpinions()

    trainUsers, trainItems, trainLables = data.dataframeToLists(properties.getProperty('TRAIN_EPINIONS'), columns=['userId', 'itemId', 'ratingValue'], sep=',')

    print(len(user_to_users_social_adjacency), len(user_to_items_history))
    print(len(trainItems), len(trainUsers), len(trainLables))
    """

    """
    rawSocialGraph = open('/home/giri/DemoData/ciao/trustnetwork.csv', 'r')
    socialLines = rawSocialGraph.readlines()

    socialG = {}
    i =1

    for line in socialLines:
        if i==1:
            pass
        else:
            ls = line.strip().split(',')
            key = int(ls[0].strip())
            vals = [int(ls[1].strip())]
            if key in socialG:
                socialG[key].update(set(vals))
            else:
                socialG[key] = set(vals)
        i = 2
    print(socialG)
    """
    """
    data = DataWrangler('/home/giri/DemoData/ciao/trustnetwork.csv', '/home/giri/DemoData/ciao/rating.csv')
    user_to_users_social_adjacency, user_to_items_history, user_ratings_to_items_history, item_by_users_history, item_ratings_by_users_history = data.getCiaoData()
    trainTestValDatasetsPreparationCiao()
    print(user_to_users_social_adjacency[1781])
    print(user_to_users_social_adjacency[7300])
    """

