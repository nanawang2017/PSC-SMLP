"""
Created on Nov 17, 2018
从user_cluster.pkl得到进行谱聚类后的用户，本py文件主要完成的就是对一类user构造得到训练集和测试集
在主函数中修改dataset_name，
当选择dataset_name = 'Yelp'，记得修改self.users_sim = user_cluster[0]
dataset_name =Gowalla，记得修改self.users_sim = user_cluster[1]（在109行）

May 28,2019 改动，进一步筛选用户 生成训练集和测试集存放于final_data
train.dat test.dat

"""
import os
import numpy as np
import pickle
import sys


class Dataset(object):

    def __init__(self, dataset_name, num_cluster, negative=10, split=0.7):
        """
        train_data:是用户签到训练集 全为用户真实签到的数据构成 用于训练模型
        test_data:模型输入进行评估的数据集
        user_test_poi：是进行评估时使用的用户测试集 用于求命中率
        :param dataset_name:
        :param prefix:
        :param negative:
        :param split:
        """
        self.users_sim = []

        self.train_data = {}
        self.test_data = {}

        self.user_enum, self.spot_enum = 0, 0

        self.num_cluster = num_cluster
        self.negative = negative
        self.split = split

        # inter_path => user_poi and user_cluster
        # data_path => train.dat test.dat
        self.project_path = os.path.dirname(os.getcwd())
        self.dataset_path = self.project_path + '/data/' + dataset_name
        self.inter_path = self.dataset_path + '/inter/'
        self.data_path = self.dataset_path + '/final_data/'

        # the data we used
        self.checkins_dict_path = self.inter_path + 'user_poi.pkl'
        self.social_dict_path = self.inter_path + 'user_friend.pkl'
        self.user_cluster_path = self.inter_path + 'user_cluster.pkl'

        # the data we generated
        self.train_path = self.data_path + 'train.pkl'
        self.test_path = self.data_path + 'test.pkl'
        # inter_path to check whether train.dat and test.data were generated
        self.inter_pkl = self.data_path + 'inter.pkl'

        # self.generate()
        self.getCrossLabels()

    def generate(self):
        writeToFile = False
        try:
            f = open(self.inter_pkl, 'rb')
            f.close()
        except IOError:
            writeToFile = True
        if not writeToFile:
            with open(self.inter_pkl, 'rb') as f:
                inter_data = pickle.load(f)
            with open(self.train_path, 'rb') as f:
                self.train_data = pickle.load(f)
            with open(self.test_path, 'rb') as f:
                self.test_data = pickle.load(f)

            self.user_enum = inter_data['user_enum']
            self.spot_enum = inter_data['spot_enum']
            print(str(len(self.user_enum)) + ' users in enum loaded')
            print(str(len(self.spot_enum)) + ' spots in enum loaded')
            print(str(len(self.train_data)) + ' training labels loaded')
            print(str(len(self.test_data)) + ' test set loaded')

        else:
            inter_data = {}
            self.train_data['user'] = []
            self.train_data['spot'] = []
            self.train_data['label'] = []
            # self.test_data['user'] = []
            # self.test_data['spot'] = []
            # self.test_data['label'] = []

            self.user_enum, self.spot_enum = self.getCrossLabels()
            print('#users:', len(self.user_enum), '#POIs', len(self.spot_enum))
            inter_data['user_enum'] = self.user_enum
            inter_data['spot_enum'] = self.spot_enum

            with open(self.inter_path, 'wb') as f:
                pickle.dump(inter_data, f)

            print('Writing ' + str(len(self.train_data)) + ' training data to file')
            with open(self.train_path, 'wb') as f:
                pickle.dump(self.train_data, f)

            print('Writing ' + str(len(self.test_data)) + ' testing data to file')
            with open(self.test_path, 'wb') as f:
                pickle.dump(self.test_data, f)

    def getCrossLabels(self):

        user_dict = {}  # user_id:[place_id_1, place_id_2, ...]
        spot_dict = {}  # place_id:[user_id_1,...]
        split_portion = self.split  # 训练及测试集通过随机抽取比例进行分割

        with open(self.user_cluster_path, 'rb')as f:
            user_cluster = pickle.load(f)
            # for i in range(len(user_cluster)):
            #     print(i,":",len(user_cluster[i]))
        # 当数据集为Gowalla是 选择user_cluster[0] 即第1组，当数据集为Yelp时，选择user_cluster[2] 即第3组
        self.users_sim = user_cluster[self.num_cluster]
        with open(self.checkins_dict_path, 'rb') as f:
            user_dict = pickle.load(f)
            poi_lower, poi_upper = 20, 30
            for user in list(self.users_sim ):
                poi_array = user_dict[user]
                for poi in poi_array:
                    if poi not in spot_dict.keys():
                        spot_dict[poi] = []
                    spot_dict[poi].append(user)
            for poi in list(spot_dict.keys()):
                if len(spot_dict[poi])<poi_lower or len(spot_dict[poi]) > poi_upper:
                    del spot_dict[poi]
            # print(len(self.users_sim ),len(spot_dict))

            # count #POI of user checked in min_user,max_user=15,1387
            # min_user,max_user=50,0
            # for user in list(self.users_sim):
            #     if len(user_dict[user])>max_user:
            #         max_user=len(user_dict[user])
            #     elif len(user_dict[user])<= min_user:
            #         min_user=len(user_dict[user])
            #     else:
            #         continue
            # print(max_user,min_user)


        print('Generating labels')
        user_enum = {}
        spot_enum = {}

        # 用于给user_id重新编号
        u_counter = 0
        s_counter = 0

        # self.R_train = np.zeros((len(set(self.users_sim)), len(set(spot_dict))), dtype=np.float32)
        # R_train = np.zeros((len(set(self.users_sim)), len(set(spot_dict))), dtype=np.float32)
        all_pois = set(spot_dict.keys())
        for user in self.users_sim:

            train_pois = np.random.choice(list(user_dict[user]), size=int(len(user_dict[user]) * split_portion),
                                          replace=False)
            test_pois = list(set(user_dict[user]) - set(train_pois))
            all_neg_pois = list(all_pois - set(user_dict[user]))
            # neg_pois = np.random.choice(all_neg_pois, size=self.negative, replace=False)
            # neg_pois = np.random.choice(all_neg_pois, size=self.negative * len(test_pois), replace=False)

            user_enum[user] = u_counter

            u_counter += 1
            # 构造训练集
            for poi in train_pois:
                if poi in spot_dict.keys():
                    if poi not in spot_enum:
                        spot_enum[poi] = s_counter
                        s_counter += 1
                    self.train_data['user'].append(user_enum[user])
                    self.train_data['spot'].append(spot_enum[poi])
                    self.train_data['label'].append(1)
                    R_train[user_enum[user]][spot_enum[poi]] = 1
                    # self.R_train[user_enum[user]][spot_enum[poi]] = 1
            # 为每一个用户 构造测试集 和 负样本
            user_test_pois = []
            neg = []
            for poi in test_pois:
                if poi in spot_dict.keys():
                    if poi not in spot_enum:
                        spot_enum[poi] = s_counter
                        s_counter += 1
                    user_test_pois.append(spot_enum[poi])
                    R_train[user_enum[user]][spot_enum[poi]] = 1

                neg_pois = np.random.choice(all_neg_pois, size=self.negative, replace=False)
                for k in neg_pois:
                    if k in spot_dict.keys():
                        if k not in spot_enum:
                            spot_enum[k] = s_counter
                            s_counter += 1
                        neg.append(spot_enum[k])
            if len(user_test_pois) == 0 or len(neg) == 0:
                continue
            else:

                self.test_set.append([user_enum[user], user_test_pois])
                self.test_negatives.append(neg)

        for poi in test_pois:
            if poi in spot_dict.keys():
                if poi not in spot_enum:
                    spot_enum[poi] = s_counter
                    s_counter += 1
                self.test_data['user'].append(user_enum[user])
                self.test_data['spot'].append(spot_enum[poi])
                self.test_data['label'].append(1)

                for k in neg_pois:
                    if k in spot_dict.keys():
                        if k not in spot_enum:
                            spot_enum[k] = s_counter
                            s_counter += 1
                        self.test_data['user'].append(user_enum[user])
                        self.test_data['spot'].append(spot_enum[k])
                        self.test_data['label'].append(0)

        print('Writing bi-graph to file')
        with open(self.R_train_path, 'wb') as f:
            pickle.dump(R_train, f)

    return user_enum, spot_enum


if __name__ == "__main__":
    # dataset_name = str(sys.argv[1])
    # num_cluster = int(sys.argv[2])
    # inter_path=str(sys.argv[3])
    dataset_name = "Gowalla"
    num_cluster = 1
    # inter_path='inter'

    # train_path = data_path + 'train.pkl'
    # test_path = data_path + 'test.pkl'
    # test_set_path = data_path + 'test_set.pkl'
    # test_negatives_path = data_path + 'test_negatives.pkl'
    # R_path = data_path + 'R.pkl'
    Dataset(dataset_name, num_cluster)
