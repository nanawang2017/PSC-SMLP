"""
Step2：
对用户进行谱聚类
读取 Step1中生成的用户签到信息数据集（user_poi.pkl）和社交关系数据集(user_friend.pkl)
重要部分就是求用户的相似度矩阵 考虑用户社交信息和签到信息 借鉴的是Salton系数的思想
sim(u1,u2)有两部分：
    1)【|P(u1）与P(u2)的交集|/（sqrt(|P(u1）|)*sqrt(|P(u2)|)】从签到信息（用户签到偏好）出发
    2)【|S(u1）与S(u2)的交集|/（sqrt(|S(u1）|)*sqrt(|S(u2)|)】 从好友信息（用户社交影响）出发

"""
import os
import numpy as np
import pickle
from sklearn.cluster import SpectralClustering
from sklearn import metrics


class CSSC(object):
    def __init__(self, dataset_name):
        self.user_poi = {}
        self.user_friend = {}
        self.user_cluster = {}
        self.project_path = os.path.dirname(os.getcwd())
        self.checkins_dict_path = self.project_path + '/data/' + dataset_name + '/inter/' + 'user_poi.pkl'
        self.social_dict_path = self.project_path + '/data/' + dataset_name + '/inter/' + 'user_friend.pkl'
        self.similarity_matrix_path = self.project_path + '/data/' + dataset_name + '/inter/' + 'similarity_matrix.pkl'
        self.user_cluster_path = self.project_path + '/data/' + dataset_name + '/inter/' + 'user_cluster.pkl'
        writeToFile = False
        # 首先 看归一化相似度矩阵是否生成
        try:
            f = open(self.similarity_matrix_path, 'rb')
            f.close()
        except IOError:
            writeToFile = True

        if writeToFile:
            # 生成归一化相似度矩阵W
            print('The similarity_matrix.pkl is creating now!')
            with open(self.checkins_dict_path, 'rb')as f:
                self.user_poi = pickle.load(f)
            with open(self.social_dict_path, 'rb')as f:
                self.user_friend = pickle.load(f)
            W = self.similarity_matrix(self.user_poi, self.user_friend)
            # 将归一化相似度矩阵写入文件
            with open(self.similarity_matrix_path, 'wb+')as f:
                pickle.dump(W, f)
        else:
            # 若归一化相似度矩阵已经生成 则 进行谱聚类
            print('The similarity_matrix.pkl is finished! Now we are on Spectral Clustering')
            with open(self.similarity_matrix_path, 'rb')as f:
                user_similarity = pickle.load(f)
            with open(self.checkins_dict_path, 'rb')as f:
                self.user_poi = pickle.load(f)
            with open(self.social_dict_path, 'rb')as f:
                self.user_friend = pickle.load(f)

            # 做循环看聚类的效果好不好
            # for index, k in enumerate((2, 5, 8, 10, 15)):
            #     y_pred = SpectralClustering(n_clusters=k,affinity='precomputed').fit_predict(user_similarity)
            #     print("n_clusters=", k, "score:", metrics.calinski_harabaz_score(user_similarity, y_pred))
            #  聚类个数得到:Gowalla聚类为8,Yelp 聚类个数为2的时候是不错的
            y_pred = SpectralClustering(n_clusters=2,affinity='precomputed').fit(user_similarity)
            clusters = y_pred.labels_
            users = list(set(self.user_poi.keys()).intersection(set(self.user_friend.keys())))
            for i in list(set(clusters)):
                indexs = [j for j, x in enumerate(clusters) if x == i]
                self.user_cluster[i] = [users[x] for x in indexs]
            # for i in range(len(set(clusters))):
            #     indexs = [j for j, x in enumerate(clusters) if x == i]
            #     self.user_cluster[i] = [users[indexs[x]] for x in range(len(indexs))]
            # # print(len(user_cluster))
            with open(self.user_cluster_path, 'wb+')as f:
                pickle.dump(self.user_cluster, f)

    def similarity_matrix(self, user_poi, user_friend):
        user_similarity = []
        common_users = list(set(user_poi.keys()).intersection(set(user_friend.keys())))
        print(len(common_users))
        for user1 in common_users:
            for user2 in common_users:
                s1 = set(user_friend[user1])
                s2 = set(user_friend[user2])
                countf = len(s1.intersection(s2))
                p1 = set(user_poi[user1])
                p2 = set(user_poi[user2])
                countp = len(p1.intersection(p2))
                uf = float(countf) / (np.math.sqrt(len(s1)) * np.math.sqrt(len(s2)))
                up = float(countp) / (np.math.sqrt(len(p1)) * np.math.sqrt(len(p2)))
                user_similarity.append(uf * up)
        n = len(common_users)
        user_similarity = np.array(user_similarity).reshape(n, n)
        # 让对角线为0
        # U = np.triu(user_similarity)
        # U += U.T - np.diag(U.diagonal())
        # U = U - np.diag(np.diag(U))
        #  求度矩阵
        Degree = np.sum(user_similarity, axis=1, keepdims=False)
        # 求随机游走思想进行归一化W=D-1U
        W = np.dot(np.diag(np.power(Degree, -1)), user_similarity)
        for i in range(np.shape(W)[0]):
            for j in range(np.shape(W)[1]):
                if np.isnan(W[i][j]) or np.isinf(W[i][j]):
                    W[i][j] = 0
        U = np.triu(W)
        U += U.T - np.diag(U.diagonal())
        W = U - np.diag(np.diag(U))
        return W


if __name__ == '__main__':
    datasetname = 'Yelp'
    CSSC(datasetname)
