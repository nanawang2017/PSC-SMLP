"""
laplacian eigen matrix enhanced Embedding for MLP
Created on 4/17,2019
"""
import tensorflow as tf
import os
import pickle
import numpy as np
from evaluate import evaluate_model
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class eMLP(object):
    def __init__(self, graph, n_users, n_pois, emb_dim, labels, lr, batch_size, decay, K=1, layers=[32, 16, 8]):
        self.model_name = 'Enhanced Embedding Based Bi-Graphwith eigen decomposition'

        self.K = K
        self.graph = graph
        self.n_users = n_users
        self.n_pois = n_pois
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.decay = decay
        self.layers = layers

        self.A = self.adjacient_matrix(self_connection=True)
        self.D = self.degree_matrix()
        self.L = self.laplacian_matrix(normalized=True)

        # np.linalg.eig（）返回矩阵的特征值和特征向量，特征值以数组形式返回，特征向量以【【】，【】，【】】形式（也是数组不过更像矩阵）
        # np.diag（）创建一个矩阵或者返回一个数组的对角元素
        self.lamda, self.U = np.linalg.eig(self.L)
        self.lamda = np.diag(self.lamda)

        # placeholder definition  placeholder是用来保存数据的
        # placeholder（type,structure…)
        # 它的第一个参数是你要保存的数据的数据类型大多数是tensorflow中的float32数据类型，
        # 后面的参数就是要保存数据的结构，比如要保存一个1×2的矩阵，则struct=[1 2]。
        # 它在使用的时候和前面的variable不同的是在session运行阶段，需要给placeholder提供数据，利用feed_dict的字典结构给placeholdr变量“喂数据”
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pois = tf.placeholder(tf.int32, shape=(self.batch_size,))
        # self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.poi_embeddings = tf.Variable(
            tf.random_normal([self.n_pois, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='poi_embeddings')
        # tf.Variable(initializer,name),参数initializer是初始化参数
        # [self.emb_dim, self.emb_dim]是初始化的shape
        self.filters = []
        for k in range(self.K):
            self.filters.append(
                tf.Variable(
                    tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32))

            )

        A_hat = np.dot(self.U, self.U.T) + np.dot(np.dot(self.U, self.lamda), self.U.T)
        # A_hat += np.dot(np.dot(self.U, self.lamda_2), self.U.T)
        A_hat = A_hat.astype(np.float32)  # astype变化数据类型

        embeddings = tf.concat([self.user_embeddings, self.poi_embeddings], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.K):
            # .tf.multiply（）两个矩阵中对应元素各自相乘
            # .tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
            embeddings = tf.matmul(A_hat, embeddings)
            # filters = self.filters[k]#tf.squeeze(tf.gather(self.filters, k))
            # sigmoid函数是对矩阵或者向量中的每一个元素求sigmoid值
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings, self.filters[k]))
            all_embeddings += [embeddings]  # 将向量进行连接 [a]+[b]=[[a],[b]]
        self.vector = tf.concat(all_embeddings, 1)
        # tf.concat(all_embeddings, 1)的作用
        # 将[ array([[0.7310586, 0.880797 ],
        #        [0.7310586, 0.880797 ]], dtype=float32),
        #     array([[0.880797 , 0.7310586],
        #        [0.880797 , 0.7310586]], dtype=float32)
        #   ]变成
        # [[0.7310586 0.880797  0.880797  0.7310586]
        #  [0.7310586 0.880797  0.880797  0.7310586]]

        # self.u_embeddings, self.p_embeddings = tf.split(all_embeddings, [self.n_users, self.n_pois], 0)

        # self.u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        # self.p_embeddings = tf.nn.embedding_lookup(self.p_embeddings, self.pois)

        # self.all_ratings = tf.matmul(self.u_embeddings, self.p_embeddings, transpose_a=False, transpose_b=True)
        self.weights, self.bias = self._initialize_weights()
        vector = self.vector
        layers = self.layers

        # 添加神经网络
        for i in range(len(layers)):
            lay = str(layers[i])
            hidden = tf.add(tf.matmul(vector, self.weights[lay]), self.bias[lay])
            vector = tf.nn.relu(hidden)
            # self.layer1_MLP = tf.layers.dense(inputs=self.interaction,
            #                                   units=self.embed_size * 2,
            #                                   activation=self.activation_func,
            #                                   kernel_initializer=self.initializer,
            #                                   kernel_regularizer=self.regularizer,
            #                                   name='layer1_MLP')

        # calculate the log loss
        self.out = tf.sigmoid(vector)
        self.loss = tf.losses.log_loss(labels, self.out, epsilon=1e-15, scope=None)

        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vector,labels=))
        self.opt = tf.train.RMSPropOptimizer(learning_rate=lr)

        self.updates = self.opt.minimize(self.loss,
                                         var_list=[self.user_embeddings, self.poi_embeddings] + self.filters)

    def _initialize_weights(self):

        # 定义所有的权重
        all_weights = dict()
        all_bias = dict()
        input_size = len(self.vector)
        layers = self.layers
        for i in range(len(layers)):
            lay = str(layers[i])
            output_dim = layers[i]
            all_weights[lay] = tf.Variable(tf.truncated_normal([input_size, output_dim], 0.0, 0.1),
                                           name='up_embeddings', dtype=tf.float32, trainable=True)
            all_bias[lay] = tf.Variable(tf.truncated_normal([output_dim, 1], 0.0, 0.1),
                                        name='up_bias', dtype=tf.float32, trainable=True)

        return all_weights, all_bias

    def adjacient_matrix(self, self_connection=False):
        A = np.zeros([self.n_users + self.n_pois, self.n_users + self.n_pois], dtype=np.float32)
        A[:self.n_users, self.n_users:] = self.graph
        A[self.n_users:, :self.n_users] = self.graph.T
        if self_connection == True:
            return np.identity(self.n_users + self.n_pois, dtype=np.float32) + A
        return A

    def degree_matrix(self):
        degree = np.sum(self.A, axis=1, keepdims=False)
        # degree = np.diag(degree)
        return degree

    def laplacian_matrix(self, normalized=False):
        if normalized == False:
            return self.D - self.A

        temp = np.dot(np.diag(np.power(self.D, -1)), self.A)
        # temp = np.dot(temp, np.power(self.D, -0.5))
        return np.identity(self.n_users + self.n_pois, dtype=np.float32) - temp


def load_data(data_path, R_path, train=True):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(R_path, 'rb')as f:
        R = pickle.load(f)
    U = spectral_matrix(R)

    if train:
        user_input = data['user']
        spot_input = data['spot']
        us_label = data['label']

        data = {'user_input': user_input, 'spot_input': spot_input, 'us_label': us_label}
        n_users = len(user_input)
        n_pois = len(spot_input)
        labels = us_label

        return data, R, n_users, n_pois, labels, user_input, spot_input
    else:
        return data
def spectral_matrix(graph):
    (n_users, n_pois) = graph.shape
    print('graph.shape:', graph.shape)
    # A = np.zeros([n_users + n_pois, n_users + n_pois], dtype=np.float32)
    # A[:n_users, n_users:] = graph
    # A[n_users:, :n_users] = graph.T

    # A= tf.placeholder(tf.int32, shape=[n_users + n_pois, n_users + n_pois])
    # A[:n_users, n_users:] = tf.Variable(graph)
    # A[n_users:, :n_users] = tf.Variable(tf.transpose(graph,perm=[1,0]))

    A = tf.Variable(tf.zeros([n_users + n_pois, n_users + n_pois]), name='A')
    B = tf.Variable(tf.zeros([n_users + n_pois, n_users + n_pois]), name='B')

    A = tf.assign(A[:n_users, n_users:], graph)
    print("A[:n_users, n_users:]=graph,over")

    B = tf.assign(B[n_users:, :n_users], graph.T)
    print("A[n_users:, :n_users]=graph.T over")
    A = tf.add(A, B)

    # A_temp=tf.constant(A)
    # B=tf.stack(A_temp)
    D = tf.reduce_sum(A, axis=0, keepdims=False)
    L = D - A
    lamda, U = tf.self_adjoint_eig(L)
    lamda = tf.diag(lamda)
    UT = tf.transpose(U, perm=[1, 0])
    A_hat = tf.matmul(U, UT) + tf.matmul(tf.matmul(U, lamda), UT)
    return A_hat
    # D = np.sum(A, axis=1, keepdims=False)
    # L = D - A
    # temp = np.dot(np.diag(np.power(D, -1)), A)
    # for i in range(np.shape(temp)[0]):
    #     for j in range(np.shape(temp)[1]):
    #         if np.isnan(temp[i][j]) or np.isinf(temp[i][j]):
    #             temp[i][j] = 0
    # L = np.identity(n_users + n_pois, dtype=np.float32) - temp
    # lamda, U = np.linalg.eig(L)
    # A_hat = np.dot(U, U.T) + np.dot(np.dot(U, lamda), U.T)
    # A_hat = A_hat.astype(np.float32)
    # return A_hat

if __name__ == '__main__':
    param = sys.argv
    dataset_name = 'Gowalla'
    project_path = os.path.dirname(os.getcwd())
    data_path = project_path + '/data/' + dataset_name + '/einter/'

    train_path = data_path + 'train.pkl'
    test_path = data_path + 'test.pkl'
    test_set_path = data_path + 'test_set.pkl'
    test_negatives_path = data_path + 'test_negatives.pkl'
    R_path = data_path + 'R_train.pkl'

    layers = eval("[4]")
    reg_layers = eval("[0]")
    emb_dim=10
    N_EPOCH=1
    # layers = eval(param[1])
    # reg_layers = eval(param[2])
    # emb_dim = eval(param[3])
    # N_EPOCH = eval(param[4])
    learner = "Adam"
    # learner ='RMSprop'
    learning_rate = 0.0001
    epochs = 1
    batch_size = 256
    verbose = 2
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    evaluation_threads = 1
    losses = ['binary_crossentropy']

    train_data, R, n_users, n_pois, labels, users, pois = load_data(train_path, R_path, True)
    test_set = load_data(test_set_path, False)
    test_negatives = load_data(test_negatives_path, False)

    model = eMLP(R, n_users, n_pois, emb_dim, labels, learning_rate, batch_size, decay=0.001, K=1, layers=[32, 16, 8])
    print(model.model_name)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_EPOCH):
        _, loss = sess.run([model.updates, model.loss],
                           feed_dict={model.users: users, model.pois: pois})

    print("Training Finished!")
        # 后面这两行要多注意看一下 是跟CSSC-MLP的不同之处
        # users_to_test = list(data_generator.test_set.keys())
        #
        # ret = test(sess, model, users_to_test)
        #
        # print('Epoch %d training loss %f' % (epoch, loss))
        # print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
        #       % (ret[0], ret[1], ret[2], ret[3], ret[4]))
        # print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
        #       % (ret[5], ret[6], ret[7], ret[8], ret[9]))
