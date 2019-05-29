"""
构造神经网络
"""
import os
import pickle
import numpy as np
from keras import backend as K
import tensorflow as tf

from keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import sys
from evaluate import evaluate_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def init_normal(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)


def get_Model(U, num_users, num_spots, layers=[20, 10, 5], regs=[0, 0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    spot_input = Input(shape=(1,), dtype='int32', name='spot_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='user_embedding',
                               embeddings_initializer='uniform', embeddings_regularizer=l2(regs[0]),
                               input_length=1)
    spot_embedding = Embedding(input_dim=num_spots, output_dim=layers[0] // 2, name='spot_embedding',
                               embeddings_initializer='uniform', embeddings_regularizer=l2(regs[0]),
                               input_length=1)

    # user_latent = Flatten()(user_embedding(user_input))
    # spot_latent = Flatten()(spot_embedding(spot_input))
    user_latent = user_embedding(user_input)
    spot_latent = spot_embedding(spot_input)
    embeddings = Concatenate(axis=0)([user_latent, spot_latent])
    embeddings = tf.matmul(U, embeddings)
    vector = Flatten()(embeddings)

    for i in range(len(layers)):
        hidden = Dense(layers[i], activation='relu', kernel_initializer='lecun_uniform', name='us_hidden_' + str(i))
        vector = hidden(vector)

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)
    model = Model(inputs=[user_input, spot_input], outputs=[prediction])
    return model


def get_train_instances(train_data):
    user_input = train_data['user_input']
    spot_input = train_data['spot_input']
    us_label = train_data['us_label']
    for i in range(len(us_label)):
        u = []  # user
        s = []  # spot
        p = []  # prediction
        u.append(user_input[i])
        s.append(spot_input[i])
        p.append(us_label[i])
        x = {'user_input': np.array(u), 'spot_input': np.array(s)}
        y = {'prediction': np.array(p)}
        yield (x, y)


def load_data(data_path, R_path, train=True):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(R_path, 'rb')as f:
        R = pickle.load(f)
    U=spectral_matrix(R)

    if train:
        user_input = data['user']
        spot_input = data['spot']
        us_label = data['label']

        data = {'user_input': user_input, 'spot_input': spot_input, 'us_label': us_label}
        num_users, num_spots = len(user_input), len(spot_input)

        n_users, n_spots = len(set(user_input)), len(set(spot_input))

        return data, num_users, num_spots, U, n_users, n_spots
    else:
        return data


def spectral_matrix(graph):
    (n_users, n_pois) = graph.shape
    print('graph.shape:', graph.shape)

    A = tf.Variable(tf.zeros([n_users + n_pois, n_users + n_pois]), name='A')
    B = tf.Variable(tf.zeros([n_users + n_pois, n_users + n_pois]), name='B')

    A = tf.assign(A[:n_users, n_users:], graph)
    print("A[:n_users, n_users:]=graph,over")

    B = tf.assign(B[n_users:, :n_users], graph.T)
    print("A[n_users:, :n_users]=graph.T over")
    A = tf.add(A, B)

    D = tf.reduce_sum(A, axis=0, keepdims=False)
    L = D - A
    lamda, U = tf.self_adjoint_eig(L)
    lamda = tf.diag(lamda)
    UT = tf.transpose(U, perm=[1, 0])
    A_hat = tf.matmul(U, UT) + tf.matmul(tf.matmul(U, lamda), UT)
    return A_hat


if __name__ == '__main__':

    # param=sys.argv
    dataset_name = str(sys.argv[1])
    einter_path = str(sys.argv[2])

    project_path = os.path.dirname(os.getcwd())
    data_path = project_path + '/data/' + dataset_name + '/' + einter_path + '/'

    train_path = data_path + 'train.pkl'
    test_path = data_path + 'test.pkl'
    test_set_path = data_path + 'test_set.pkl'
    test_negatives_path = data_path + 'test_negatives.pkl'
    R_path = data_path + 'R.pkl'

    layers = eval("[4]")
    reg_layers = eval("[0]")
    # layers=eval(param[1])
    # reg_layers=eval(param[2])
    learner = "Adam"
    # learner ='RMSprop'
    learning_rate = 0.0001
    epochs = 1
    batch_size = 256
    verbose = 2
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    evaluation_threads = 1
    losses = ['binary_crossentropy']

    train_data, train_users, train_spots, U, n_users, n_spots = load_data(train_path, R_path, True)

    test_set = load_data(test_set_path, False)
    test_negatives = load_data(test_negatives_path, False)

    model = get_Model(U, train_users, train_spots, layers, reg_layers)
    model.compile(optimizer=Adam(lr=learning_rate), loss=['binary_crossentropy'], metrics=['accuracy'])
    #  Train Model
    print('Start Trainining')
    for epoch in range(epochs):
        model_hist = model.fit_generator(get_train_instances(train_data), steps_per_epoch=4, epochs=10, verbose=1)
    # model_hist = model.fit_generator(get_train_instances(train_data), steps_per_epoch=4, epochs=1, verbose=2)

    print('Evaluating the model')
    # 输入测试集就行啦
    # for topK in [10, 20, 30, 40]:
    #     precision, recall, map, ndcg = evaluate_model(model, test_set, test_negatives, topK, evaluation_threads)
    #     print(topK, 'Init: Precision = %.4f,Recall = %.4f,MAP = %.4f,NDCG = %.4f' % (precision, recall, map, ndcg))
    # topK = 10
    # precision, recall, map, ndcg = evaluate_model(model, test_set, test_negatives, topK, evaluation_threads)
    # print(topK, 'Init: Precision = %.4f,Recall = %.4f,MAP = %.4f,NDCG = %.4f' % (precision, recall, map, ndcg))
