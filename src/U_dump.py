import tensorflow as tf
import numpy as np
import pickle
import os
import sys


def spectral_matrix(graph):
    (n_users, n_pois) = graph.shape
    print('graph.shape:', graph.shape)
    A = np.zeros([n_users + n_pois, n_users + n_pois], dtype=np.float32)
    print('A.shape', A.shape)
    A[:n_users, n_users:] = graph
    A[n_users:, :n_users] = graph.T
    D = np.sum(A, axis=1, keepdims=False)
    L = D - A
    # temp = np.dot(np.diag(np.power(D, -1)), A)
    # for i in range(np.shape(temp)[0]):
    #     for j in range(np.shape(temp)[1]):
    #         if np.isnan(temp[i][j]) or np.isinf(temp[i][j]):
    #             temp[i][j] = 0
    # L = np.identity(n_users + n_pois, dtype=np.float32) - temp
    lamda, U = np.linalg.eig(L)
    A_hat = np.dot(U, U.T) + np.dot(np.dot(U, lamda), U.T)
    A_hat = A_hat.astype(np.float32)
    return A_hat


if __name__ == '__main__':
    dataset_name = str(sys.argv[1])
    einter_path=str(sys.argv[2])
    project_path = os.path.dirname(os.getcwd())
    data_path = project_path + '/data/' + dataset_name + '/'+einter_path+'/'
    R_path = data_path + 'R_train.pkl'

    with open(R_path, 'rb')as f:
        R = pickle.load(f)

    U = spectral_matrix(R)

    with open(data_path + 'U.pkl', 'wb')as f:
        pickle.dump(U,f)
