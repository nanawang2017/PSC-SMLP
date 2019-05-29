import pickle
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    dataset_name = 'Gowalla'
    project_path = os.path.dirname(os.getcwd())
    data_path = project_path + '/data/' + dataset_name + '/einter/'
    R_path = data_path + 'R_train.pkl'

    with open(R_path, 'rb')as f:
        R = pickle.load(f)

    A_hat = spectral_matrix(R)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        U = sess.run(A_hat)
        with open(data_path + 'U_tf.pkl', 'wb')as f:
            pickle.dump(U,f)
