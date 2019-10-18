import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../util'))
import tensorflow as tf
import tf_util


def input_transform_net(edge_feature, is_training, bn_decay, randinit):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf.squeeze(net, -2)

    net = tf.reduce_max(net, axis=1, name='tmaxpool')

    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform') as sc:
        weights_init = tf.contrib.layers.xavier_initializer() if randinit \
            else tf.zeros_initializer()
        weights = tf.get_variable('weights', [256, 7],
                                  initializer=tf.zeros_initializer(),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [7],
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32)
        if not randinit:
            biases = biases + tf.constant([1,0,0,0,0,0,0], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)
    return transform


def get_model(points, n_iter, is_training, bn_decay, randinit=False, nostop=False, k=10):
    T = tf.eye(4, batch_shape=(points.shape[0],))
    T_deltas = []
    for i in range(n_iter):
        transformed_points = tf_util.transform_points(points, T)
        if not nostop:
            transformed_points = tf.stop_gradient(transformed_points)

        adj_matrix = tf_util.pairwise_distance(transformed_points)
        nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(points, nn_idx=nn_idx, k=k)

        qt = input_transform_net(edge_feature, is_training, bn_decay, randinit)
        T_delta = tf.map_fn(tf_util.qt2mat, qt, dtype=tf.float32)
        T_deltas.append(T_delta)
        T = tf.matmul(T_delta, T)
    transformed_points = tf_util.transform_points(points, T)
    return transformed_points, T, T_deltas
