import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../util'))
import tensorflow as tf
import tf_util


def get_model(point_cloud, is_training, bn_decay, k=20):
    """ Classification DGCNN, input is BxNx3, output Bx40 """

    # EdgeConv functions (MLP implemented as conv2d)
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn1', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net1 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn2', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net2 = net
  
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  

    net = tf_util.conv2d(edge_feature, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn3', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net3 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
    
    net = tf_util.conv2d(edge_feature, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='dgcnn4', bn_decay=bn_decay)
    net = tf.reduce_max(net, axis=-2, keepdims=True)
    net4 = net

    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)
    net = tf.squeeze(net, -2)

    # Symmetric function: max pooling
    net = tf.reduce_max(net, axis=1, name='maxpool')

    # MLP on global point cloud vector
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net


def get_loss(pred, label):
    """ pred: B x NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('train/classify loss', classify_loss, collections=['train'])
    return classify_loss


def get_loss_reg(pred, label, transform, reg_weight=0.001):
    """ pred: B x NUM_CLASSES,
        label: B,
        transform: B x 3 x 3 """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('train/classify loss', classify_loss, collections=['train'])

    # Enforce the transformation as orthogonal matrix
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1])) - tf.eye(3)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('train/mat loss', mat_diff_loss, collections=['train'])

    return classify_loss + mat_diff_loss * reg_weight
