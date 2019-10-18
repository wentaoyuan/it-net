import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../util'))
import tensorflow as tf
import tf_util


def get_model(point_cloud, is_training, bn_decay):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    point_cloud = tf.expand_dims(point_cloud, -2)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(point_cloud, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    net = tf.squeeze(net, -2)

    # Symmetric function: max pooling
    net = tf.reduce_max(net, axis=1, name='maxpool')

    # MLP on global point cloud vector
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
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
