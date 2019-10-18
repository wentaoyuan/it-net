import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../util'))
import tensorflow as tf
import tf_util


def get_model(point_cloud, input_label, is_training, bn_decay, k=20):
  """ Classification DGCNN, input is BxNx3, output BxNx50 """
  cat_num = 16
  part_num = 50
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value

  adj = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv1', bn_decay=bn_decay)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv2', bn_decay=bn_decay)

  net_max_1 = tf.reduce_max(out2, axis=-2, keepdims=True, name='maxpool1')
  net_mean_1 = tf.reduce_mean(out2, axis=-2, keepdims=True, name='meanpool1')

  out3 = tf_util.conv2d(tf.concat([net_max_1, net_mean_1], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv3', bn_decay=bn_decay)

  adj = tf_util.pairwise_distance(tf.squeeze(out3, axis=-2))
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(tf.squeeze(out3, axis=-2), nn_idx=nn_idx, k=k)

  out4 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv4', bn_decay=bn_decay)
  
  net_max_2 = tf.reduce_max(out4, axis=-2, keepdims=True, name='maxpool2')
  net_mean_2 = tf.reduce_mean(out4, axis=-2, keepdims=True, name='meanpool2')

  out5 = tf_util.conv2d(tf.concat([net_max_2, net_mean_2], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv5', bn_decay=bn_decay)

  adj = tf_util.pairwise_distance(tf.squeeze(out5, axis=-2))
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(tf.squeeze(out5, axis=-2), nn_idx=nn_idx, k=k)

  out6 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv6', bn_decay=bn_decay)

  net_max_3 = tf.reduce_max(out6, axis=-2, keepdims=True, name='maxpool3')
  net_mean_3 = tf.reduce_mean(out6, axis=-2, keepdims=True, name='meanpool3')

  out7 = tf_util.conv2d(tf.concat([net_max_3, net_mean_3], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay)

  out8 = tf_util.conv2d(tf.concat([out3, out5, out7], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv8', bn_decay=bn_decay)

  out_max = tf.reduce_max(out8, axis=1, keepdims=True, name='maxpool4')

  one_hot_label = tf.one_hot(input_label, cat_num)
  one_hot_label_expand = tf.reshape(one_hot_label, [batch_size, 1, 1, cat_num])
  one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 128, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='one_hot_label_expand', bn_decay=bn_decay)
  out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     net_max_1,
                                     net_mean_1,
                                     out3,
                                     net_max_2,
                                     net_mean_2,
                                     out5,
                                     net_max_3,
                                     net_mean_3,
                                     out7,
                                     out8])

  net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv1')
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
  net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2')
  net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
  net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv3')
  net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg/conv4')

  net2 = tf.reshape(net2, [batch_size, num_point, part_num])

  return net2


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('train/classify loss', classify_loss, collections=['train'])
    return classify_loss
