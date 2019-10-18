import argparse
import importlib
import json
import numpy as np
import tensorflow as tf
import time
from tensorpack import dataflow

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'segmenter'))
sys.path.append(os.path.join(BASE_DIR, 'transformer'))
sys.path.append(os.path.join(BASE_DIR, 'util'))
from data_util import random_idx
from visu_util import plot_iters_seg

n_classes = 16
n_parts = 50
synsets = ['02691156', '02773838', '02954340', '02958343',
           '03001627', '03261776', '03467517', '03624134',
           '03636649', '03642806', '03790512', '03797390',
           '03948459', '04099429', '04225987', '04379243']


def mean_iou(pred, labels, num_classes):
    conf = np.zeros((num_classes, num_classes))
    for i in range(pred.shape[0]):
        conf[pred[i], labels[i]] += 1
    tp = np.diag(conf)
    total = np.sum(conf, 0) + np.sum(conf, 1) - tp
    return np.mean(tp[total > 0] / total[total > 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path')
    parser.add_argument('--checkpoint')
    parser.add_argument('--results_dir')
    parser.add_argument('--segmenter', choices=['pointnet', 'dgcnn'])
    parser.add_argument('--transformer', choices=['t_net', 'it_net', 'it_net_dgcnn'])
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--plot_freq', type=int, default=40)
    parser.add_argument('--plot_lim', type=float, default=0.3)
    parser.add_argument('--plot_size', type=float, default=5)
    args = parser.parse_args()

    is_training_pl = tf.placeholder(tf.bool, (), 'is_training')
    points_pl = tf.placeholder(tf.float32, (1, args.num_points, 3), 'points')
    cat_labels_pl = tf.placeholder(tf.int32, (1,), 'cat_labels')

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        transformer = importlib.import_module(args.transformer)
        transformed_points, T_out, Ts = transformer.get_model(points_pl, args.n_iter,
                                                              is_training_pl, 0.99)
    with tf.variable_scope('segmenter'):
        segmenter = importlib.import_module(args.segmenter)
        logits = segmenter.get_model(transformed_points, cat_labels_pl, is_training_pl, 0.99)
        prediction = tf.argmax(logits, axis=2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)

    df = dataflow.LMDBSerializer.load(args.lmdb_path, shuffle=False)

    times = []
    acc = {synset_id: [] for synset_id in synsets}
    iou = {synset_id: [] for synset_id in synsets}
    rotation = [[[] for i in range(args.n_iter+1)] for j in range(n_classes)]
    translation = [[[] for i in range(args.n_iter+1)] for j in range(n_classes)]
    r_mag = [[] for i in range(args.n_iter)]
    t_mag = [[] for i in range(args.n_iter)]
    for i, (model_id, points, labels, cat_label, init_pose) in enumerate(df.get_data()):
        orig_size = points.shape[0]
        idx, perm = random_idx(orig_size, args.num_points)

        start = time.time()
        pred, ts = sess.run([prediction, Ts],
                            feed_dict={is_training_pl: False, points_pl: [points[idx]],
                                       cat_labels_pl: [cat_label]})
        times.append(time.time() - start)

        points = points[perm]
        labels = labels[perm]
        pred = pred[0][:orig_size]

        acc[synsets[cat_label]].append(np.sum(pred == labels) / orig_size)
        iou[synsets[cat_label]].append(mean_iou(pred, labels, n_parts))

        T = np.eye(4)
        transforms = []
        part_ids = []
        titles = []
        for j in range(args.n_iter+1):
            transforms.append(T)
            part_ids.append(pred)
            titles.append('Iteration %d' % j)
            if j < args.n_iter:
                T = np.dot(ts[j][0], T)
        transforms.append(T)
        part_ids.append(labels)
        titles.append('Ground truth')
        if (i+1) % args.plot_freq == 0:
            titles[0] = 'Input'
            figpath = os.path.join(args.results_dir, 'plots', '%s.png' % model_id)
            plot_iters_seg(figpath, points, transforms, part_ids, titles, args.plot_lim, args.plot_size)

    total_acc = 0
    n_shapes = 0
    for synset_id in acc:
        n_shapes += len(acc[synset_id])
        total_acc += np.sum(acc[synset_id])
        acc[synset_id] = np.mean(acc[synset_id])
    avg_acc = total_acc / n_shapes
    total_iou = 0
    for synset_id in iou:
        total_iou += np.sum(iou[synset_id])
        iou[synset_id] = np.mean(iou[synset_id])
    avg_iou = total_iou / n_shapes
    print('Average time', np.mean(times))
    print('Average accuracy', avg_acc)
    print('Average IOU', avg_iou)
    with open(os.path.join(args.results_dir, 'accuracy.txt'), 'w') as file:
        file.write('\n'.join(['%s: %.4f' % (synsets[i], acc[synsets[i]]) for i in range(n_classes)]))
        file.write('\naverage: %.4f' % avg_acc)
    with open(os.path.join(args.results_dir, 'iou.txt'), 'w') as file:
        file.write('\n'.join(['%s: %.4f' % (synsets[i], iou[synsets[i]]) for i in range(n_classes)]))
        file.write('\naverage: %.4f' % avg_iou)
