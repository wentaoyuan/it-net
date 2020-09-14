'''
MIT License

Copyright (c) 2019 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import importlib
import numpy as np
import tensorflow as tf
import time
from tensorpack import dataflow

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'classifier'))
sys.path.append(os.path.join(BASE_DIR, 'transformer'))
sys.path.append(os.path.join(BASE_DIR, 'util'))
from data_util import ResampleData
from visu_util import plot_iters, plot_conf

n_classes = 40
cat_names = ['airplane','bathtub','bed','bench','bookshelf',
             'bottle','bowl','car','chair','cone',
             'cup','curtain','desk','door','dresser',
             'flower_pot','glass_box','guitar','keyboard','lamp',
             'laptop','mantel','monitor','night_stand','person',
             'piano','plant','radio','range_hood','sink',
             'sofa','stairs','stool','table','tent',
             'toilet','tv_stand','vase','wardrobe','xbox']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path')
    parser.add_argument('--checkpoint')
    parser.add_argument('--results_dir')
    parser.add_argument('--classifier', choices=['pointnet', 'dgcnn'])
    parser.add_argument('--transformer', choices=['t_net', 'it_net', 'it_net_dgcnn'])
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--plot_freq', type=int, default=40)
    parser.add_argument('--plot_lim', type=float, default=0.7)
    parser.add_argument('--plot_size', type=float, default=5)
    args = parser.parse_args()

    is_training_pl = tf.placeholder(tf.bool, (), 'is_training')
    points_pl = tf.placeholder(tf.float32, (1, args.num_points, 3), 'points')

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        transformer = importlib.import_module(args.transformer)
        transformed_points, T_out, Ts = transformer.get_model(points_pl, args.n_iter,
                                                              is_training_pl, 0.99)
    with tf.variable_scope('classifier'):
        classifier = importlib.import_module(args.classifier)
        logits = classifier.get_model(transformed_points, is_training_pl, 0.99)
        prediction = tf.argmax(logits, axis=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)

    df = dataflow.LMDBSerializer.load(args.lmdb_path, shuffle=False)
    if args.num_points is not None:
        df = ResampleData(df, args.num_points, 'cls')

    times = []
    conf = np.zeros((n_classes, n_classes))
    rotation = [[[] for i in range(args.n_iter+1)] for j in range(n_classes)]
    translation = [[[] for i in range(args.n_iter+1)] for j in range(n_classes)]
    is_correct = [[] for i in range(n_classes)]
    r_mag = [[] for i in range(args.n_iter)]
    t_mag = [[] for i in range(args.n_iter)]
    for i, (model_id, points, label, init_pose) in enumerate(df.get_data()):
        start = time.time()
        pred, ts = sess.run([prediction, Ts],
                            feed_dict={is_training_pl: False, points_pl: [points]})
        times.append(time.time() - start)

        conf[pred[0], label] += 1
        is_correct[label].append([pred[0] == label])

        T = np.eye(4)
        transforms = []
        titles = []
        for j in range(args.n_iter+1):
            transforms.append(T)
            titles.append('Iteration %d' % j)
            if j < args.n_iter:
                T = np.dot(ts[j][0], T)

        if (i+1) % args.plot_freq == 0:
            figpath = os.path.join(args.results_dir, 'plots', '%s.png' % model_id)
            titles[0] = 'Input'
            suptitle = 'Predicted %s   Actual %s' % (cat_names[pred[0]], cat_names[label])
            figpath = os.path.join(args.results_dir, 'plots', '%s.png' % model_id)
            plot_iters(figpath, points, transforms, titles, args.plot_lim, args.plot_size, suptitle)

    print('Average time', np.mean(times))
    print('Average accuracy', np.trace(conf) / np.sum(conf))
    print('Average class accuracy', np.nanmean(np.diag(conf) / np.sum(conf, axis=0)))
    plot_conf(os.path.join(args.results_dir, 'confusion.png'), conf)
    np.savetxt(os.path.join(args.results_dir, 'confusion.txt'), conf, '%d')
    with open(os.path.join(args.results_dir, 'precision.txt'), 'w') as file:
        file.write('\n'.join(['%s: %.4f' % (cat_names[i], conf[i, i] / np.sum(conf[i, :])) for i in range(n_classes)]))
    with open(os.path.join(args.results_dir, 'recall.txt'), 'w') as file:
        file.write('\n'.join(['%s: %.4f' % (cat_names[i], conf[i, i] / np.sum(conf[:, i])) for i in range(n_classes)]))
