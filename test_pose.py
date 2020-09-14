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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorpack import dataflow

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'transformer'))
sys.path.append(os.path.join(BASE_DIR, 'util'))
from tf_util import rotation_error, translation_error
from visu_util import plot_iters, plot_mean_std, plot_cdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path')
    parser.add_argument('--checkpoint')
    parser.add_argument('--results_dir')
    parser.add_argument('--transformer', choices=['t_net', 'it_net', 'it_net_dgcnn'])
    parser.add_argument('--randinit', action='store_true')
    parser.add_argument('--nostop', action='store_true')
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--iter_plot_freq', type=int, default=3)
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--plot_lim', type=float, default=0.3)
    parser.add_argument('--plot_size', type=float, default=7)
    parser.add_argument('--plot_nbins', type=int, default=100)
    args = parser.parse_args()

    is_training_pl = tf.placeholder(tf.bool, (), 'is_training')
    points_pl = tf.placeholder(tf.float32, (1, None, 3), 'points')

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        transformer = importlib.import_module(args.transformer)
        transformed_points, T_out, Ts = transformer.get_model(points_pl, args.n_iter,
                                                              is_training_pl, 0.99,
                                                              args.randinit, args.nostop)

    R_pl = tf.placeholder(tf.float32, (3, 3), 'rotation')
    R_gt_pl = tf.placeholder(tf.float32, (3, 3), 'rotation_gt')
    t_pl = tf.placeholder(tf.float32, (3,), 'translation')
    t_gt_pl = tf.placeholder(tf.float32, (3,), 'translation_gt')
    rot_error_op = rotation_error(R_pl, R_gt_pl)
    trans_error_op = translation_error(t_pl, t_gt_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    df = dataflow.LMDBSerializer.load(args.lmdb_path, shuffle=False)
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)

    times = []
    r_errors = [[] for i in range(args.n_iter + 1)]
    t_errors = [[] for i in range(args.n_iter + 1)]
    g_errors = [[] for i in range(args.n_iter + 1)]
    r_mag = [[] for i in range(args.n_iter)]
    t_mag = [[] for i in range(args.n_iter)]
    n_correct = np.zeros(args.n_iter + 1)
    for n, (model_id, pcd, T_gt) in enumerate(df.get_data()):
        start = time.time()
        ts = sess.run(Ts, {is_training_pl: False, points_pl: [pcd]})
        times.append(time.time() - start)

        T = np.eye(4)
        transforms = []
        titles = []
        for i in range(args.n_iter + 1):
            r_error = sess.run(rot_error_op, {R_pl: T[:3, :3], R_gt_pl: T_gt[:3, :3]})
            t_error = sess.run(trans_error_op, {t_pl: T[:3, 3], t_gt_pl: T_gt[:3, 3]})
            r_errors[i].append(r_error)
            t_errors[i].append(t_error)
            if i % args.iter_plot_freq == 0:
                error_str = 'Rotation error %.4f\nTranslation error %.4f\n' % (r_error, t_error)
                transforms.append(T)
                if i == 0:
                    titles.append('Input\n%s' % error_str)
                else:
                    titles.append('Iteration %d\n%s' % (i, error_str))
            if r_error < 10 and t_error < 0.1:
                n_correct[i] += 1
            if i < args.n_iter:
                T = np.dot(ts[i][0], T)
                r_mag = sess.run(rot_error_op, {R_pl: ts[i][0, :3, :3], R_gt_pl: np.eye(3, dtype=np.float32)})
                t_mag = sess.run(trans_error_op, {t_pl: ts[i][0, :3, 3], t_gt_pl: np.zeros(3, dtype=np.float32)})
                r_mag[i].append(r_mag)
                t_mag[i].append(t_mag)
        transforms.append(T_gt)
        titles.append('Ground truth\n\n')

        if n % args.plot_freq == 0:
            figpath = os.path.join(args.results_dir, 'plots', '%s.png' % model_id)
            plot_iters(figpath, pcd, transforms, titles, args.plot_lim, args.plot_size)

    np.savetxt(os.path.join(args.results_dir, 'r_err.txt'), r_errors)
    np.savetxt(os.path.join(args.results_dir, 't_err.txt'), t_errors)

    print('Average time', np.mean(times), 'std', np.std(times))
    print('Percentage < 10 degree, 0.1', n_correct / (n+1))

    plt.figure()
    plt.plot(np.arange(args.n_iter + 1), n_correct / (n+1))
    plt.xlabel('number of iterations')
    plt.ylabel('percentage < 10 degree, 0.1')
    plt.savefig(os.path.join(args.results_dir, 'pose_accuracy.png'))

    plot_mean_std(os.path.join(args.results_dir, 'r_err.png'),
                  np.arange(args.n_iter + 1), r_errors, 'rotation error')
    plot_mean_std(os.path.join(args.results_dir, 't_err.png'),
                  np.arange(args.n_iter + 1), t_errors, 'translation error')
    plot_mean_std(os.path.join(args.results_dir, 'r_mag.png'),
                  np.arange(args.n_iter) + 1, r_mag, 'Predicted rotation magnitude')
    plot_mean_std(os.path.join(args.results_dir, 't_mag.png'),
                  np.arange(args.n_iter) + 1, t_mag, 'Predicted translation magnitude')

    for i in range(args.n_iter + 1):
        if i % args.iter_plot_freq == 0:
            plot_cdf(os.path.join(args.results_dir, 'r_err_cdf_iter-%d.png' % i),
                     r_errors[i], 'Rotation error')
            plot_cdf(os.path.join(args.results_dir, 't_err_cdf_iter-%d.png' % i),
                     t_errors[i], 'Translation error')
