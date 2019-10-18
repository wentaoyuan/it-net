import argparse
import importlib
import numpy as np
import tensorflow as tf
from termcolor import colored

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'transformer'))
sys.path.append(os.path.join(BASE_DIR, 'util'))
from data_util import lmdb_dataflow
from log_util import create_log_dir
from tf_util import geometric_error, rotation_error, translation_error
from visu_util import plot_iters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--transformer', choices=['t_net', 'it_net', 'it_net_dgcnn'])
    parser.add_argument('--nostop', action='store_true')
    parser.add_argument('--randinit', action='store_true')
    parser.add_argument('--n_iter', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_points', type=int, default=256)
    parser.add_argument('--max_step', type=int, default=20000)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_decay_steps', type=float, default=200000)
    parser.add_argument('--lr_clip', type=float, default=0.00001)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--init_bn_decay', type=float, default=0.5)
    parser.add_argument('--bn_decay_decay_rate', type=float, default=0.5)
    parser.add_argument('--bn_decay_decay_steps', type=int, default=200000)
    parser.add_argument('--bn_decay_clip', type=float, default=0.99)
    parser.add_argument('--steps_per_print', type=int, default=50)
    parser.add_argument('--steps_per_eval', type=int, default=100)
    parser.add_argument('--steps_per_plot', type=int, default=400)
    parser.add_argument('--steps_per_save', type=int, default=4000)
    parser.add_argument('--plot_lim', type=float, default=0.4)
    parser.add_argument('--plot_size', type=float, default=5)
    parser.add_argument('--iter_plot_freq', type=int, default=1)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(args.init_lr, global_step * args.batch_size,
                                               args.lr_decay_steps, args.lr_decay_rate,
                                               staircase=True, name='learning_rate')
    learning_rate = tf.maximum(learning_rate, args.lr_clip)
    bn_decay = 1 - tf.train.exponential_decay(args.init_bn_decay, global_step * args.batch_size,
                                              args.bn_decay_decay_steps, args.bn_decay_decay_rate,
                                              staircase=True, name='bn_decay')
    bn_decay = tf.minimum(bn_decay, args.bn_decay_clip)

    is_training_pl = tf.placeholder(tf.bool, (), 'is_training')
    points_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_points, 3), 'points')
    transforms_pl = tf.placeholder(tf.float32, (args.batch_size, 4, 4), 'transform')

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        transformer = importlib.import_module(args.transformer)
        transformed_points, T_out, Ts = transformer.get_model(points_pl, args.n_iter,
                                                              is_training_pl, bn_decay,
                                                              args.randinit, args.nostop)

    g_error = geometric_error(points_pl, T_out, transforms_pl)
    r_error = rotation_error(T_out[:, :3, :3], transforms_pl[:, :3, :3])
    t_error = translation_error(T_out[:, :3, 3], transforms_pl[:, :3, 3])
    loss_op = tf.reduce_mean(g_error)

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(loss_op, global_step)

    avg_loss, update0 = tf.metrics.mean(loss_op)
    percent_10deg, update1 = tf.metrics.percentage_below(r_error, 10)
    percent_01, update2 = tf.metrics.percentage_below(t_error, 0.1)
    update_ops = [update0, update1, update2]

    tf.summary.scalar('train/learning rate', learning_rate, collections=['train'])
    tf.summary.scalar('train/bn decay', bn_decay, collections=['train'])
    tf.summary.scalar('train/loss', loss_op, collections=['train'])
    tf.summary.scalar('valid/loss', avg_loss, collections=['valid'])
    tf.summary.scalar('valid/percent 10deg', percent_10deg, collections=['valid'])
    tf.summary.scalar('valid/percent 0.1', percent_01, collections=['valid'])

    train_summary = tf.summary.merge_all('train')
    valid_summary = tf.summary.merge_all('valid')

    lmdb_train = os.path.join(args.data_dir, 'train.lmdb')
    lmdb_valid = os.path.join(args.data_dir, 'test.lmdb')
    df_train, num_train = lmdb_dataflow(lmdb_train, args.batch_size, args.num_points,
                                        shuffle=True, task='pose')
    df_valid, num_valid = lmdb_dataflow(lmdb_valid, args.batch_size, args.num_points,
                                        shuffle=False, task='pose')
    train_gen = df_train.get_data()
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    writer = create_log_dir(args, sess)

    saver = tf.train.Saver()
    last_step = sess.run(global_step)
    for step in range(last_step, args.max_step):
        epoch = step * args.batch_size // num_train + 1
        instance_id, points, transforms = next(train_gen)
        _, loss, summary = sess.run([train_op, loss_op, train_summary],
                                    feed_dict={is_training_pl: True,
                                               points_pl: points,
                                               transforms_pl: transforms})
        writer.add_summary(summary, step + 1)
        if (step + 1) % args.steps_per_print == 0:
            print('Epoch %d  Step %d  Loss %f' % (epoch, step + 1, loss))
        if (step + 1) % args.steps_per_eval == 0:
            sess.run(tf.local_variables_initializer())
            for i in range(num_valid // args.batch_size):
                instance_id, points, transforms = next(valid_gen)
                ts, _ = sess.run([Ts, update_ops],
                                 feed_dict={is_training_pl: False,
                                            points_pl: points,
                                            transforms_pl: transforms})
                if (step + 1) % args.steps_per_plot == 0:
                    T = np.eye(4)
                    T_gt = transforms[0]
                    plot_ts = []
                    titles = []
                    for j in range(args.n_iter+1):
                        if j > 0:
                            T = np.dot(ts[j-1][0], T)
                        if j % args.iter_plot_freq == 0:
                            rerr = sess.run(rotation_error(T[:3, :3], T_gt[:3, :3]))
                            terr = sess.run(translation_error(T[:3, 3], T_gt[:3, 3]))
                            gerr = sess.run(geometric_error(points[0], T, T_gt))
                            plot_ts.append(T)
                            titles.append('Iteration %d\nRotation error %.4f\nTranslation error %.4f\n'
                                          'Geometric error %.4f' % (j, rerr, terr, gerr))
                    plot_ts.append(T_gt)
                    titles.append('Ground truth\nRotation error %.4f\nTranslation error %.4f\n'
                                  'Geometric error %.4f' % (0, 0, 0))
                    figpath = os.path.join(args.log_dir, 'plots', '%s_step_%d.png' % (instance_id[0], step + 1))
                    plot_iters(figpath, points[0], plot_ts, titles, args.plot_lim, args.plot_size)
            l, p1, p2 = sess.run([avg_loss, percent_10deg, percent_01])
            print(colored('Validation loss %f  percentage < 10 degree %.4f  percentage < 0.1 %.4f' % (l, p1, p2),
                          'white', 'on_blue'))
            summary = sess.run(valid_summary)
            writer.add_summary(summary, step + 1)
        if (step + 1) % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step + 1)
