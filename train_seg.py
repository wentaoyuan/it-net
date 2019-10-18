import argparse
import importlib
import numpy as np
import tensorflow as tf
from termcolor import colored

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'segmenter'))
sys.path.append(os.path.join(BASE_DIR, 'transformer'))
sys.path.append(os.path.join(BASE_DIR, 'util'))
from data_util import lmdb_dataflow
from log_util import create_log_dir
from visu_util import plot_iters_seg

n_classes = 16
n_parts = 50
parts_idx = {'02691156': 0, '02773838': 4, '02954340': 6, '02958343': 8,
             '03001627': 12, '03261776': 16, '03467517': 19, '03624134': 22,
             '03636649': 24, '03642806': 28, '03790512': 30, '03797390': 36,
             '03948459': 38, '04099429': 41, '04225987': 44, '04379243': 47}


def mean_iou(pred, labels, num_classes):
    conf = np.zeros((num_classes, num_classes))
    for i in range(pred.shape[0]):
        conf[pred[i], labels[i]] += 1
    tp = np.diag(conf)
    total = np.sum(conf, 0) + np.sum(conf, 1) - tp
    return np.mean(tp[total > 0] / total[total > 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--segmenter', choices=['pointnet', 'dgcnn'])
    parser.add_argument('--transformer', choices=['t_net', 'it_net', 'it_net_dgcnn'])
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_parts', type=int, default=50)
    parser.add_argument('--init_lr', type=float, default=0.0005)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_decay_steps', type=int, default=200000)
    parser.add_argument('--lr_clip', type=float, default=0.00001)
    parser.add_argument('--init_bn_decay', type=float, default=0.5)
    parser.add_argument('--bn_decay_decay_rate', type=float, default=0.5)
    parser.add_argument('--bn_decay_decay_steps', type=int, default=200000)
    parser.add_argument('--bn_decay_clip', type=float, default=0.99)
    parser.add_argument('--grad_clip', type=float, default=30.0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--epoch_per_save', type=int, default=50)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=500)
    parser.add_argument('--steps_per_plot', type=int, default=3000)
    parser.add_argument('--val_steps_per_plot', type=int, default=7)
    parser.add_argument('--plot_lim', type=float, default=0.5)
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
    labels_pl = tf.placeholder(tf.int32, (args.batch_size, args.num_points), 'labels')
    cat_labels_pl = tf.placeholder(tf.int32, (args.batch_size,), 'cat_labels')

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        transformer = importlib.import_module(args.transformer)
        transformed_points, T_out, Ts = transformer.get_model(points_pl, args.n_iter,
                                                              is_training_pl, bn_decay)
    with tf.variable_scope('segmenter'):
        segmenter = importlib.import_module(args.segmenter)
        logits = segmenter.get_model(transformed_points, cat_labels_pl, is_training_pl, bn_decay)
        prediction = tf.argmax(logits, axis=2)

    loss_op = segmenter.get_loss(logits, labels_pl)

    trainer = tf.train.AdamOptimizer(learning_rate)
    grad, var = zip(*trainer.compute_gradients(loss_op, tf.trainable_variables()))
    grad, global_norm = tf.clip_by_global_norm(grad, args.grad_clip)
    train_op = trainer.apply_gradients(zip(grad, var), global_step)

    avg_loss, loss_update = tf.metrics.mean(loss_op)
    avg_acc, acc_update = tf.metrics.accuracy(labels_pl, prediction)
    avg_iou, iou_update = tf.metrics.mean_iou(labels_pl, prediction, n_parts)

    tf.summary.scalar('train/learning rate', learning_rate, collections=['train'])
    tf.summary.scalar('train/gradient norm', global_norm, collections=['train'])
    tf.summary.scalar('train/bn decay', bn_decay, collections=['train'])
    tf.summary.scalar('valid/loss', avg_loss, collections=['valid'])
    tf.summary.scalar('valid/accuracy', avg_acc, collections=['valid'])
    tf.summary.scalar('valid/mean iou', avg_iou, collections=['valid'])
    update_ops = [loss_update, acc_update, iou_update]
    train_summary = tf.summary.merge_all('train')
    valid_summary = tf.summary.merge_all('valid')

    lmdb_train = os.path.join(args.data_dir, 'train.lmdb')
    lmdb_valid = os.path.join(args.data_dir, 'valid.lmdb')
    df_train, num_train = lmdb_dataflow(lmdb_train, args.batch_size, args.num_points,
                                        shuffle=True, render=True, task='seg')
    df_valid, num_valid = lmdb_dataflow(lmdb_valid, args.batch_size, args.num_points,
                                        shuffle=False, task='seg')
    train_gen = df_train.get_data()
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    writer = create_log_dir(args, sess)

    saver = tf.train.Saver()
    step = sess.run(global_step)
    epoch = step * args.batch_size // num_train + 1
    next_epoch = epoch
    while next_epoch <= args.max_epoch:
        step += 1
        epoch = step * args.batch_size // num_train + 1
        model_id, points, labels, cat_labels, poses = next(train_gen)
        _, loss, summary = sess.run([train_op, loss_op, train_summary],
                                    feed_dict={is_training_pl: True,
                                               points_pl: points,
                                               labels_pl: labels,
                                               cat_labels_pl: cat_labels})
        writer.add_summary(summary, step)
        if step % args.steps_per_print == 0:
            print('Epoch %d  Step %d  Loss %f' % (epoch, step, loss))
        if step % args.steps_per_eval == 0:
            sess.run(tf.local_variables_initializer())
            for i in range(num_valid // args.batch_size):
                instance_id, points, labels, cat_labels, poses = next(valid_gen)
                ts, pred, _ = sess.run([Ts, prediction, update_ops],
                                       feed_dict={is_training_pl: False,
                                                  points_pl: points,
                                                  labels_pl: labels,
                                                  cat_labels_pl: cat_labels})
                if step % args.steps_per_plot == 0 and (i + 1) % args.val_steps_per_plot == 0:
                    synset_id = instance_id[0].split('_')[0]
                    T = np.eye(4)
                    transforms = []
                    part_ids = []
                    titles = []
                    for j in range(args.n_iter+1):
                        if j > 0:
                            T = np.dot(ts[j-1][0], T)
                        if j % args.iter_plot_freq == 0:
                            transforms.append(T)
                            part_ids.append(pred[0])
                            titles.append('Iteration %d' % j)
                    titles[0] = 'Input'
                    transforms.append(T)
                    part_ids.append(labels[0])
                    titles.append('Ground truth')
                    acc = np.sum(pred[0] == labels[0]) / args.num_points
                    iou = mean_iou(pred[0], labels[0], n_parts)
                    suptitle = 'Accuracy %.4f  Mean IOU %.4f' % (acc, iou)
                    figpath = '%s/plots/%s_step_%d.png' % (args.log_dir, instance_id[0], step)
                    plot_iters_seg(figpath, points[0], transforms, part_ids, titles,
                                   args.plot_lim, args.plot_size, suptitle)
            loss, acc, iou = sess.run([avg_loss, avg_acc, avg_iou])
            print(colored('Loss %f  Accuracy %.4f  Mean IOU %.4f' % (loss, acc, iou), 'white', 'on_blue'))
            summary = sess.run(valid_summary)
            writer.add_summary(summary, step)
        next_epoch = (step+1) * args.batch_size // num_train + 1
        if epoch % args.epoch_per_save == 0 and epoch < next_epoch:
            saver.save(sess, os.path.join(args.log_dir, 'model_epoch_%d' % epoch))
