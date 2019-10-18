import os
import tensorflow as tf
from termcolor import colored


def create_log_dir(args, sess):
    if args.restore:
        restorer = tf.train.Saver()
        if args.checkpoint is not None:
            restorer.restore(sess, args.checkpoint)
        else:
            restorer.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        writer = tf.summary.FileWriter(args.log_dir)
    else:
        if os.path.exists(args.log_dir):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                        % args.log_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % args.log_dir)
                os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    return writer
