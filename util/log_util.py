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
