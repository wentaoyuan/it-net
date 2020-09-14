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
import os
import subprocess
from functools import partial
from multiprocessing.dummy import Pool
from termcolor import colored


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_path')
    parser.add_argument('intrinsics')
    parser.add_argument('output_dir')
    parser.add_argument('-n', type=int, default=1, help='number of trajectories')
    parser.add_argument('-p', type=int, default=8, help='number of processes')
    args = parser.parse_args()
    
    with open(os.path.join(args.list_path)) as file:
        model_list = [line.strip() for line in file]

    commands = [['/opt/blender/blender', '-b', '-P', 'render_single.py',
        model_id, args.intrinsics, args.output_dir, '%d' % args.n]
        for model_id in model_list]

    pool = Pool(args.p)
    print(colored('=== Rendering %d models on %d workers...' % (len(commands), args.p), 'white', 'on_blue'))
    for idx, completed in enumerate(pool.imap(partial(subprocess.run), commands)):
        pass
