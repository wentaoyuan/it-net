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
