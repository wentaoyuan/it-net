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

import numpy as np
import os


def read_off(filepath):
    with open(filepath) as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, _ = [int(s) for s in file.readline().strip().split(' ')]
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i in range(n_faces)]
        return np.array(verts), np.array(faces)


def write_off(filepath, verts, faces):
    with open(filepath, 'w') as file:
        file.write('OFF\n')
        file.write('%d %d 0\n' % (verts.shape[0], faces.shape[0]))
        for vert in verts:
            file.write('%f %f %f\n' % tuple(vert))
        for face in faces:
            file.write('3 %d %d %d\n' % tuple(face))


input_dir = '/usr0/home/Datasets/modelnet40_aligned'
output_dir = '/usr0/home/Datasets/modelnet40_normalized'
mode = 'train'
with open('%s/%s.txt' % (input_dir, mode)) as file:
    model_list = file.read().splitlines()

for model_id in model_list:
    cat = model_id.rsplit('_', 1)[0]
    verts, faces = read_off('%s/%s/%s/%s.off' % (input_dir, cat, mode, model_id))
    verts -= (verts.max(0) + verts.min(0)) / 2
    verts /= np.linalg.norm(verts, axis=1).max()
    os.makedirs('%s/%s' % (output_dir, cat), exist_ok=True)
    write_off('%s/%s/%s.off' % (output_dir, cat, model_id), verts, faces)
