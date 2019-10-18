import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d import *


def read_exr(exr_path, width, height):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    return depth


def depth2pcd(depth, intrinsics, pose=None):
    # Camera coordinate system in Blender is x: right, y: up, z: inwards
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    if pose is not None:
        points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0))[:3, :]
    return points.T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_file')
    parser.add_argument('intrinsics_file')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('num_traj', type=int)
    args = parser.parse_args()

    with open(args.list_file) as file:
        model_list = file.read().splitlines()
    intrinsics = np.loadtxt(args.intrinsics_file)
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    for model_id in model_list:
        category = model_id.rsplit('_', 1)[0]
        in_dir = os.path.join(args.input_dir, category, model_id)
        out_dir = os.path.join(args.output_dir, category, model_id)
        num_scans = np.loadtxt(os.path.join(in_dir, 'num_scans.txt'), dtype=np.int)
        if len(num_scans.shape) == 0:
            num_scans = [num_scans]
        os.makedirs(out_dir, exist_ok=True)
        for i in range(args.num_traj):
            for j in range(num_scans[i]):
                depth = read_exr(os.path.join(in_dir, '%d/%d.exr' % (i, j)), width, height)
                depth[np.isinf(depth)] = 0

                pose = np.loadtxt(os.path.join(in_dir, '%d/%d.txt' % (i, j)))
                if j == 0:
                    pose0 = pose
                    points = depth2pcd(depth, intrinsics)
                else:
                    points = np.concatenate([points,
                        depth2pcd(depth, intrinsics, np.dot(np.linalg.inv(pose0), pose))], axis=0)

            if points.shape[0] == 0:
                print(model_id, i, j)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # print(num_scans[i], points.shape[0], points.min(0), points.max(0))
            # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
            # ax.set_xlim(-1, 1)
            # ax.set_ylim(-1, 1)
            # ax.set_zlim(-3, -1)
            # plt.show()

            pcd = PointCloud()
            pcd.points = Vector3dVector(points)
            write_point_cloud(os.path.join(out_dir, '%d.pcd' % i), pcd)
            np.savetxt(os.path.join(out_dir, '%d.txt' % i), np.linalg.inv(pose0), '%.20f')
