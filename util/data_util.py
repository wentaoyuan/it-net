import numpy as np
from tensorpack import dataflow


def random_idx(m, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    perm = np.random.permutation(m)
    idx = perm.copy()
    while idx.shape[0] < n:
        idx = np.concatenate([idx, idx])
    return idx[:n], perm


def random_rotation():
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * np.pi
    A = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


class VirtualRenderData(dataflow.ProxyDataFlow):
    def __init__(self, ds):
        super(VirtualRenderData, self).__init__(ds)

    def get_data(self):
        for model_id, points, labels, cat_label in self.ds.get_data():
            pose = np.eye(4)
            R = random_rotation()
            pose[:3, :3] = R
            points = np.dot(points, R.T)
            d = np.ones((500, 500))
            ids = -np.ones((500, 500), dtype=np.int)
            for i, point in enumerate(points):
                x = int((point[0] + 0.5) / 0.02)
                y = int((point[1] + 0.5) / 0.02)
                z = point[2] + 0.5
                if z < d[x, y]:
                    d[x, y] = z
                    ids[x, y] = i
            ids = np.ravel(ids[ids > 0])
            points = points[ids]
            labels = labels[ids]
            pose[:3, 3] = -points.mean(axis=0)
            points -= points.mean(axis=0)
            yield model_id, points, labels, cat_label, pose


class ResampleData(dataflow.ProxyDataFlow):
    def __init__(self, ds, num_points, task):
        super(ResampleData, self).__init__(ds)
        self.num_points = num_points
        self.task = task

    def get_data(self):
        for data in self.ds.get_data():
            data = list(data)
            idx, _ = random_idx(data[1].shape[0], self.num_points)
            # For pose estimation, data = [id, points, pose]
            # For classfication, data = [id, points, cls_label, pose]
            # For segmentation, data = [id, points, part_labels, cls_label, pose]
            data[1] = data[1][idx]
            if self.task == 'seg':
                data[2] = data[2][idx]
            yield data


def lmdb_dataflow(lmdb_path, batch_size, num_points, shuffle, task, render=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if render:
        df = VirtualRenderData(df)
    if num_points is not None:
        df = ResampleData(df, num_points, task)
    if shuffle:
        df = dataflow.LocallyShuffleData(df, 1000)
        df = dataflow.PrefetchDataZMQ(df, 8)
    df = dataflow.BatchData(df, batch_size, use_list=True)
    df = dataflow.RepeatedData(df, -1)
    df.reset_state()
    return df, size
