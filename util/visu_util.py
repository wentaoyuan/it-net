import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

part_colormap = np.array([[0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65],
    [0.95, 0.95, 0.65], [0.95, 0.65, 0.05], [0.35, 0.05, 0.05], [0.65, 0.05, 0.05],
    [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], [0.65, 0.05, 0.35], [0.05, 0.35, 0.35],
    [0.65, 0.65, 0.35], [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35],
    [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05],
    [0.95, 0.65, 0.95], [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65],
    [0.95, 0.35, 0.95], [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65],
    [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], [0.05, 0.35, 0.95],
    [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05],
    [0.05, 0.65, 0.35], [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95],
    [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65],
    [0.35, 0.35, 0.65], [0.65, 0.95, 0.35], [0.05, 0.95, 0.65], [0.65, 0.65, 0.95],
    [0.35, 0.05, 0.95], [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]])


def plot_pcd(ax, pcd, size, azim, elev, lim, color=None, cmap='Blues'):
    if color is None:
        color = pcd[:, 0]
        vmax = color.max()
        vmin = color.min() - (color.max() - color.min()) / 2
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color, s=size, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color, s=size)
    ax.view_init(elev, azim)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-lim, lim))


def plot_iters(figpath, pcd, transforms, titles, lim, size, suptitle=None, axis_on=False):
    fig = plt.figure(figsize=(len(transforms) * 4, 3 * 4))
    if not isinstance(lim, list):
        lim = [lim] * len(transforms)
    for i in range(3):
        azim = 80 * i - 30
        elev = 15 * (i+1)
        for j in range(len(transforms)):
            ax = fig.add_subplot(3, len(transforms), i*len(transforms)+j+1, projection='3d')
            pcd_trans = np.dot(np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1), transforms[j].T)
            plot_pcd(ax, pcd_trans, size, azim, elev, lim[j])
            if axis_on:
                for axis in 'xyz':
                    getattr(ax, 'set_{}ticks'.format(axis))([-lim[j], 0, lim[j]])
                ax.tick_params(labelsize='large')
            else:
                ax.set_axis_off()
            if i == 0:
                ax.set_title(titles[j], fontsize=20)
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0, hspace=0)
    fig.savefig(figpath)
    plt.close(fig)


def plot_iters_seg(figpath, pcd, transforms, part_ids, titles, lim, size, suptitle=None, axis_on=False):
    fig = plt.figure(figsize=(len(transforms) * 4, 3 * 4))
    if not isinstance(lim, list):
        lim = [lim] * len(transforms)
    for i in range(3):
        azim = 80 * i - 30
        elev = 15 * (i+1)
        for j in range(len(transforms)):
            ax = fig.add_subplot(3, len(transforms), i*len(transforms)+j+1, projection='3d')
            pcd_trans = np.dot(np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1), transforms[j].T)
            plot_pcd(ax, pcd_trans, size, azim, elev, lim[j], part_colormap[part_ids[j]])
            if axis_on:
                for axis in 'xyz':
                    getattr(ax, 'set_{}ticks'.format(axis))([-lim[j], 0, lim[j]])
                ax.tick_params(labelsize='large')
            else:
                ax.set_axis_off()
            if i == 0:
                ax.set_title(titles[j], fontsize=20)
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0)
    fig.savefig(figpath)
    plt.close(fig)


def plot_conf(figpath, conf):
    plt.figure(figsize=(6, 6))
    plt.matshow(conf)
    plt.colorbar()
    plt.xlabel('Ground truth', verticalalignment='top')
    plt.ylabel('Prediction')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(figpath)


def plot_cdf(figpath, data, label):
    data = sorted(data)
    fig = plt.figure()
    plt.plot(data, np.arange(0, 1, 1 / len(data)))
    plt.ylabel('Percentage', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.xlabel(label, fontsize=18)
    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.98)
    plt.savefig(figpath)
    plt.close(fig)


def plot_mean_std(figpath, x, y, label):
    mean = np.mean(y, axis=1)
    std = np.std(y, axis=1)
    fig = plt.figure()
    plt.plot(x, mean)
    plt.fill_between(x, mean-std, mean+std, alpha=0.3)
    plt.xlabel('Iteration', fontsize=18)
    plt.xticks(x)
    plt.ylabel(label, fontsize=18)
    plt.ylim(bottom=0)
    plt.tick_params(labelsize=18)
    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.98)
    plt.savefig(figpath)
    plt.close(fig)
