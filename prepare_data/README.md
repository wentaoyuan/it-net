This directory contains code that generates partial point clouds from [ModelNet](https://modelnet.cs.princeton.edu) models. To use it:
1. Install [Blender](https://blender.org/download) and [Blender OFF Addon](https://github.com/alextsui05/blender-off-addon).
2. Download aligned [ModelNet40](https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar).
3. Create a list of model IDs. Each line of the model list should have the format `airplane_0001`.
4. Run `python parallel_render.py [model list] [intrinsics file] [output directory]` to render depth scan trajectories. The depth scans will be stored in OpenEXR format. Use `-n` to control the number of scan trajectories to generate for each model and `-p` to control the number of parallel threads.
5. Run `python traj2pcd.py [model list] [intrinsics file] [depth trajectory directory] [output directory] [number of trajectories per model]` to fuse the depth scans into partial point clouds in the initial camera's coordinate frame.
6. There are a couple of parameters such as distance from the camera to the object center and number of scans per trajectories in `render_single.py` that can be adjusted.
