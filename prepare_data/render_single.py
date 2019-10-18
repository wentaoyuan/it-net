import bpy
import addon_utils
from mathutils import Matrix
import numpy as np
import os
import sys
import time
addon_utils.enable('import_off')


def setup_blender(width, height, focal_length):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.sensor_height = camera.data.sensor_width
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2

    # render layer
    scene = bpy.context.scene
    scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.use_zbuffer = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['Depth'], output.inputs[0])

    # remove default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

    return scene, camera, output


if __name__ == '__main__':
    model_dir = sys.argv[-5]
    model_id = sys.argv[-4]
    intrinsics_file = sys.argv[-3]
    output_dir = sys.argv[-2]
    num_trajs = int(sys.argv[-1])

    max_num_scans = 4
    min_dist = 2
    max_dist = 4

    intrinsics = np.loadtxt(intrinsics_file)
    focal = intrinsics[0, 0]
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)
    scene, camera, output = setup_blender(width, height, focal)

    category = model_id.rsplit('_', 1)[0]
    output_dir = os.path.join(output_dir, category, model_id)
    os.makedirs(output_dir, exist_ok=True)
    output.base_path = output_dir
    scene.render.filepath = os.path.join(output_dir, 'buffer.png')
    log_path = os.path.join(output_dir, 'blender_render.log')

    # Redirect output to log file
    open(log_path, 'a').close()
    old = os.dup(1)
    os.close(1)
    os.open(log_path, os.O_WRONLY)

    start = time.time()
    model_path = os.path.join(model_dir, category, '%s.off' % model_id)
    bpy.ops.import_mesh.off(filepath=model_path)

    num_scans = np.random.randint(1, max_num_scans+1, num_trajs)
    np.savetxt(os.path.join(output_dir, 'num_scans.txt'), np.array(num_scans), '%d')

    for i in range(num_trajs):
        traj_dir = os.path.join(output_dir, '%d' % i)
        os.makedirs(traj_dir, exist_ok=True)
        output.file_slots[0].path = os.path.join('%d/#.exr' % i)

        axis = np.random.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        angle = np.random.rand() * np.pi
        rot = Matrix.Rotation(angle, 4, axis)

        dist = np.random.uniform(min_dist, max_dist)
        R = np.array(rot)
        trans = Matrix.Translation(R[:3, 2] * dist)

        for j in range(num_scans[i]):
            camera.matrix_world = trans * rot
            np.savetxt(os.path.join(traj_dir, '%d.txt' % j),
                       np.array(camera.matrix_world), '%.20f')

            scene.frame_set(j)
            bpy.ops.render.render(write_still=True)
            
            axis = np.random.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            angle = np.random.rand() * np.pi / 6
            rot *= Matrix.Rotation(angle, 4, axis)
            R = np.array(rot)
            trans = Matrix.Translation(R[:3, 2] * dist)

    # clean up
    bpy.ops.object.delete()

    # show time
    os.close(1)
    os.dup(old)
    os.close(old)
    print('{1} done, time={0:.4f} sec'.format(time.time() - start, model_id))
