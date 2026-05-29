import blenderproc as bproc
import argparse
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from PIL import Image


# TODO: generate apple with white background.
# TODO: generate apple with black background.
# TODO: save everything as video file.
# TODO: train your neural network prior.

# background: $ python3 -c "from PIL import Image;Image.new('RGB', (1900, 1080), color = (255,255,255)).save('Img.jpg')"


def main(args):
    bproc.init()

    # load the objects into the scene
    objs = bproc.loader.load_obj(args.obj)
    print(objs)

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # define the camera resolution
    bproc.camera.set_resolution(280, 280)

    for j in range(100):
        # set the location
        objs[0].set_location(np.random.uniform([0, 1, 2], [1, 2, 3])) # only one object each
        
        # Find point of interest, all cam poses should look towards it
        poi = bproc.object.compute_poi(objs)

        # Sample ten camera poses
        for i in range(25):
            # Sample random camera location above objects
            location = np.random.uniform([-3, -3, -0.5], [3, 3, 0.5])

            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)

    # activate normal and depth rendering
    # bproc.renderer.set_world_background([0, 0, 0])
    if args.background =="white":
        bproc.renderer.set_world_background([5, 5, 5])
    else:
        bproc.renderer.set_world_background([0, 0, 0])
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.set_noise_threshold(0.01)  # this is the default value

    # render the whole pipeline
    data = bproc.renderer.render()    
    images = data['colors']

    for i in range(len(images)): 
        img = Image.fromarray(images[i], 'RGB')
        # save the images with black background; create also the directory
        if not os.path.exists(args.output_dir + args.object + "/"):
            os.makedirs(args.output_dir + args.object + "/")
        outputname = args.output_dir + args.object + "/" +  args.object + '_' + str(i) + '.jpg'
        img.save(outputname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Lets try to learn a prior distribution")
    parser.add_argument("--object", type=str, default="unknown", help="Name of the object")
    parser.add_argument("--background", type=str, default="white", help="Background name")
    parser.add_argument("--obj", type=str, default="/home_local/lee_jn/synthetic/cad/object11/11.obj", help="Object file")
    parser.add_argument("--camera", type=str, default="/home_local/lee_jn/CLEVER/data/model/cad/camera_positions", help="Camera file")
    parser.add_argument("--output_dir", type=str, default="/home_local/lee_jn/synthetic/cad/object11/", help="Output directory")
    args = parser.parse_args()
    main(args)
