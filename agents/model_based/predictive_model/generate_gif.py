import imageio
import os
import glob
import sys

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
GIF_DIR = os.path.join(PARENT_DIR, "gifs/")

if not os.path.exists(GIF_DIR):
    os.umask(0o000)
    os.makedirs(GIF_DIR)

def create_gif(image_folder, env_name):
    images = []

    for infile in sorted(glob.glob(image_folder + '/*.png')):
        image = imageio.imread(infile)
        images.append(image)

    imageio.mimsave(GIF_DIR + 'triple_comparison_' + env_name + '.gif',
                    images, fps=15)
