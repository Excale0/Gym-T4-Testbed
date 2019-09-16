import imageio
import os
import glob
import sys

def create_gif(env_name):
    images = []

    for infile in sorted(glob.glob('./images/*.png')):
        image = imageio.imread(infile)
        images.append(image)

    imageio.mimsave('./gifs/triple_comparison_' + env_name + '.gif',
                    images, fps=15)


if __name__ == "__main__":
    create_gif(sys.argv[1])
