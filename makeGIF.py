# -*- coding:utf-8 -*-
import imageio

def create_gif(image_list, gif_name, duration=1.0):
    '''Generate gif file using a list of images

    Args:
        image_list (list): A list of all image names.
        gif_name (str): GIF file name. (xxx.gif) 
        duration (float): Duration of each image. 

    Return:
        None
    '''
    import warnings
    warnings.filterwarnings('ignore')
    print('Collecting images for GIF...')
    frames = [imageio.imread(image_name) for image_name in image_list]
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


if __name__ == '__main__':
    import os

    root = '/Volumes/GoogleDrive/Other computers/Lab/Feiyi_GoogleDrive/Feiyi_images'
    calibRange = (927, 959)
    calibImagePath = []
    for i in range(calibRange[0], calibRange[1]+1):
        path = os.path.join(root, '000'+str(i)+'.pgm')
        calibImagePath.append(path)

    calibImagePath.sort()
    gif_name = 'whiteImages' + '.gif'

    create_gif(calibImagePath, gif_name, 0.1)
