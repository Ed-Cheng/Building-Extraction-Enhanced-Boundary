'''
Note that this data loader is specifically for our dataset and need modify for other datasets
'''

import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image


def load_images(image_dir, mask_dir):
    '''load images and masks, decide to normalize images or not (0-1) and masks (0,1)'''
    image_dataset = []  
    mask_dataset = []

    # laod the data from files
    files_list = os.listdir(image_dir)
    for file_name in files_list:
        image_dataset.append(np.asarray(Image.open(os.path.join(image_dir, file_name)).convert('RGB').resize((256, 256))))
        mask_dataset.append(np.asarray(Image.open(os.path.join(mask_dir, file_name)).convert('RGB').resize((256, 256)))) 

    # Normalize the data
    image_dataset = np.array(image_dataset)/255
    mask_dataset =  np.array(mask_dataset)
    mask_dataset[mask_dataset <= 130] = 0
    mask_dataset[mask_dataset > 130] = 255

    # check dimension 
    if image_dataset.shape == mask_dataset.shape:
        print('image & masks match')
    else:
        print('dimension do not match, check data')

    return image_dataset, mask_dataset


def patchify_image(store_list, image, img_size=256):
    '''split large images into 256*256'''
    # img_size = 256

    # calculate the overlapping range to fully utilize the whole image
    step_size = img_size - int(np.ceil((img_size - image.shape[0]%img_size)/(image.shape[0]//img_size)))

    patches = patchify(image, (img_size, img_size, 3), step=step_size)
    # print(patches.shape)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            one_patch = patches[i,j,0,:,:]
            store_list.append(one_patch)

    return store_list

def load_Massachusett(image_dir, mask_dir, img_size=256):
    '''load Massachusett dataset'''

    image_dataset = []  
    mask_dataset = []

    # load the data from files
    files_list = os.listdir(image_dir)
    for file_name in files_list:
        image = np.array(Image.open(os.path.join(image_dir, file_name)).convert('RGB'))
        # slice the images to 256*256
        image_dataset = patchify_image(image_dataset, image, img_size)

        mask = np.array(Image.open(os.path.join(mask_dir, file_name)).convert('RGB'))
        # slice the masks to 256*256
        mask_dataset = patchify_image(mask_dataset, mask, img_size)

    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)

    # check dimension 
    if image_dataset.shape == mask_dataset.shape:
        print('image & masks match')
    else:
        print('dimension do not match, check data')
    
    return image_dataset, mask_dataset


def load_INRIA(image_dir, mask_dir, locations, type = 'train', img_size=256):
    '''load INRIA dataset with given locations'''

    image_dataset = []  
    mask_dataset = []

    # load the data from files
    files_list = os.listdir(image_dir)
    for loc in locations:
        # for each location, first 5 images are for testing and the rest for training
        file_idx = 0

        for file_name in files_list:
            if loc in file_name:
                # update the file index
                file_idx += 1
                # print(file_idx)

                # the test set
                if file_idx <= 5 and type == 'test':
                    # print(type)
                    image = np.array(Image.open(os.path.join(image_dir, file_name)))
                    # slice the images to 256*256
                    image_dataset = patchify_image(image_dataset, image, img_size)

                    mask = np.array(cv2.imread(os.path.join(mask_dir, file_name)))
                    # slice the masks to 256*256
                    mask_dataset = patchify_image(mask_dataset, mask, img_size)

                # the train set
                elif file_idx > 5 and type == 'train':
                    # print(type)
                    image = np.array(Image.open(os.path.join(image_dir, file_name)))
                    # slice the images to 256*256
                    image_dataset = patchify_image(image_dataset, image, img_size)

                    mask = np.array(cv2.imread(os.path.join(mask_dir, file_name)))
                    # slice the masks to 256*256
                    mask_dataset = patchify_image(mask_dataset, mask, img_size)

    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)

    # check dimension 
    print(f'image size: {image_dataset.shape}')
    print(f'mask size: {mask_dataset.shape}')
    
    return image_dataset, mask_dataset

