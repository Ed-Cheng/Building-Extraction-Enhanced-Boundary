import cv2
import numpy as np


def coloured_contours(img, thick):
    #convert img to grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #create an empty image for contours
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image  
    cv2.drawContours(img_contours, contours, -1, color=(120, 0, 0), thickness=thick)

    img_contours = np.where(img == 255, img, img_contours)

    return img_contours

def rgb_to_2D_label(label):
    """
    Supply our labal masks as input in RGB format. 
    Replace pixels with specific RGB values
    building = 0, background = 1, added border = 2
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg[np.all(label == 255,axis=-1)] = 0
    label_seg[np.all(label < 255,axis=-1)] = 2
    label_seg[np.all(label == 0,axis=-1)] = 1
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

def preprocess_mask(mask, thick):
    '''
    Pre-process the images with thickened border
    '''
    # add a thick border on the mask if required
    if thick != 0:
        boundary = []
        for _, image in enumerate(mask):
            boundary.append(coloured_contours(image, thick))

        mask = np.array(boundary)

    print("Unique values in mask dataset are: ", np.unique(mask[:50]))

    labels = []
    for i in range(mask.shape[0]):
        label = rgb_to_2D_label(mask[i])
        labels.append(label)  

    labels = np.array(labels)   
    labels = np.expand_dims(labels, axis=3)

    print("Unique labels in label dataset are: ", np.unique(labels[:50]))

    return labels
