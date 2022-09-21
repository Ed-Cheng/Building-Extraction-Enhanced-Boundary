import numpy as np
import random
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import albumentations as A


def batch_predict(pred_data, model, one_batch, labels=2):
    '''
    Prediction on a large dataset. Avoids resource exhaustion.
    '''
    print('Data size: ', pred_data.shape[0])
    batch_slice = pred_data.shape[0]//one_batch
    batch_size = np.linspace(0, pred_data.shape[0], num=batch_slice, dtype=int)

    # initialize a array to store the following predictions
    pred = np.zeros((1, pred_data.shape[1], pred_data.shape[2], labels))

    # predicting by batches
    for i in tqdm(range(batch_slice-1), desc="Predicting batches..."):
        pred_batch = model.predict(pred_data[batch_size[i]:batch_size[i+1]])
        pred = np.vstack((pred, pred_batch))

    # delete the blank row that we created
    pred = np.delete(pred, 0, 0)

    # # currently we are returning argmax value, can tune the threshold instead of takig the max
    # pred_argmax = np.argmax(pred, axis=3)

    return pred



def classwiseIoU(pred_argmax, true_argmax, model_path):
    """
    Return the per class Intersection over Union (I/U) from confusion matrix.
    Args:
        confusion: the confusion matrix between ground truth and predictions
    Returns:
        a vector representing the per class I/U
    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    
    # flattent the results to create a confusion matrix
    pred_argmax_flat = pred_argmax.flatten()
    true_argmax_flat = true_argmax.flatten()
    confusion = confusion_matrix(true_argmax_flat,pred_argmax_flat)

    # get |intersection| (AND) from the diagonal of the confusion matrix
    intersection = (confusion * np.eye(len(confusion))).sum(axis=-1)
    # calculate the total ground truths and predictions per class
    preds = confusion.sum(axis=0)
    trues = confusion.sum(axis=-1)
    # get |union| (OR) from the predictions, ground truths, and intersection
    union = trues + preds - intersection
    # The intersection over the union
    class_IoU = intersection/union

    print('The classwise IOU:', np.round(class_IoU, 4))
    print('The average IOU:', np.average(class_IoU))
    print(f'---------- Results of {model_path} ----------')



def prepocess_image(image, mask, scale, norm=True):
    '''
    https://towardsdatascience.com/image-data-augmentation-pipeline-9841bc7bb56d
    Data augmentation by rotating images. Image values must be 0-255. Choose if need to normalize.
    '''
    print(f'Before prepocessing: {image.shape}, {mask.shape}')

    if scale != 1:
        total_images = len(image)

        expand_image = np.vstack([image]*scale)
        expand_labels = np.vstack([mask]*scale)
        
        for i in range(total_images):
            for repetition in range(1, scale):   # 1 is original img, anything above is a rotation
                rotate_angle = random.randint(10, 360)
                transform = A.Compose([A.Rotate(
                    limit=[rotate_angle, rotate_angle+1], 
                    border_mode=cv2.BORDER_CONSTANT,
                    always_apply=True)])

                # Apply transformation
                img_transformed = transform(image=image[i])["image"]
                msk_transformed = transform(image=mask[i])["image"]

                expand_image[i + total_images*repetition] = img_transformed
                expand_labels[i + total_images*repetition] = msk_transformed
    else:
        expand_image = image
        expand_labels = mask
    
    if norm:
        expand_labels = expand_labels/255
        expand_labels[expand_labels <= 0.5] = 0
        expand_labels[expand_labels > 0.5] = 1

    print(f'After prepocessing: {expand_image.shape}, {expand_labels.shape}')

    return expand_image, expand_labels

def plot4images(data1, data2, data3, data4, idx_list):
    '''
    Plot 4 images side by side. data1-data4 is a list, eg. [images, image label]
    '''
    for img_idx in idx_list:
        print(img_idx)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(141)
        plt.title(data1[1])
        plt.imshow(data1[0][img_idx])

        plt.subplot(142)
        plt.title(data2[1])
        plt.imshow(data2[0][img_idx])

        plt.subplot(143)
        plt.title(data3[1])
        plt.imshow(data3[0][img_idx])

        plt.subplot(144)
        plt.title(data4[1])
        plt.imshow(data4[0][img_idx])
        plt.show()


def plot6images(data1, data2, data3, data4, data5, data6, idx_list):
    '''
    Plot 6 images side by side. data1-data6 is a list, eg. [images, image label]
    '''
    for img_idx in idx_list:
        print(img_idx)
        
        plt.figure(figsize=(20, 8))
        plt.subplot(161)
        plt.title(data1[1])
        plt.imshow(data1[0][img_idx])

        plt.subplot(162)
        plt.title(data2[1])
        plt.imshow(data2[0][img_idx])

        plt.subplot(163)
        plt.title(data3[1])
        plt.imshow(data3[0][img_idx])

        plt.subplot(164)
        plt.title(data4[1])
        plt.imshow(data4[0][img_idx])

        plt.subplot(165)
        plt.title(data5[1])
        plt.imshow(data5[0][img_idx])

        plt.subplot(166)
        plt.title(data6[1])
        plt.imshow(data6[0][img_idx])
        plt.show()