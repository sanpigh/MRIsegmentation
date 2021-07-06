import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_io as tfio
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def tversky(y_true, y_pred, smooth=1.0e-7, alpha=0.7, beta=0.3):
    """
    Compute the Tversky score.

    The Tversky index, named after Amos Tversky, is an asymmetric similarity measure on sets
    that compares a prediction to a ground truth.

    :y_true: the ground truth mask
    :y_pred: the predicted mask
    :return: The tversky score
    :rtype: float

    :Example:

    >>> tversky([0, 0, 0], [1, 1, 1])
    smooth / (smooth + 3*alpha)
    >>> tversky([1, 1, 1], [1, 1, 1]))
    1

    .. seealso:: tensorflow.keras.metrics.MeanIoU
                 https://en.wikipedia.org/wiki/Tversky_index
    .. warning:: smooth needs to adapt to the size of the mask
    .. note:: Setting alpha=beta=0.5 produces the Sørensen–Dice coefficient.
              alpha=beta=1 produces the Tanimoto coefficient aka Jaccard index
    .. todo:: Better understand how to set the parameters
    """
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + beta * false_pos + smooth
    )


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def process_path(mri_path, mask_path):
    """
    Load images from files.

    :mri_path: the path to the mri file
    :mask_path: the path to the mask file
    :return: The image and mask
    :rtype: [H, W, 3 RGB] array and [H, W] binary array


    .. note:: Works with TIFF images whhich are read by TF as RGBA arrays
              A is dropped for the images, and only the first channel is kept for the mask
    """

    mri_img = tfio.experimental.image.decode_tiff(tf.io.read_file(mri_path))
    mri_img = mri_img[:, :, :-1]
    mask_img = tfio.experimental.image.decode_tiff(tf.io.read_file(mask_path))
    mask_img = mask_img[:, :, :-1]

    # . for label processisng use tf.strings.[split, substr, to_number]
    # tf.strings.split()

    return mri_img, mask_img


def normalize(image, mask):
    # image = tf.cast(image, tf.float32) / 255.

    return tf.math.divide(image, 255), tf.math.divide(mask, 255)


def augment_data(image, mask):
    """
    Take one image and its mask and apply one or several transformations
    Return image and mask
    """
    choice = tf.random.uniform((), minval=0, maxval=1)
    if True and choice < 0.5:
        brightness_val = tf.random.uniform((), minval=-0.3, maxval=0.3)
        image = tf.image.adjust_brightness(image, delta=brightness_val)
        return image, mask

    choice = tf.random.uniform((), minval=0, maxval=1)
    if True and choice < 0.5:
        contrast_val = tf.random.uniform((), minval=0.40, maxval=0.45)
        image = tf.image.adjust_contrast(image, contrast_factor=contrast_val)
        return image, mask

    choice = tf.random.uniform((), minval=0, maxval=1)
    if True and choice < 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        return image, mask

    choice = tf.random.uniform((), minval=0, maxval=1)
    if True and choice < 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
        return image, mask

    choice = tf.random.uniform((), minval=0, maxval=1)
    if True and choice < 0.5:
        angle = tf.random.uniform((), minval=-0.5, maxval=0.5)
        image = tfa.image.rotate(image, angles=angle)
        mask = tfa.image.rotate(mask, angles=angle)
        return image, mask

    choice = tf.random.uniform((), minval=0, maxval=1)
    if False and choice < 0.5:
        crop_val = tf.random.uniform((), minval=0.8, maxval=0.9)
        image = tf.image.central_crop(image, central_fraction=crop_val)
        mask = tf.image.central_crop(mask, central_fraction=crop_val)
        return image, mask

    return image, mask


def flatten_mask(image, mask):
    """
    Flatten the mask to a binary value
    Image is left unchanged
    """
    return image, mask[:, :, 0]

def Dataviz(data_set, number_of_samples, model=True, threshold=1, ascending= True):
    ''' takes one dataset and shows the original images with their predicted mask and tversky_score
    it can also show only the original images and their masks if the parameter model is set to None '''
    if model == None:

        for image, mask in data_set.map(process_path).map(flatten_mask).map(normalize).take(number_of_samples):

            image = image.numpy()
            mask = mask.numpy()

            fig, axs = plt.subplots(1, 3, figsize=(15, 15))
            
            axs[0].set_title("Original Image",fontsize=14)
            axs[0].get_xaxis().set_visible(False)
            axs[0].get_yaxis().set_visible(False)
            axs[0].imshow(image)

            axs[1].set_title("Original Mask", fontsize=14)
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
            axs[1].imshow(mask, cmap="gray")

            axs[2].set_title("Original MRI with Mask", fontsize=14)
            axs[2].get_xaxis().set_visible(False)
            axs[2].get_yaxis().set_visible(False)
            image[mask == 1] = (255, 0,0)
            axs[2].imshow(image)
            fig.tight_layout(pad=5)
    else:
        image_list=[]
        mask_list=[]

        for image, mask in data_set.take(number_of_samples):
            image_list.append(image.numpy().decode())
            mask_list.append(mask.numpy().decode())
        df_test=pd.DataFrame({"image_path": image_list, "mask_path": mask_list})
        df_test['Tversky_score']=None

        for index, row in df_test.iterrows():
            image=row['image_path']
            mask=row['mask_path']
            image, mask=normalize(*flatten_mask(*process_path(image, mask)))
            mask_p = model.predict(np.expand_dims(image, axis=0))
            mask_p_binary = mask_p[0, :, :, 0] >= threshold
            mask_p_binary = np.float32(mask_p_binary)
            image = image.numpy()
            mask = mask.numpy()
            score=np.round(tversky(mask, mask_p_binary).numpy(), 2)
            print(index)
            df_test['Tversky_score'][index]=score
            df_test=df_test.sort_values(by=['Tversky_score'], ascending=ascending)
        ds_test = tf.data.Dataset.from_tensor_slices((df_test["image_path"].values, df_test["mask_path"].values))

        for image, mask in ds_test.map(process_path).map(flatten_mask).map(normalize).take(number_of_samples):

            mask_p = model.predict(np.expand_dims(image, axis=0))
            mask_p_binary = mask_p[0, :, :, 0] >= threshold
            mask_p_binary = np.float32(mask_p_binary)
            image = image.numpy()
            mask = mask.numpy()
            score=np.round(tversky(mask, mask_p_binary).numpy(), 2)
            
            fig, axs = plt.subplots(1, 4, figsize=(15, 15))
            
            axs[0].set_title("Original Image", fontsize=14)
            axs[0].get_xaxis().set_visible(False)
            axs[0].set_yticklabels([])
            axs[0].set_ylabel(f"Tversky score = {score:.2f}",  fontsize=12, color='C0', weight='bold')
            axs[0].imshow(image)

            axs[1].set_title("Original Mask", fontsize=14)
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
            axs[1].imshow(mask, cmap="gray")

            axs[2].set_title("Predicted Mask", fontsize=14)
            axs[2].get_xaxis().set_visible(False)
            axs[2].get_yaxis().set_visible(False)
            axs[2].imshow(mask_p_binary, cmap="gray")
            
            intersect=np.stack((mask, mask_p_binary, mask*0), axis=-1)*255
            intersect=intersect.astype(int)
            axs[3].set_title("Original vs. Predicted mask",fontsize=14)
            axs[3].get_xaxis().set_visible(False)
            axs[3].get_yaxis().set_visible(False)
            axs[3].imshow(intersect)
            fig.tight_layout(pad=2)
