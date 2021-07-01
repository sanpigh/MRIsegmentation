import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_io as tfio
import matplotlib.pyplot as plt


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


def flatten_mask(image, mask):
    """
    Flatten mask from [H, W, RGB] to [H, W]. Leave image unchanged.

    :image: the image (3 RGB channels)
    :mask: the mask (3 RGB channels)
    :return: The image and mask
    :rtype: [H, W, 3 RGB] array and [H, W] binary array
    """

    mask = mask[:, :, 0]

    # . for label processing use tf.strings.[split, substr, to_number]
    # tf.strings.split()

    return image, mask


def normalize(image, mask):
    return tf.math.divide(image, 255), tf.math.divide(mask, 255)


def augment_data(tf_dataset, number_of_samples):

    for image, mask in tf_dataset.take(number_of_samples):
        image = tfio.experimental.image.decode_tiff(tf.io.read_file(image))
        image = image[:, :, :-1]
        mask = tfio.experimental.image.decode_tiff(tf.io.read_file(mask))
        mask = mask[:, :, :-1]

        if tf.random.uniform((), minval=0, maxval=1) <= 0.1:
            image = tf.image.adjust_brightness(image, delta=0.1)
            mask = tf.image.adjust_brightness(mask, delta=0.1)
            print("brightness")

        elif tf.random.uniform((), minval=0, maxval=1) <= 0.2:
            image = tf.image.adjust_contrast(image, contrast_factor=0.1)
            mask = tf.image.adjust_contrast(mask, contrast_factor=0.1)
            print("contrast")

        elif tf.random.uniform((), minval=0, maxval=1) <= 0.4:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
            print("flip L/R")

        elif tf.random.uniform((), minval=0, maxval=1) <= 0.6:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
            print("flip up/down")

        elif tf.random.uniform((), minval=0, maxval=1) <= 0.7:
            image = tfa.image.rotate(image, angles=0.5)
            mask = tfa.image.rotate(mask, angles=0.5)
            print("rotate")

        else:
            image = image
            mask = mask
            print("no augmentation")

        return image, mask


def dataviz_image_and_mask(tf_dataset, number_of_samples):
    """Import process_path and call the function following the example below:
    from MRIsegmentation.utils import process_path
    dataviz_image_and_mask(ds_train.map(process_path), 5)"""

    for image, mask in tf_dataset.take(number_of_samples):
        fig, axs = plt.subplots(1, 3, figsize=(8, 15))
        image = image.numpy()
        mask = mask.numpy()

        axs[0].set_title("Image")
        axs[0].imshow(image)

        axs[1].set_title("Mask")
        axs[1].imshow(mask, cmap="gray")

        axs[2].set_title("MRI with Mask")
        image[mask == 255] = (255, 0, 0)
        axs[2].imshow(image)
