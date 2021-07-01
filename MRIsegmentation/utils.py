import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_io as tfio
from tensorflow.python.ops import io_ops
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import MeanIoU
import matplotlib as plt
from skimage import io


def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def F1_score(y_true, y_pred):
    """input and output are np array"""
    metric = F1Score(num_classes=1, threshold=0.5)
    metric.update_state(y_true, y_pred)
    result = metric.result()
    f1 = result.numpy()
    return f1


def IoU_score(y_true, y_pred):
    m = MeanIoU(num_classes=2)
    m.reset_state()
    m.update_state(y_true, y_pred)
    IoU = round(m.result().numpy(), 2)
    return IoU


def load_scan_and_mask(x, y):
    return (tfio.experimental.image.decode_tiff(io_ops.read_file(x)),
            tfio.experimental.image.decode_tiff(io_ops.read_file(y)))

def dataviz_image_and_mask(tf_dataset, number_of_samples):
  for image_path, mask_path in tf_dataset.take(number_of_samples):
    fig,axs = plt.subplots(1,3, figsize=(8,15))
    
    image = io.imread(image_path.numpy().decode('ascii'))
    mask = io.imread(mask_path.numpy().decode('ascii'))
    axs[0].set_title('Image')
    axs[0].imshow(image)

    axs[1].set_title('Mask')
    axs[1].imshow(mask, cmap='gray')
    
    axs[2].set_title('MRI with Mask')
    image[mask==255] = (255, 0, 0)
    axs[2].imshow(image)