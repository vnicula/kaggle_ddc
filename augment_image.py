import constants
import tensorflow as tf

def random_jitter(image):

    image = tf.image.resize(image, [272, 272]) # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(
        image, size=[constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    return image


def random_rotate(image):
    # if image.shape.__len__() == 4:
    #     random_angles = tf.random.uniform(shape = (tf.shape(image)[0], ), minval = -np.pi / 4, maxval = np.pi / 4)
    # if image.shape.__len__() == 3:
    #     random_angles = tf.random.uniform(shape = (), minval = -np.pi / 4, maxval = np.pi / 4)

    # # BUG in Tfa ABI undefined symbol: _ZNK10tensorflow15shape_inference16InferenceContext11DebugStringEv
    # return tfa.image.rotate(image, random_angles)

    # NOTE this needs numpy
    image_array = tf.keras.preprocessing.image.random_rotation(
        image.numpy(), 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0.0,
        interpolation_order=1
    )
    # image = tf.convert_to_tensor(image_array)
    return image


@tf.function
def image_augment(x: (tf.Tensor, tf.Tensor), y: tf.Tensor) -> ((tf.Tensor, tf.Tensor), tf.Tensor):
    """augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # TODO investigate these two
    # x = tf.image.random_hue(x, 0.08)
    # x = tf.image.random_saturation(x, 0.6, 1.6)
    
    x0 = tf.image.random_brightness(x[0], 0.1)
    x0 = tf.image.random_contrast(x0, 0.8, 1.2)
    x0 = tf.image.random_flip_left_right(x0)

    x1 = tf.image.random_brightness(x[1], 0.1)
    x1 = tf.image.random_contrast(x1, 0.8, 1.2)
    x1 = tf.image.random_flip_left_right(x1)

    jitter_choice0 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x0 = tf.cond(jitter_choice0 < 0.75, lambda: x0, lambda: random_jitter(x0))

    jitter_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x1 = tf.cond(jitter_choice1 < 0.75, lambda: x1, lambda: random_jitter(x1))

    rotate_choice0 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x0 = tf.cond(rotate_choice0 < 0.75, lambda: x0, lambda: tf.py_function(random_rotate, [x0], tf.float32))
    x0 = tf.reshape(x0, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])
    
    rotate_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x1 = tf.cond(rotate_choice1 < 0.75, lambda: x1, lambda: tf.py_function(random_rotate, [x1], tf.float32))
    x1 = tf.reshape(x1, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    jpeg_choice0 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x0 = tf.cond(jpeg_choice0 < 0.75, lambda: x0, lambda: tf.image.random_jpeg_quality(
        x0, min_jpeg_quality=40, max_jpeg_quality=90))

    jpeg_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x1 = tf.cond(jpeg_choice1 < 0.75, lambda: x1, lambda: tf.image.random_jpeg_quality(
        x1, min_jpeg_quality=40, max_jpeg_quality=90))

    return ((x0, x1), y)

