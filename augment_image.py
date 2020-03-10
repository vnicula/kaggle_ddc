import constants
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def random_jitter(image):

    image = tf.image.resize(image, [272, 272]) # method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if image.shape.__len__() == 4:
        batch_size = tf.shape(image)[0]
        image = tf.image.random_crop(
            image, size=[batch_size, constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])
    elif image.shape.__len__() == 3:
        image = tf.image.random_crop(
            image, size=[constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    return image


def random_rotate(image):
    if image.shape.__len__() == 4:
        random_angles = tf.random.uniform(shape = (tf.shape(image)[0], ), minval = -np.pi / 8, maxval = np.pi / 8)
    if image.shape.__len__() == 3:
        random_angles = tf.random.uniform(shape = (), minval = -np.pi / 8, maxval = np.pi / 8)

    # BUG in Tfa ABI undefined symbol: _ZNK10tensorflow15shape_inference16InferenceContext11DebugStringEv
    # Needs tf pip install not conda
    return tfa.image.rotate(image, random_angles)


def random_rotate_py(image):
    # NOTE this needs numpy
    image_array = tf.keras.preprocessing.image.random_rotation(
        image.numpy(), 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0.0,
        interpolation_order=1
    )
    # image = tf.convert_to_tensor(image_array)
    return image_array


@tf.function
def image_augment(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # TODO investigate these two
    # x = tf.image.random_hue(x, 0.08)
    # x = tf.image.random_saturation(x, 0.6, 1.6)
    
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_flip_left_right(x)

    jitter_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(jitter_choice < 0.75, lambda: x, lambda: random_jitter(x))

    rotate_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(rotate_choice < 0.75, lambda: x, lambda: random_rotate(x))
    # x = tf.reshape(x, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    jpeg_choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x = tf.cond(jpeg_choice < 0.75, lambda: x, lambda: tf.image.random_jpeg_quality(
        x, min_jpeg_quality=40, max_jpeg_quality=90))

    return (x, y)


@tf.function
def image_augment2(x: (tf.Tensor, tf.Tensor), y: tf.Tensor) -> ((tf.Tensor, tf.Tensor), tf.Tensor):
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
    # x0 = tf.cond(rotate_choice0 < 0.75, lambda: x0, lambda: tf.py_function(random_rotate, [x0], tf.float32))
    x0 = tf.cond(rotate_choice0 < 0.75, lambda: x0, lambda: random_rotate(x0))
    # x0 = tf.reshape(x0, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])
    
    rotate_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    # x1 = tf.cond(rotate_choice1 < 0.75, lambda: x1, lambda: tf.py_function(random_rotate, [x1], tf.float32))
    x1 = tf.cond(rotate_choice1 < 0.75, lambda: x1, lambda: random_rotate(x1))
    # x1 = tf.reshape(x1, [constants.MESO_INPUT_HEIGHT, constants.MESO_INPUT_WIDTH, 3])

    jpeg_choice0 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x0 = tf.cond(jpeg_choice0 < 0.75, lambda: x0, lambda: tf.image.random_jpeg_quality(
        x0, min_jpeg_quality=40, max_jpeg_quality=90))

    jpeg_choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x1 = tf.cond(jpeg_choice1 < 0.75, lambda: x1, lambda: tf.image.random_jpeg_quality(
        x1, min_jpeg_quality=40, max_jpeg_quality=90))

    return ((x0, x1), y)


def preprocess_symbolic_input_vggface(x, data_format=None, version=1):
    """Preprocesses a tensor encoding a batch of images.
    Returns:
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if tf.keras.backend.ndim(x) == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
    
    mean_v1 = [93.5940, 104.7624, 129.1863]
    mean_v2 = [91.4953, 103.8827, 131.0912]
    mean_imgnet = [103.939, 116.779, 123.68]

    if version == 1:
        mean_tensor = tf.keras.backend.constant(-np.array(mean_v1))
    else:
        mean_tensor = tf.keras.backend.constant(-np.array(mean_v2))

    # Zero-center by mean pixel
    if tf.keras.backend.dtype(x) != tf.keras.backend.dtype(mean_tensor):
        x = tf.keras.backend.bias_add(
            x, tf.keras.backend.cast(mean_tensor, tf.keras.backend.dtype(x)), data_format=data_format)
    else:
        x = tf.keras.backend.bias_add(x, mean_tensor, data_format)

    return x
