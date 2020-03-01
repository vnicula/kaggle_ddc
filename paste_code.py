def save_sample_img(name, label, values):
    IMG_SIZE = values[0].shape[0]    

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 4
    font_scale = 2

    # line_shape = (IMG_SIZE, max_elems*IMG_SIZE, 3)
    tile_shape = (IMG_SIZE, constants.SEQ_LEN*IMG_SIZE, 3)
    tile_img = np.zeros(tile_shape, dtype=np.float32)
    for j in range(len(values)):
        color = (0, 255, 0) if label == 0 else (255, 0, 0)
        cv2.putText(tile_img, name, (10, 50),
                        font_face, font_scale,
                        color, thickness, 2)
        
        tile_img[:, j*IMG_SIZE:(j+1)*IMG_SIZE, :] = values[j]

    plt.imsave(name+'.jpg', tile_img)

    return tile_img


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def get_simple_feature(face):
    t0 = time.time()

    N = 300
    img = np.dot(face[...,:3], [0.2989, 0.5870, 0.1140])    
    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(magnitude_spectrum)

    # Interpolation
    points = np.linspace(0,N,num=psd1D.size) 
    xi = np.linspace(0,N,num=N) 
    interpolated = griddata(points,psd1D,xi,method='cubic')
    
    # Normalization
    interpolated /= interpolated[0]
    # print('simple feature shape: {}, took {}'.format(interpolated.shape, time.time()-t0))

    return interpolated             


def get_image_feature(face):
    # I think this is what preprocess input does with 'tf' mode
    return (face.astype(np.float32) / 127.5) - 1.0


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size

    # Get the number to skip between iterations.
    skip = len(input_list) // size

    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    # Cut off the last one if needed.
    return output[:size]


# TODO oversample REAL
def read_file(file_path):
    
    t0 = time.time()
    # names = []
    labels = []
    samples = []
    masks = []
    
    with open(file_path.numpy(), 'rb') as f_p:
        data = pickle.load(f_p)
        selected_keys = [k for k in data.keys() if data[k][0] == 1]
        initial_positives = len(selected_keys)
        if len(selected_keys) > 2:
            random.shuffle(selected_keys)
            selected_keys = selected_keys[:int(len(selected_keys) * 0.7)]
        print('Loaded {}, dropped {} positives.'.format(file_path, initial_positives-len(selected_keys)))
        selected_set = set(selected_keys)
        feature_func = get_image_feature
        # feature_func = get_simple_feature

        for key in data.keys():
            label = data[key][0]
            if label == 1 and key not in selected_set:
                continue
            # names.append(key)
            # sample = data[key][1][0]

            # feat_shape = feature_func(data[key][1][0]).shape
            feat_shape = FEAT_SHAPE
            my_seq_len = len(data[key][1])
            data_seq_len = min(my_seq_len, SEQ_LEN)
            sample = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            sample_f = np.zeros((SEQ_LEN,) + feat_shape, dtype=np.float32)
            mask = np.zeros(SEQ_LEN, dtype=np.float32)
            for indx in range(data_seq_len):
                sample[indx] = feature_func(data[key][1][indx])
                if label == 0:
                    sample_f[indx] = feature_func(np.fliplr(data[key][1][indx]))
                mask[indx] = 1.0
            
            # sample = preprocess_input(sample)
            # print(file_path, len(samples))
            samples.append(sample)
            masks.append(mask)
            labels.append(label)

            if label == 0:
                samples.append(sample_f)
                masks.append(mask)
                labels.append(0)
                # save_sample_img(key+'_o', 0, sample)
                # save_sample_img(key+'_f', 0, sample_f)
        
        del data
    # NOTE if one sample doesn't have enough frames Keras will error out here with 'assign a sequence'
    npsamples = np.array(samples, dtype=np.float32)
    npmasks = np.array(masks, dtype=np.float32)
    nplabels = np.array(labels, dtype=np.int32)

    print('file {} Shape samples {}, labels {} took {}'.format(file_path, npsamples.shape, nplabels.shape, time.time()-t0))
    # return tf.data.Dataset.from_tensor_slices((npsamples, npmasks, nplabels))
    return npsamples, npmasks, nplabels

"""
npsamples = np.zeros((64,) + (SEQ_LEN,) + FEAT_SHAPE, dtype=np.float32)
npmasks = np.ones((64,) + (SEQ_LEN,), dtype=np.float32)
nplabels = np.ones((64,), dtype = np.int32)
def fake_read_file(file_path):
    t0 = time.time()

    print('Fake read {}, thread {}, took {}'.format(file_path, threading.get_ident(), time.time()-t0))
    return npsamples, npmasks, nplabels
"""

# def input_dataset(input_dir):
#     print('Using dataset from: ', input_dir)
#     dataset = tf.data.Dataset.list_files(input_dir)
#     # f_list = os.listdir(input_dir)
#     # dataset_files = [os.path.join(input_dir, fn) for fn in f_list if fn.endswith('pkl')]
#     # dataset = tf.data.Dataset.from_tensor_slices((dataset_files))
    
#     dataset = dataset.flat_map(
#         lambda file_name: tf.data.Dataset.from_tensor_slices(
#             tuple(tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.float32]))
#         )
#     )
#     def final_map(s, m, l):
#         return  {'input_1': tf.reshape(s, [-1, 224, 224, 3]), 'input_2': tf.reshape(m, [-1, 1])}, tf.reshape(l, [-1])
#     dataset = dataset.map(final_map)
#     return dataset

# TODO wip finish the optimal one
# def input_dataset(input_dir, is_training):
#     print('Using dataset from: ', input_dir)

#     dataset = tf.data.Dataset.list_files(input_dir).shuffle(1024)
#     # dataset = tf.data.Dataset.range(1, 2000)
#     def map_function_wrapper(filename):
#         features, masks, labels = tf.py_function(
#            read_file, [filename], (tf.float32, tf.float32, tf.int32))
        
#         return tf.data.Dataset.from_tensor_slices((features, masks, labels))
#         # return tf.data.Dataset.from_tensor_slices((npsamples, npmasks, nplabels))
    
#     dataset = dataset.map(
#         map_function_wrapper,
#         num_parallel_calls=8
#     ).prefetch(4)
#     dataset = dataset.interleave(
#         # lambda *x: tf.data.Dataset.from_tensor_slices(x).map(
#         lambda x: x.map(
#             lambda s, m, l: ({'input_1': tf.reshape(s, (-1,)+FEAT_SHAPE), 'input_2': tf.reshape(m, [-1])}, tf.reshape(l, [-1]))
#         ),
#         cycle_length=16,
#         num_parallel_calls=16
#     )

#     return dataset


def input_dataset(input_dir, is_training):
    print('Using dataset from: ', input_dir)

    dataset = tf.data.Dataset.list_files(input_dir).shuffle(1024)
    # options = tf.data.Options()
    # options.experimental_deterministic = False
    # dataset = dataset.with_options(options)
    # dataset = dataset.interleave(
    #     # map_func=lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     map_func=lambda file_name: tf.data.Dataset.from_tensor_slices(
    #         tuple(tf.py_function(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))
    #     ),
    #     cycle_length=16, #tf.data.experimental.AUTOTUNE
    #     block_length=1,
    #     num_parallel_calls=16
    # )

    # dataset = dataset.map(
    #     lambda file_name: tf.py_func(read_file, [file_name], [tf.float32, tf.float32, tf.int32]),
    #     num_parallel_calls=32
    # ).prefetch(64) #.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.flat_map(
    #     lambda *x: tf.data.Dataset.from_tensor_slices(x)
    # )

    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        lambda file_name: tf.data.Dataset.from_tensor_slices(
            tuple(tf.py_function(read_file, [file_name], [tf.float32, tf.float32, tf.int32]))),
        cycle_length=8,
        block_length=1,
        sloppy=True,
        buffer_output_elements=4,
        )
    )

    def final_map(s, m, l):
        return  {'input_1': tf.reshape(s, (-1,)+FEAT_SHAPE), 'input_2': tf.reshape(m, [-1])}, tf.reshape(l, [-1])
    dataset = dataset.map(final_map, num_parallel_calls=16)

    # def class_func(sample, mask, label):
    #     return label

    # if is_training:
    #     resampler = tf.data.experimental.rejection_resample(class_func, target_dist=[0.3, 0.7])
    #     dataset = dataset.apply(resampler)

    return dataset


def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded

