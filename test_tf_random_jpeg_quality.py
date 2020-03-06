import matplotlib.pyplot as plt
import sys
import tensorflow as tf


def visualize(original, augmented, name):
    fig = plt.figure()
    original_plt=fig.add_subplot(1,2,1)
    original_plt.set_title('original image')
    original_plt.imshow(original)

    augmented_plt=fig.add_subplot(1,2,2) 
    augmented_plt.set_title('augmented image')
    augmented_plt.imshow(augmented)
    plt.savefig(name, block=True)


if __name__ == '__main__':

    image_path = sys.argv[1]
    image_string=tf.io.read_file(image_path) 
    image=tf.image.decode_png(image_string, channels=3)

    for i in range(10, 100, 10):
        image_aug = tf.image.random_jpeg_quality(image, min_jpeg_quality=i, max_jpeg_quality=i+10)
        visualize(image.numpy(), image_aug.numpy(), 'augmented_%d.png' % i)
