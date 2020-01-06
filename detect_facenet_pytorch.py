import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
# from tqdm.notebook import tqdm
import tqdm
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

sample = 'C:/Downloads/deepfake-detection-challenge/train_sample_videos/aagfhgtpmv.mp4'

reader = cv2.VideoCapture(sample)
images_1080_1920 = []
images_720_1280 = []
images_540_960 = []
# for i in tqdm.tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):
for i in tqdm.tqdm(range(100)):
    _, image = reader.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_1080_1920.append(image)
    images_720_1280.append(cv2.resize(image, (1280, 720)))
    images_540_960.append(cv2.resize(image, (960, 540)))
reader.release()

images_1080_1920 = np.stack(images_1080_1920)
images_720_1280 = np.stack(images_720_1280)
images_540_960 = np.stack(images_540_960)

print('Shapes:')
print(images_1080_1920.shape)
print(images_720_1280.shape)
print(images_540_960.shape)

def plot_faces(images, figsize=(10.8/2, 19.2/2)):
    shape = images[0].shape
    images = images[np.linspace(0, len(images)-1, 16).astype(int)]
    im_plot = []
    for i in range(0, 16, 4):
        im_plot.append(np.concatenate(images[i:i+4], axis=0))
    im_plot = np.concatenate(im_plot, axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(im_plot)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.grid(False)
    fig.tight_layout()
    plt.show()

def timer(detector, detect_fn, images, *args):
    start = time.time()
    faces = detect_fn(detector, images, *args)
    elapsed = time.time() - start
    print(f', {elapsed:.3f} seconds')
    return faces, elapsed

# plot_faces(images_540_960, figsize=(10.8, 19.2))

from facenet_pytorch import MTCNN
detector = MTCNN(device=device, post_process=False)
burnin = np.zeros_like(images_540_960)
detector(burnin)

def detect_facenet_pytorch(detector, images, batch_size):
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        faces.extend(detector(imgs_pil))
    return faces

times_facenet_pytorch = []    # batched
times_facenet_pytorch_nb = [] # non-batched

print('Detecting faces in 540x960 frames', end='')
# _, elapsed = timer(detector, detect_facenet_pytorch, images_540_960, 60)
faces_540_960, elapsed = timer(detector, detect_facenet_pytorch, images_540_960, 256)
times_facenet_pytorch.append(elapsed)

# print('Detecting faces in 720x1280 frames', end='')
# # _, elapsed = timer(detector, detect_facenet_pytorch, images_720_1280, 40)
# _, elapsed = timer(detector, detect_facenet_pytorch, images_720_1280, 80)
# times_facenet_pytorch.append(elapsed)

# print('Detecting faces in 1080x1920 frames', end='')
# # faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 20)
# faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 32)
# times_facenet_pytorch.append(elapsed)

faces_numpy = torch.stack(faces_540_960).permute(0, 2, 3, 1).int().numpy()
plot_faces(faces_numpy)
print(faces_numpy.shape)

# print('Detecting faces in 540x960 frames', end='')
# _, elapsed = timer(detector, detect_facenet_pytorch, images_540_960, 1)
# times_facenet_pytorch_nb.append(elapsed)

# print('Detecting faces in 720x1280 frames', end='')
# _, elapsed = timer(detector, detect_facenet_pytorch, images_720_1280, 1)
# times_facenet_pytorch_nb.append(elapsed)

# print('Detecting faces in 1080x1920 frames', end='')
# faces, elapsed = timer(detector, detect_facenet_pytorch, images_1080_1920, 1)
# times_facenet_pytorch_nb.append(elapsed)

del detector
torch.cuda.empty_cache()