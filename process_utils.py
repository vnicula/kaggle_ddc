import cv2
import constants
import iou_tracker
import math
import numpy as np

from PIL import Image

def parse_vid(video_path, max_detection_size, max_frame_count, sample_fps, skip_inital_sec):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    print('{}: cv2.FRAME_COUNT {}, cv2.PROP_FPS {}, cv2.FRAME_WIDTH {}, cv2.FRAME_HEIGHT {}'.format(
        video_path, frame_num, fps, width, height))
    
    skip_n = max(math.floor(fps / sample_fps), 0)
    max_dimension = max(width, height)
    img_scale = 1.0
    if max_dimension > max_detection_size:
        img_scale = max_detection_size / max_dimension
    print('Skipping %1.1f frames, scaling: %1.4f' % (skip_n, img_scale))

    imrs = []
    imgs = []
    count = 0

    #TODO make this robust to video reading errors
    for i in range(frame_num):
        success = vidcap.grab()
            
        if success:
            if i < skip_inital_sec * fps:
                continue
            if i % (skip_n+1) == 0:
                success, im = vidcap.retrieve()
                if success:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    if img_scale < 1.0:
                        imr = cv2.resize(im, (int(im.shape[1] * img_scale), int(im.shape[0] * img_scale)))
                    else:
                        imr = im
                    imgs.append(im)
                    imrs.append(imr)
                    count += 1
                    if count >= max_frame_count:
                        break
        else:
            break

    vidcap.release()
    return imgs, imrs, img_scale


def get_faces_from_tracks(originals, tracks, img_scale):
    """returns not resized faces"""
    faces = []
    for track in tracks:
        track_faces = []
        for i, bbox in enumerate(track['bboxes']):
            frame_indx = track['start_frame'] + i - 1
            if frame_indx < len(originals):
                original = originals[frame_indx]
                (x,y,w,h) = (
                    max(int(bbox[0] / img_scale) - constants.MARGIN, 0),
                    max(int(bbox[1] / img_scale) - constants.MARGIN, 0),
                    int((bbox[2]-bbox[0]) / img_scale) + 2*constants.MARGIN,
                    int((bbox[3]-bbox[1]) / img_scale) + 2*constants.MARGIN
                )
                face_extract = original[y:y+h, x:x+w].copy() # Without copy() memory leak with GPU
                track_faces.append(face_extract)
        faces.append(track_faces)

    return faces


def detect_faces_bbox(detector, label, originals, images, batch_size, img_scale, face_size, keep_tracks):
    faces = []
    detections = []

    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        if (frames_boxes is not None) and (len(frames_boxes) > 0):
            # print(frames_boxes, frames_confidences)
            for i in range(len(frames_boxes)):
                if frames_boxes[i] is not None:
                    boxes = []
                    for box, confidence in zip(frames_boxes[i], frames_confidences[i]):
                        boxes.append({'bbox': box, 'score':confidence})
                    detections.append(boxes)
    
    tracks = iou_tracker.track_iou(detections, 0.8, 0.9, 0.1, constants.MIN_TRACK_FACES)

    # TODO remove this
    # Can't use anything since it's multitrack fake
    if label == 1 and len(tracks) > 1:
        return faces, tracks[:keep_tracks]

    tracks.sort(key = lambda x:x['max_score'], reverse=True)
    faces = get_faces_from_tracks(originals, tracks[:keep_tracks], img_scale)

    if face_size > 0:
        for track_face in faces:
            for face in track_face:
                face = cv2.resize(face, (face_size, face_size))
    # print(faces, tracks)
    return faces, tracks[:keep_tracks]


def detect_faces_no_tracks(detector, images, batch_size, face_size):
    faces = []

    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [Image.fromarray(image) for image in images[lb:lb+batch_size]]
        frames_boxes, frames_confidences = detector.detect(imgs_pil, landmarks=False)
        if (frames_boxes is not None) and (len(frames_boxes) > 0):
            # print(frames_boxes, frames_confidences)
            for i in range(len(frames_boxes)):
                if frames_boxes[i] is not None:
                    for bbox in frames_boxes[i]:
                        (x,y,w,h) = (
                            max(int(bbox[0]) - constants.MARGIN, 0),
                            max(int(bbox[1]) - constants.MARGIN, 0),
                            int(bbox[2]-bbox[0]) + 2*constants.MARGIN,
                            int(bbox[3]-bbox[1]) + 2*constants.MARGIN
                        )
                    image = images[i]
                    face_extract = image[y:y+h, x:x+w].copy() # Without copy() memory leak with GPU
                    if face_size > 0:
                        face_extract = cv2.resize(face_extract, (face_size, face_size))
                    faces.append(face_extract)
    
    return faces


def isotropically_resize_image(img, size, resample=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = (size - h) // 2
    b = (size - h) - t
    l = (size - w) // 2
    r = (size - w) - l
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def square_resize(img, input_size):
    resized_img = isotropically_resize_image(img, input_size)
    resized_img = make_square_image(resized_img)
    return resized_img