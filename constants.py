META_DATA = "metadata.json"
MARGIN = 16
MAX_DETECTION_SIZE = 960
TRAIN_FACE_SIZE = 224
TRAIN_FRAME_COUNT = 32
TRAIN_FPS = 3
# SKIP_INITIAL_SEC = 8
SKIP_INITIAL_SEC = 0
MIN_TRACK_FACES = 5

SEQ_LEN = 30
FEAT_SHAPE = (224, 224, 3)

MESO_INPUT_HEIGHT = 256
MESO_INPUT_WIDTH = 256

def save_predictions(predictions):
    with open('sample_submission.csv', 'w') as sf:
        sf.write('filename,label\n')
        for name, score in predictions:
            sf.write('%s,%1.6f\n' % (name, score))
