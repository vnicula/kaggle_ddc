import cv2
import dlib
import math
import multiprocessing
import numpy as np
import os
from PIL import Image as pil_image
import pretrainedmodels
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model


def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load('/work/FaceForensics/models/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
               True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def parse_vid(video_path, skip_n):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('cv2.CAP_PROP_FRAME_COUNT: {}'.format(frame_num))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    count = 0
    while True:
        # success, image = vidcap.read()
        success = vidcap.grab()
        if success:
            if count % (skip_n+1) == 0:
                success, image = vidcap.retrieve()
                imgs.append(image)
            count += 1
        else:
            break

    vidcap.release()
    frame_num = len(imgs)
    return imgs, frame_num, fps, width, height


def save_faces(batch, path):
    if len(batch):
        im_ary = []
        for im in batch:
            rim = cv2.resize(im, (256, 256))
            im_ary.append(rim)
        im_concat = np.concatenate(im_ary, 1)
        cv2.imwrite(path, im_concat)


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def get_rois_dlib(images, img_scale=0.75):
    if not hasattr(get_rois_dlib, "counter"):
        get_rois_dlib.counter = 0
    rois = []
    t0 = time.time()

    for im in images:
        if img_scale > 0.99:
            imr = im
        else:
            imr = cv2.resize(im, (int(im.shape[1] * img_scale), int(im.shape[0] * img_scale)))
        height, width = imr.shape[:2]
        imr = cv2.cvtColor(imr, cv2.COLOR_BGR2GRAY)
        # eq hist needs grayscale or one channel
        imr = cv2.equalizeHist(imr)
        imr = np.uint8(imr)
        # print('resized im for detection shape {}'.format(imr.shape))
        # im_test.counter += 1
        # cv2.imwrite('resized_%d.jpg' % im_test.counter, imr)

        faces = face_detector(imr, 0)
        if len(faces):
            #TODO check if multiple faces are actually used
            for face in faces:            
                # only for cnn face detector, need to get rect from mmod_rectangles
                # see http://quabr.com/56322815/how-to-convert-mmod-rectangles-to-rectangles-via-dlib
                if type(face) == dlib.mmod_rectangle:
                    face = face.rect
                x, y, size = get_boundingbox(face, width, height)
                x = int(x / img_scale)
                y = int(y / img_scale)
                size = int(size / img_scale)
                cropped_face = im[y:y+size, x:x+size]
                rois.append(cropped_face)

    t1 = time.time()
    print("rois size: {}, took {}".format(len(rois), t1-t0))
    # save_faces(rois[-10:], 'rois_%d.jpg' % get_rois_dlib.counter)
    # get_rois_dlib.counter += 1

    return rois


def inference(face_list, model):
    face_preds = []
    for face in face_list:
        prediction, output = predict_with_model(face, model, cuda=True)
        face_preds.append(output.detach().cpu().numpy()[0][1])
    
    return face_preds


def run(input_dir, pool, model):

    f_list = os.listdir(input_dir)
    prob_list = []
    for f_name in f_list:
        # Parse video
        f_path = os.path.join(input_dir, f_name)
        print('Testing: ' + f_path)
        suffix = f_path.split('.')[-1]
        if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'nef', 'raf']:
            im = cv2.imread(f_path)
            if im is not None:
                _ = get_rois_dlib([im])

        elif suffix.lower() in ['mp4', 'avi', 'mov']:
            # Parse video
            imgs, frame_num, fps, width, height = parse_vid(f_path, skip_n=4)
            if frame_num == 0:
                continue
            print('imgs {}, frame_num {}, fps {}, width {}, height {}'.format(len(imgs), frame_num, fps, width, height))
            probs = []
            bsize = (1 + len(imgs)) // 2
            batch_imgs = [imgs[i * bsize:(i + 1) * bsize] for i in range((len(imgs) + bsize - 1) // bsize)] 
            print('pool batch len {}, first batch len is {}'.format(len(batch_imgs), len(batch_imgs[0])))

            # for i in range(0, len(imgs), bsize):
            #     batch = imgs[i:i+bsize]
            #     prob = im_test(batch, pool)
            #     probs.extend(prob)
            if pool is not None:
                rois = pool.map(get_rois_dlib, batch_imgs)
            else:
                print('No process pool for face detection.')
                rois = []
                for batch in batch_imgs:
                    rois.append(get_rois_dlib(batch))
            
            rois = [item for sublist in rois for item in sublist]
            probs = inference(rois, model)
            print('probs: ', probs)
            # Remove opt out frames
            if len(probs) == 0:
                prob = 0.5
            else:
                prob_rois = zip(probs, rois)
                prob_rois = zip(*sorted(prob_rois, key=lambda z:z[0]))
                sprob, srois = (list(t) for t in prob_rois)

                print('highest prob per im_test: ', sprob[-1:])
                save_faces(srois[-10:], '{}.jpg'.format(os.path.basename(f_path)))

                # prob = np.mean(sprob[-int(frame_num / 3):])
                # prob = np.mean(sprob[-int(frame_num / 2):])
                prob = np.mean(sprob[-int(frame_num / 3):-1])
                # prob = np.median(sprob[-int(frame_num / 2):-int(frame_num / 4)])
                prob = max(0.1, min(0.95, prob))

        #TODO check why metadata.json gets appended with 0.14 probability
        prob_list.append([os.path.basename(f_path), prob])
        print('Prob: ' + str(prob))

    return prob_list

face_detector = dlib.get_frontal_face_detector()
    
if __name__ == '__main__':
    t0 = time.time()

    # Load model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    model_path = '/work/FaceForensics/models/full/xception/full_raw.p'
    print('Load model from {}'.format(model_path))
    model = torch.load(model_path)
    model = model.cuda()

    # context = multiprocessing
    # if "forkserver" in multiprocessing.get_all_start_methods():
    #     context = multiprocessing.get_context("forkserver")
    # pool = context.Pool(processes=2)
    pool = multiprocessing.Pool(processes=2)
    # run('D:/work/CVPRW2019_Face_Artifacts_mod/test', pool, model)
    # prob_list = run('D:/work/MesoNet/test_videos', pool, model)
    # prob_list = run('easy_videos', None, model)
    # prob_list = run('D:\work\CVPRW2019_Face_Artifacts_mod\detection_challenge', pool, model)
    prob_list = run('C:/Downloads/deepfake-detection-challenge/train_sample_videos', pool, model)

    with open('sample_submission.csv', 'w') as sf:
        sf.write('filename,label\n')
        for name, score in prob_list:
            sf.write('%s,%1.6f\n' % (name, score))

    t1 = time.time()
    print("Execution took: {}".format(t1-t0))
