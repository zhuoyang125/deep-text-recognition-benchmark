import string
import argparse
import time
import cv2

import os
import importlib
import numpy as np
import matplotlib

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image
from torchvision import transforms

def show_video(model, opt):

    cap = cv2.VideoCapture(opt.file_dir)

    cap2 = cv2.VideoCapture('ANW.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:" + str(fps))

    #sample = 0.5 # every <sample> sec take one frame                               # Use only if you do not want the infer every frame
    #sample_num = sample * fps

    if not cap.isOpened():
        print("Error in opening video stream or file")

    log = open(f'./log_video_demo_result.txt', 'a')
    dashed_line = '-' * 80
    head = f'{"Frames":25s}\t{"predicted_labels":25s}\tconfidence score'

    print(f'{dashed_line}\n{head}\n{dashed_line}')
    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

    length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
    text_for_pred = torch.LongTensor(opt.batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        ret2, frame2 = cap2.read()
        frame2 = cv2.resize(frame2, (0, 0), None, .6, .6)

        #print(frame.shape)
        cv2.rectangle(frame2, (558, 43), (612, 67), (0,0, 255), 2)
        cv2.imshow("Frame", frame)
        cv2.moveWindow("Frame", 270, 650)
        cv2.imshow("Frame2", frame2)
        cv2.moveWindow("Frame2", 70, 100)
        frame_count += 1
        if ret:
            #time.sleep(0.1)
            if opt.input_channel == 1:
                frame = Image.fromarray(frame).convert('RGB')
                frame = frame.convert('L')
                frame = transforms.ToTensor()(frame)
                frame = frame.unsqueeze(0)
                image = frame.to(device)
            else:
                frame = Image.fromarray(frame).convert('RGB')
                frame = transforms.ToTensor()(frame)
                frame = frame.unsqueeze(0)
                image = frame.to(device)
            
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                frame_log = 'Frame' + str(frame_count)

                print(f'{frame_log:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{frame_log:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        else:
            break

    log.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Demo")
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='.0123456789', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    parser.add_argument("--file", dest="file_dir", help="video file path", type=str)                    # Path to video for detection
                                               
    opt = parser.parse_args()

    print("Video File:" + str(opt.file_dir))

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    model.eval()

    show_video(model, opt)
