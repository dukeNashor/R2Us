import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

import os
os.environ['TORCH_HOME'] = 'Cache'
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt
from utils import *
from utils import COCO_INSTANCE_CATEGORY_NAMES



class ObjectDetector():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load a model pre-trained pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
       
        
    def Predict(self, image, score_thresh = 0.8):
        im = (self.transform(image.copy())).to(self.device)
        res = self.model([im])[0]
        boxes_out, labels_out, scores_out = self.FilterResults(res, score_thresh)
        return boxes_out, labels_out, scores_out


    @staticmethod
    def FilterResults(res, score_thresh):
        boxes = res["boxes"].cpu().detach().numpy()
        labels = res["labels"].cpu().detach().numpy()
        scores = res["scores"].cpu().detach().numpy()
        valid_indices = scores > score_thresh
        boxes_out = boxes[valid_indices,:]
        labels_out = labels[valid_indices]
        scores_out = scores[valid_indices]
        return boxes_out, labels_out, scores_out



if __name__ == "__main__":
    od = ObjectDetector()
    image = np.asarray(Image.open("./data/val2017/000000001268.jpg"))
    boxes_out, labels_out, scores_out = od.Predict(image)
    utils.PlotBoxes(image, boxes_out, labels_out, scores_out)