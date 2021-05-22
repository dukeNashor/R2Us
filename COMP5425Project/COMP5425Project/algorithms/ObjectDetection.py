import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

import os
os.environ['TORCH_HOME'] = 'Cache'
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                               'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
                              'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                             'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
                            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
                          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush']


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
    def Plot(img, boxes, pred_cls, cls_score):
        for i in range(len(boxes)): 
            cv2.rectangle(img = img,
                          pt1 = (round(boxes[i][0]), round(boxes[i][1])),
                          pt2 = (round(boxes[i][2]), round(boxes[i][3])),
                          color = (0, 255, 0),
                          thickness = 1) 
            cv2.putText(img = img,
                        text = COCO_INSTANCE_CATEGORY_NAMES[pred_cls[i]] + " {:.3f}".format(cls_score[i].item()),
                        org = (round(boxes[i][0]), round(boxes[i][1])),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.5,
                        color = (0, 255, 0),
                        thickness = 1) 
        plt.figure(figsize=(20,30)) 
        plt.imshow(img)
        plt.show()


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
    od.Plot(image, boxes_out, labels_out, scores_out)