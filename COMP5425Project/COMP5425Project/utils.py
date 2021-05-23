import numpy as np
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter
import torch
import cv2
from PIL import Image
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

g_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def QImageToNumpy(image):
    size = image.size()
    s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)
    arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
    return arr

def QPixmapToNumpy(image, channels_count = 4):
    if not isinstance(image, QImage):
        image = image.toImage()
    s = image.bits().asstring(image.width() * image.height() * channels_count)
    arr = np.fromstring(s, dtype=np.uint8).reshape((image.height(), image.width(), channels_count))
    return arr


def FillTransparent(image):
    white = QImage(image.width(), image.height(), PyQt5.QtGui.QImage.Format_RGB888)
    white.fill(Qt.white)
    painter = QPainter(white)
    painter.drawImage(0, 0, image)
    painter.end()
    grayscale = white.convertToFormat(PyQt5.QtGui.QImage.Format_Grayscale8)
    #grayscale.save("saved.png")
    
    return grayscale

def QPixmapToTensor(pixmap):
    arr = QPixmapToNumpy(pixmap)
    return torch.from_numpy(arr).permute(2, 0, 1).to(g_device)


def GetBoxImage(img, boxes, pred_cls, cls_score):
    if isinstance(img, np.ndarray):
        image = np.array(img, copy = True)
    elif isinstance(img, str):
        image = np.asarray(Image.open(img))


    for i in range(len(boxes)): 
        cv2.rectangle(img = image,
                        pt1 = (round(boxes[i][0]), round(boxes[i][1])),
                        pt2 = (round(boxes[i][2]), round(boxes[i][3])),
                        color = (0, 255, 0),
                        thickness = 1) 
        cv2.putText(img = image,
                    text = COCO_INSTANCE_CATEGORY_NAMES[pred_cls[i]] + " {:.3f}".format(cls_score[i].item()),
                    org = (round(boxes[i][0]), round(boxes[i][1])),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.5,
                    color = (0, 255, 0),
                    thickness = 1) 
    return image

def PlotBoxes(img, boxes, pred_cls, cls_score):
    image = GetBoxImage(img, boxes, pred_cls, cls_score)
    plt.figure(figsize=(20,30)) 
    plt.imshow(image)
    plt.show()