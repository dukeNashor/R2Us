import numpy as np
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter
import torch

g_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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