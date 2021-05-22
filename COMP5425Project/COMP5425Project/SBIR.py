import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QVariant, QRect
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage

from utils import *
import pickle
import random

ROLE_UID = Qt.UserRole + 1
ROLE_SHOWTEXT = Qt.UserRole + 2
ROLE_SKETCH = Qt.UserRole + 3
ROLE_RECT = Qt.UserRole + 4
ROLE_IMAGE_PATH = Qt.UserRole + 5

SKETCH_WIDTH = 400
SKETCH_HEIGHT = 400

thumb_pos = QRect(0, 0, 200, 200)

class Canvas(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.drawing = False
        self.lastPoint = QPoint()
        self.image = QPixmap(SKETCH_WIDTH, SKETCH_HEIGHT)
        self.image.fill(Qt.white)
        self.setMinimumWidth(self.image.width())
        self.setMinimumHeight(self.image.height())

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()


    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()


    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False


    def ClearCanvas(self):
        self.image.fill(Qt.white)
        self.update()


    def GetImage(self):
        return QPixmap(self.image)

from torchvision import transforms
# subclassed to allow classifier to work
class ClassifierCanvas(Canvas):
    def __init__(self, classifier = None, idx_to_class = None, *args, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier
        self.idx_to_class = idx_to_class

        self.transform = transforms.Compose([ transforms.ToPILImage(),
                                              transforms.Resize(224),
                                              transforms.ToTensor()])
        # initialize to avoid slow first pass.
        self.classifier(self.transform(torch.rand(3, 224, 224)).unsqueeze(0).to(g_device))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            arr = QPixmapToNumpy(self.image, channels_count = 3)
            arr = self.transform(arr).to(g_device)
            pred = self.classifier(arr.unsqueeze(0))
            idx = pred.argmax(1).item()
            print(self.idx_to_class[idx])


class PaintWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.image = QPixmap(SKETCH_WIDTH, SKETCH_HEIGHT)
        self.image.fill(QColor(255, 255, 255))
        self.setMinimumWidth(self.image.width())
        self.setMinimumHeight(self.image.height())

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)


    def SetImage(self, image):
        if image is None:
            self.image = QPixmap(SKETCH_WIDTH, SKETCH_HEIGHT)
            self.image.fill(QColor(255, 255, 255))
        else:
            self.image = image

        self.update()

    def GetImage(self):
        return self.image.copy()

from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsRectItem
class MovingObject(QGraphicsRectItem):
    def __init__(self, x, y, w, h, image, parent = None):
        super().__init__(x, y, w, h, parent = parent)
        self.setRect(x, y, w, h)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.image = image
        self.draw_bound = False

    def SetDrawBoundary(self, draw_bound):
        self.draw_bound = draw_bound
        self.update()

    def hoverEnterEvent(self, event):
        QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.instance().restoreOverrideCursor()

    def paint(self, painter, option, widget):
        painter.setPen(QPen(Qt.cyan, 3))
        painter.drawPixmap(self.rect(), self.image.scaled(self.rect().width(), self.rect().height()), self.rect())
        if self.draw_bound:
            painter.drawRect(self.rect())



class ObjectView(QGraphicsView):
    def __init__(self, item_model):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.setSceneRect(0, 0, 620, 460)

        self.item_model = item_model
        
        #self.scene.changed.connect(lambda _: self.SceneChanged())

        self.item_dict = {}

    
    def UpdateObjects(self):
        visited = dict.fromkeys(self.item_dict.keys(), False)

        for i in range(self.item_model.rowCount()):
            idx = self.item_model.index(i, 0)
            uid = self.item_model.data(idx, role = ROLE_UID)

            if uid in self.item_dict:
                visited[uid] = True
                continue
            
            rect = self.item_model.data(idx, role = ROLE_RECT)
            image = self.item_model.data(idx, role = ROLE_SKETCH)
            self.item_dict[uid] = draggy = MovingObject(rect.x(), rect.y(), rect.width(), rect.height(), image = image)
            self.scene.addItem(draggy)

        # delete unvisited
        for uid, v in visited.items():
            if not v:
                self.scene.removeItem(self.item_dict[uid])
                self.item_dict.pop(uid)

    
    def SetCurrentSelected(self, current_item):
        uid = current_item.data(ROLE_UID)
        for u, item in self.item_dict.items():
            if u == uid:
                self.item_dict[u].SetDrawBoundary(True)
            else:
                self.item_dict[u].SetDrawBoundary(False)
        self.update()


    def SceneChanged(self):
        print("scene changed!")



from algorithms.SketchClassification import SketchClassifier, SketchTransferClassifier, GetTrainedClassifier

class UIManager():

    def __init__(self, ui):
        # classifier
        self.classifier, self.idx_to_class = GetTrainedClassifier()

        self.object_id = 0

        self.ui = ui
        self.ui.canvas = ClassifierCanvas(parent = ui.groupBox, classifier = self.classifier, idx_to_class = self.idx_to_class)
        ui.groupBox.layout().insertWidget(0, self.ui.canvas)

        self.ui.frame_selected = PaintWidget(parent = ui.groupBox_2)
        ui.groupBox_2.layout().insertWidget(0, self.ui.frame_selected)

        self.ui.frame_picked = PaintWidget(parent = ui.groupBox_3)
        ui.verticalLayout_11.insertWidget(0, self.ui.frame_picked)
        self.ui.frame_picked.setMinimumWidth(400)
        self.ui.frame_picked.setMinimumHeight(400)

        self.ui.object_view = ObjectView(self.ui.object_list_view.model())
        self.ui.gridLayout_4.addWidget(self.ui.object_view, 2, 1)
        self.ui.object_list_view.model().rowsInserted.connect(lambda _: self.ui.object_view.UpdateObjects())
        self.ui.object_list_view.model().rowsRemoved.connect(lambda _: self.ui.object_view.UpdateObjects())

        self.ui.btn_clear_canvas.clicked.connect(lambda _: self.ui.canvas.ClearCanvas())
        self.ui.btn_delete_object.clicked.connect(lambda _: self.DeleteObjectListItem())
        self.ui.btn_add_object.clicked.connect(lambda _: self.InsertDrawingToObjectList())

        self.ui.actionLoad_Data.triggered.connect(lambda _: self.LoadData())

        self.ui.object_list_view.currentItemChanged.connect(lambda _: self.ObjectListViewSelectionChanged())
        
        self.ui.actionLoad_sketches.triggered.connect(lambda _: self.LoadSketch())
        self.ui.picked_list_view.itemClicked.connect(lambda _: self.PickedListViewSelectionChanged())
        self.ui.btn_add_picked.clicked.connect(lambda _:self.InsertPickedToObjectList())

    def LoadData(self):
        fname = QFileDialog.getOpenFileName(None, 'Open file', './data',"Pickle files (*.pkl)")
        self.LoadProcessedDataImpl(fname[0])


    def LoadProcessedDataImpl(self, preprocessed_data_path):
        if not os.path.exists(preprocessed_data_path):
            print("Data not found: {}".format(preprocessed_data_path))
            return

        import pickle
        with open(preprocessed_data_path, "rb") as f:
            self.data_dict = pickle.load(f)

        print("Loaded {} images.".format(len(self.data_dict)))

    def LoadSketch(self):
        dir_name = QFileDialog.getExistingDirectory(None, 'Open Sketch root folder', './data')
        self.LoadSketchImpl(dir_name)

    def LoadSketchImpl(self, sketch_path):
        if not os.path.exists(sketch_path):
            print("Sketch Data not found: {}".format(sketch_path))
            return
        
        class_names = [f for f in os.listdir(sketch_path)]

        self.sketch_dict = {}

        plv = self.ui.picked_list_view

        for c in class_names:
            self.sketch_dict[c] = {}
            class_path = os.path.join(sketch_path, c)
            images = [imf for imf in os.listdir(class_path)]
            for imf in images:
                abs_path = os.path.join(sketch_path, c, imf)
                self.sketch_dict[c][abs_path] = c
                # populate picked list
            li = QtWidgets.QListWidgetItem(parent = None)
            li.setData(ROLE_SHOWTEXT, "{}".format(c))
            li.setText(li.data(ROLE_SHOWTEXT))
            li.setData(ROLE_IMAGE_PATH, self.sketch_dict[c])
            plv.addItem(li)

        return

    def PickedListViewSelectionChanged(self):
        plv = self.ui.picked_list_view
        current_selected = plv.currentItem()
        if current_selected is None:
            self.ui.frame_picked.SetImage(None)
            return

        if current_selected.data(ROLE_IMAGE_PATH) is None:
            return
        c = current_selected.text()
        # pick random image
        paths = [k for k in self.sketch_dict[c].keys()]
        random_image = QPixmap(random.choice(paths))
        self.ui.frame_picked.SetImage(random_image.scaled(self.ui.frame_picked.rect().width(), self.ui.frame_picked.rect().height()))


    def DeleteObjectListItem(self):
        olv = self.ui.object_list_view
        current_selected = olv.currentItem()
        if current_selected is None:
            return

        olv.takeItem(olv.row(current_selected))
        return


    def InsertDrawingToObjectList(self):
        olv = self.ui.object_list_view
        image = self.ui.canvas.GetImage()
        li = QtWidgets.QListWidgetItem(parent = None)
        li.setData(ROLE_UID, self.object_id)
        li.setData(ROLE_SHOWTEXT, "object #{}".format(self.object_id))
        li.setData(ROLE_SKETCH, image)
        li.setData(ROLE_RECT, thumb_pos)
        li.setText(li.data(ROLE_SHOWTEXT))
        olv.addItem(li)
        self.object_id = self.object_id + 1


    def InsertPickedToObjectList(self):
        plv = self.ui.object_list_view
        image = self.ui.frame_picked.GetImage()
        current_selected = self.ui.picked_list_view.currentItem()
        li = QtWidgets.QListWidgetItem(parent = None)
        li.setData(ROLE_UID, self.object_id)
        li.setData(ROLE_SHOWTEXT, "{}".format(current_selected.text()))
        li.setData(ROLE_SKETCH, image)
        li.setData(ROLE_RECT, thumb_pos)
        li.setText(li.data(ROLE_SHOWTEXT))
        plv.addItem(li)
        self.object_id = self.object_id + 1


    def ObjectListViewSelectionChanged(self):
        olv = self.ui.object_list_view
        current_selected = olv.currentItem()
        if current_selected is None:
            self.ui.frame_selected.SetImage(None)
            return

        if current_selected.data(ROLE_SKETCH) is None:
            return

        image = current_selected.data(ROLE_SKETCH)
        self.ui.frame_selected.SetImage(image)
        self.ui.object_view.SetCurrentSelected(current_selected)
