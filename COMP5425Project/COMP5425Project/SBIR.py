import sys
import os
import pickle
import random

from utils import COCO_INSTANCE_CATEGORY_NAMES, GetBoxImage, QImageToNumpy
from PIL import Image
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QVariant, QRect
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage

ROLE_UID = Qt.UserRole + 1
ROLE_SHOWTEXT = Qt.UserRole + 2
ROLE_IMAGE = Qt.UserRole + 3
ROLE_RECT = Qt.UserRole + 4
ROLE_IMAGE_PATH = Qt.UserRole + 5

SKETCH_WIDTH = 400
SKETCH_HEIGHT = 400

thumb_pos = QRect(0, 0, 200, 200)

QUERY_RECT = (0, 0, 600, 450)

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

        self.label = "None"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)
        painter.drawText(300, 300, self.label)


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()


    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
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

    def GetLabel(self):
        return self.label

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
        #self.classifier(self.transform(torch.rand(3, 224, 224)).unsqueeze(0).to(g_device))
        self.classifier.Predict(np.zeros((1, 64, 64, 1)))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            qimage = self.image.toImage()
            qimage.invertPixels()
            qimage = qimage.scaled(64, 64).convertToFormat(QImage.Format_Grayscale8)
            #qimage.save("invert.png")
            arr = QImageToNumpy(qimage)
            pred = self.classifier.Predict(np.expand_dims(arr, axis = 0))

            idx = pred.argmax(1).item()
            label = self.idx_to_class[idx]
            print(self.idx_to_class[idx])
            self.label = label


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
        elif isinstance(image, np.ndarray):
            height, width, channel = image.shape
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            self.image = QPixmap.fromImage(qimage)
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

    def GetRect(self):
        return self.rect()


class ObjectView(QGraphicsView):
    def __init__(self, item_model):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.setSceneRect(*QUERY_RECT)

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
            image = self.item_model.data(idx, role = ROLE_IMAGE)
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
from algorithms.SketchClassificationV2_MobileNet import MobileNetSketchClassifier, GetTrainedClassifier
from algorithms.Retriever import Retriever

class UIManager():

    def __init__(self, ui):
        # classifier
        self.classifier, self.idx_to_class, self.class_to_idx = GetTrainedClassifier()

        self.object_id = 0

        self.ui = ui
        self.ui.canvas = ClassifierCanvas(parent = ui.groupBox, classifier = self.classifier, idx_to_class = self.idx_to_class)
        ui.groupBox.layout().insertWidget(0, self.ui.canvas)
        ui.groupBox.setStyleSheet("background-color:grey;")


        self.ui.frame_selected = PaintWidget(parent = ui.groupBox_2)
        ui.groupBox_2.layout().insertWidget(0, self.ui.frame_selected)

        self.ui.frame_picked = PaintWidget(parent = ui.groupBox_3)
        ui.verticalLayout_11.insertWidget(0, self.ui.frame_picked)
        self.ui.frame_picked.setMinimumWidth(400)
        self.ui.frame_picked.setMinimumHeight(400)

        self.ui.object_view = ObjectView(self.ui.object_list_view.model())
        self.ui.gridLayout_4.addWidget(self.ui.object_view, 2, 2)
        self.ui.object_list_view.model().rowsInserted.connect(lambda _: self.ui.object_view.UpdateObjects())
        self.ui.object_list_view.model().rowsRemoved.connect(lambda _: self.ui.object_view.UpdateObjects())

        self.ui.frame_result = PaintWidget(parent = self.ui.centralwidget)
        self.ui.gridLayout_4.addWidget(self.ui.frame_result, 4, 2)
        self.ui.frame_result.setMinimumWidth(640)
        self.ui.frame_result.setMinimumHeight(480)
        self.ui.result_list_view.currentItemChanged.connect(lambda _: self.UpdateResultView())

        self.ui.btn_clear_canvas.clicked.connect(lambda _: self.ui.canvas.ClearCanvas())
        self.ui.btn_delete_object.clicked.connect(lambda _: self.DeleteObjectListItem())
        self.ui.btn_add_object.clicked.connect(lambda _: self.InsertDrawingToObjectList())

        self.ui.actionLoad_Data.triggered.connect(lambda _: self.LoadData())

        self.ui.object_list_view.currentItemChanged.connect(lambda _: self.ObjectListViewSelectionChanged())
        self.ui.btn_query.clicked.connect(lambda _: self.Query())

        self.ui.actionLoad_sketches.triggered.connect(lambda _: self.LoadSketch())
        self.ui.picked_list_view.itemClicked.connect(lambda _: self.PickedListViewSelectionChanged())
        self.ui.picked_list_view.itemDoubleClicked.connect(lambda _: self.InsertPickedToObjectList())
        self.ui.btn_add_picked.clicked.connect(lambda _:self.InsertPickedToObjectList())
        
        self.coco_idx_class_dict = dict(zip(range(len(COCO_INSTANCE_CATEGORY_NAMES)), COCO_INSTANCE_CATEGORY_NAMES))
        self.coco_class_idx_dict = dict((v,k) for k,v in self.coco_idx_class_dict.items())
        self.LoadProcessedDataImpl("./data/coco_preprocessed.pkl")
        self.LoadSketchImpl("./data/sketchnet_selected")
        

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
        self.retriever = Retriever(self.data_dict, self.coco_idx_class_dict)

        ## load image
        #print("caching coco images..")
        #for k, v in self.data_dict.items():
        #    im = Image.open(k)
        #    v["scaled_centers"] = np.zeros(v["centers"].shape)
        #    v["width"] = im.size[0]
        #    v["height"] = im.size[1]
        #    v["box_image"] = GetBoxImage(img = k,
        #                                 boxes = v["boxes"],
        #                                 pred_cls = v["labels"],
        #                                 cls_score = v["scores"],
        #                                 scale = True)
            
        #print("complete..")

        #query_width = QUERY_RECT[2]
        #query_height = QUERY_RECT[3]

        #from PIL import Image
        
        #for k, v in self.data_dict.items():
        #    im = Image.open(k)
        #    v["scaled_centers"] = np.zeros(v["centers"].shape)
        #    v["width"] = im.size[0]
        #    v["height"] = im.size[1]
        #    v["scaled_centers"][:, 0] = v["centers"][:, 0] * query_width / v["width"]
        #    v["scaled_centers"][:, 1] = v["centers"][:, 1] * query_height / v["height"]

        #with open("./coco_preprocessed.pkl", "wb") as f:
        #    pickle.dump(self.data_dict, f)


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
        label = self.ui.canvas.GetLabel()

        li = QtWidgets.QListWidgetItem(parent = None)
        li.setData(ROLE_UID, self.object_id)
        li.setData(ROLE_SHOWTEXT, "{}".format(label))
        li.setData(ROLE_IMAGE, image)
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
        li.setData(ROLE_IMAGE, image)
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

        if current_selected.data(ROLE_IMAGE) is None:
            return

        image = current_selected.data(ROLE_IMAGE)
        self.ui.frame_selected.SetImage(image)
        self.ui.object_view.SetCurrentSelected(current_selected)

    # build up query object
    def Query(self):
        olv = self.ui.object_list_view
        num_objects = olv.count()
        rect_dict = self.ui.object_view.item_dict

        hits = []

        for i in range(num_objects):
            item = olv.item(i)
            label = item.data(ROLE_SHOWTEXT)
            # skip if not exist
            if (label not in self.coco_class_idx_dict):
                continue
            hits.append(i)

        nhit = len(hits)
        query_dict = { "boxes":   np.zeros((nhit, 4)),
                       "labels":  np.zeros((nhit), dtype = np.int64),
                       "scores":  np.zeros((nhit)),
                       "centers": np.zeros((nhit, 2))}


        # loop for each row of model
        c = 0
        for i in hits:
            item = olv.item(i)
            label = item.data(ROLE_SHOWTEXT)
            uid = item.data(ROLE_UID)
            rect = item.data(ROLE_RECT)
            image = item.data(ROLE_IMAGE)
            mo = rect_dict[uid]
            rect = mo.GetRect()
            pos = mo.pos()

            # x1, y1, x2, y2
            query_dict["boxes"][c, 0] = rect.topLeft().x() + pos.x()
            query_dict["boxes"][c, 1] = rect.topLeft().y() + pos.y()
            query_dict["boxes"][c, 2] = rect.bottomRight().x() + pos.x()
            query_dict["boxes"][c, 3] = rect.bottomRight().y()+ pos.y()
            
            query_dict["labels"][c] = self.coco_class_idx_dict[label]
            query_dict["scores"][c] = 1.0
            query_dict["centers"][c, 0] = round((query_dict["boxes"][c, 0] + query_dict["boxes"][c, 2]) * 0.5)
            query_dict["centers"][c, 1] = round((query_dict["boxes"][c, 1] + query_dict["boxes"][c, 3]) * 0.5)


            #print("#{}: x: {}, y: {}".format(uid, pos.x(), pos.y()))
            c = c + 1

        results = self.retriever.Query(query_dict)

        # update view
        rlv = self.ui.result_list_view
        rlv.clear()
        for r in results:
            li = QtWidgets.QListWidgetItem(parent = None)
            img_path = r[0]
            score = r[2]
            li.setData(ROLE_SHOWTEXT, "{:.3f}: {}".format(score, os.path.basename(img_path)))
            li.setText(li.data(ROLE_SHOWTEXT))

            # get display image
            #box_image = GetBoxImage(img = img_path,
            #                        boxes = self.data_dict[img_path]["boxes"],
            #                        pred_cls = self.data_dict[img_path]["labels"],
            #                        cls_score = self.data_dict[img_path]["scores"],
            #                        scale = True)
            if "box_image" not in self.data_dict[img_path]:
                self.data_dict[img_path]["box_image"] = GetBoxImage(img = img_path,
                                                       boxes = self.data_dict[img_path]["boxes"],
                                                       pred_cls = self.data_dict[img_path]["labels"],
                                                       cls_score = self.data_dict[img_path]["scores"],
                                                       scale = True)

            li.setData(ROLE_IMAGE, self.data_dict[img_path]["box_image"])
            rlv.addItem(li)

        if (len(results)) > 0:
            rlv.setCurrentRow(0)

    def UpdateResultView(self):
        rlv = self.ui.result_list_view
        current_selected = rlv.currentItem()
        if current_selected is None:
            self.ui.frame_result.SetImage(None)
            return

        if current_selected.data(ROLE_IMAGE) is None:
            return

        self.ui.frame_result.SetImage(current_selected.data(ROLE_IMAGE))

