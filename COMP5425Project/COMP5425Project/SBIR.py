import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QVariant
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor

ROLE_SKETCH = Qt.UserRole

class Canvas(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.drawing = False
        self.lastPoint = QPoint()
        self.image = QPixmap(600, 450)
        self.image.fill(QColor(230, 230, 230))
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
        self.image.fill(QColor(230, 230, 230))
        self.update()


    def GetImage(self):
        return QPixmap(self.image)


class PaintWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.image = QPixmap(600, 450)
        self.image.fill(QColor(230, 230, 230))
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
            self.image = QPixmap(600, 450)
            self.image.fill(QColor(230, 230, 230))
        else:
            self.image = image

        self.update()


class UIManager():

    def __init__(self, ui):
        self.ui = ui
        self.ui.canvas = Canvas(parent = ui.groupBox)
        ui.groupBox.layout().insertWidget(0, self.ui.canvas)
        self.ui.frame_selected = PaintWidget(parent = ui.groupBox_2)
        ui.groupBox_2.layout().insertWidget(0, self.ui.frame_selected)

        self.ui.btn_clear_canvas.clicked.connect(lambda _: self.ui.canvas.ClearCanvas())
        self.ui.btn_delete_object.clicked.connect(lambda _: self.DeleteObjectListItem())
        self.ui.btn_add_object.clicked.connect(lambda _: self.InsertDrawingToObjectList())

        self.ui.object_list_view.currentItemChanged.connect(lambda _: self.ObjectListViewSelectionChanged())

        # dummy
        for d in range(10):
            li = QtWidgets.QListWidgetItem(parent = self.ui.object_list_view)
            li.setText(str(d))
            self.ui.object_list_view.addItem(li)

        self.ui.object_list_view.setCurrentRow(0)


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
        li = QtWidgets.QListWidgetItem(parent = olv)
        li.setText("object")
        li.setData(ROLE_SKETCH, image)
        olv.addItem(li)


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
