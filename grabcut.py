# coding=utf-8

import os
import sys
from pathlib import Path

import numpy as np
import cv2

from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QMessageBox, QFileDialog, QLabel, QSpinBox, QPushButton,
    QAction, QSizePolicy, QHBoxLayout, QActionGroup,
)

from ui_grabcut import Ui_MainWindow


class Canvas(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        # self.setStyleSheet("border: 1px solid red;")
        self.last_x, self.last_y = None, None

    def mousePressEvent(self, e):
        self.parent.pushMask()

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        self.parent.drawLine((self.last_x, self.last_y), (e.x(), e.y()))

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.img = []
        self.mask = []
        self.result = []
        self.penSize = 10
        self.iterCount = 5

        # history masks for undo
        self.masks = []

        self.imgPath = Path.cwd()

        self.initUI()

    def initUI(self):
        # merge designer ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # right box on toolbar
        rightBox = QWidget(self.ui.toolBar)
        boxLayout = QHBoxLayout()

        # pen size spinbox
        boxLayout.addWidget(QLabel("pen size:"))
        self.penSizeSpinBox = QSpinBox(self)
        self.penSizeSpinBox.setRange(1, 100)
        self.penSizeSpinBox.setSingleStep(5)
        self.penSizeSpinBox.setValue(self.penSize)
        boxLayout.addWidget(self.penSizeSpinBox)
        boxLayout.addStretch(1)

        # grabcut iterCount spinbox
        boxLayout.addWidget(QLabel("iter count:"))
        self.iterCountSpinBox = QSpinBox(self)
        self.iterCountSpinBox.setRange(1, 100)
        self.iterCountSpinBox.setValue(self.iterCount)
        boxLayout.addWidget(self.iterCountSpinBox)

        # grabcut button
        self.grabCutButton = QPushButton("grabcut")
        boxLayout.addWidget(self.grabCutButton)
        # grabcut one step button
        self.stepButton = QPushButton("step")
        boxLayout.addWidget(self.stepButton)

        rightBox.setLayout(boxLayout)
        self.ui.toolBar.addWidget(rightBox)

        self.canvas = Canvas(self)
        self.ui.scrollArea.setWidget(self.canvas)
        self.ui.scrollArea.setAlignment(Qt.AlignCenter)
        # fixed canvas that make it easier to select mask layer
        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # 4 types of mask layer flags
        actionGroup = QActionGroup(self)
        actionGroup.addAction(self.ui.fgdAction)
        actionGroup.addAction(self.ui.bgdAction)
        actionGroup.addAction(self.ui.prFgdAction)
        actionGroup.addAction(self.ui.prBgdAction)
        self.ui.prFgdAction.setChecked(True)

        # handle events
        self.ui.openAction.triggered.connect(self.onOpenActionTriggered)
        self.ui.saveAction.triggered.connect(self.onSaveActionTriggered)
        self.ui.undoAction.triggered.connect(self.onUndoActionTriggered)
        # use lambda to adapt the the problem of insufficient parameters
        self.ui.exitAction.triggered.connect(lambda: self.closeEvent(None))
        self.penSizeSpinBox.valueChanged.connect(self.onPenSizeChanged)
        self.iterCountSpinBox.valueChanged.connect(self.onIterCountChanged)
        self.grabCutButton.clicked.connect(self.onGrabCutButtonClicked)
        self.stepButton.clicked.connect(self.onStepButtonClicked)

    def onOpenActionTriggered(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open File", str(self.imgPath.parent))
        if not fileName:
            return

        self.imgPath = Path(fileName)

        self.img = cv2.imread(fileName)
        self.resetMaskLayer()
        self.result = []
        self.repaint()

    def onSaveActionTriggered(self):
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save File", str(self.imgPath.parent))
        if not fileName:
            return

        self.imgPath = Path(fileName)

        cv2.imwrite(fileName, self.result)

    def onUndoActionTriggered(self):
        if len(self.masks) == 0:
            return

        print("undo", len(self.masks))
        self.mask = self.masks.pop()
        self.repaint()

    def onPenSizeChanged(self):
        self.penSize = self.penSizeSpinBox.value()

    def onIterCountChanged(self):
        self.iterCount = self.iterCountSpinBox.value()

    def onGrabCutButtonClicked(self):
        self.grabCut(self.iterCount)

    def onStepButtonClicked(self):
        self.grabCut(1)

    def closeEvent(self, evt):
        sys.exit(0)

    def resetMaskLayer(self):
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.mask.fill(cv2.GC_PR_BGD)

    def grabCut(self, iterCount):
        img = self.img.copy()
        mask = self.mask
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        _ = cv2.grabCut(img, mask, None, bgdModel,
                        fgdModel, iterCount, cv2.GC_INIT_WITH_MASK)
        mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img*mask_final[:, :, np.newaxis]
        cv2.imwrite('output.png', img)
        self.result = img
        self.repaint()

    def drawLine(self, start, end):
        if len(self.img) == 0:
            return

        if self.ui.prFgdAction.isChecked():
            cv2.line(self.mask, start, end, cv2.GC_PR_FGD, self.penSize)
        elif self.ui.prBgdAction.isChecked():
            cv2.line(self.mask, start, end, cv2.GC_PR_BGD, self.penSize)
        elif self.ui.fgdAction.isChecked():
            cv2.line(self.mask, start, end, cv2.GC_FGD, self.penSize)
        elif self.ui.bgdAction.isChecked():
            cv2.line(self.mask, start, end, cv2.GC_BGD, self.penSize)

        self.repaint()

    def pushMask(self):
        if len(self.masks) > 0 and np.array_equal(self.masks[-1], self.mask):
            print("nothing changed")
            return

        self.masks.append(self.mask.copy())

    def getImageWithMask(self):
        # draw mask layer, exclude GC_PR_BGD
        mask = np.zeros(self.img.shape, dtype=np.uint8)
        mask[self.mask == cv2.GC_PR_FGD, :] = 0, 0, 120
        mask[self.mask == cv2.GC_BGD, :] = 1, 0, 0
        mask[self.mask == cv2.GC_FGD, :] = 0, 0, 255

        # mix mask and img
        alpha = 0.7
        indices = np.where((mask[:, :, 0] != 0) | (
            mask[:, :, 1] != 0) | (mask[:, :, 2] != 0))
        img = self.img.copy()
        img[indices] = (1 - alpha)*img[indices] + alpha*mask[indices]

        return img

    def repaint(self):
        if len(self.result) != 0:
            img = self.result
        else:
            img = self.getImageWithMask()

        # convert opencv image to qt image
        height, width, _ = img.shape
        bytesOfLine = 3*width
        image = QImage(img.data, width, height,
                       bytesOfLine, QImage.Format_RGB888).rgbSwapped()

        self.canvas.setPixmap(QPixmap.fromImage(image))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
