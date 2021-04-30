# coding=utf-8

import sys
from pathlib import Path
import webbrowser
from typing import Tuple

import numpy as np
import cv2

from PyQt5.QtCore import QDir, Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QMessageBox, QFileDialog, QLabel, QSpinBox, QPushButton,
    QActionGroup, QAction, QSizePolicy, QHBoxLayout,
)

from ui_grabcut import Ui_MainWindow


class Canvas(QLabel):
    """Canvas for drawing mask layer on Image.
    """

    mousePressed = pyqtSignal()
    mouseMoved = pyqtSignal(int, int, int, int)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        # self.setStyleSheet("border: 1px solid red;")
        self.last_x, self.last_y = None, None

    def mousePressEvent(self, e):
        self.mousePressed.emit()

    def mouseMoveEvent(self, e):
        x, y = e.x(), e.y()

        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return

        self.mouseMoved.emit(self.last_x, self.last_y, x, y)
        self.last_x, self.last_y = x, y

    def mouseReleaseEvent(self, e):
        self.last_x, self.last_y = None, None


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # orign image data
        self.img = None
        # mask layer for grabcut
        self.mask = None
        # history masks for undo
        self.masks = []
        # grabcut algorithm param iterCount
        self.iterCount = 5

        # canvas image cache
        self.imgWithMask = None
        # mask mode to color, don't use dict, too slow!
        self.mode2color = (
            # cv2.GC_BGD == 0
            np.array([0, 0, 255], dtype=np.uint8),
            # cv2.GC_FGD == 1
            np.array([0, 255, 0], dtype=np.uint8),
            # cv2.GC_PR_BGD == 2
            np.array([0, 0, 120], dtype=np.uint8),
            # cv2.GC_PR_FGD == 3
            np.array([0, 120, 0], dtype=np.uint8),
        )
        # NONE mean none of (BGD/FGD/PR_BGD/PR_FGD)
        self.GC_NONE = 255
        # mask layer alpha
        self.alpha = 0.3

        self.imgPath = Path.cwd()
        self.penSize = 40

        # init ui order matter
        self.initUI()

    def grabCut(self, iterCount):
        if self.img is None:
            self.showMessage("No image")
            return
        # before grabcut, save mask to stack
        self.pushMask()
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        _ = cv2.grabCut(self.img, self.mask, None, bgdModel,
                        fgdModel, iterCount, cv2.GC_INIT_WITH_MASK)
        self.drawPartialImgWithMask(self.masks[-1], self.mask)

        # display result
        self.ui.displayResultAction.setChecked(True)
        self.repaint()

    def drawingMask(self, x1, y1, x2, y2):
        """drawing an small partial of the mask layer,
        which is a small line segment.
        """
        if self.img is None:
            return
        # when hidden mask or display result, don't draw mask
        if self.ui.hiddenMaskAction.isChecked() or \
                self.ui.displayResultAction.isChecked():
            return

        if self.ui.prFgdAction.isChecked():
            mode = cv2.GC_PR_FGD
        elif self.ui.prBgdAction.isChecked():
            mode = cv2.GC_PR_BGD
        elif self.ui.fgdAction.isChecked():
            mode = cv2.GC_FGD
        else:  # bgdAction
            mode = cv2.GC_BGD

        cv2.line(self.mask, (x1, y1), (x2, y2), mode, self.penSize)
        partialMask = np.zeros(self.mask.shape, np.uint8)
        # GC_BGD is 0, can't use 0 as default
        partialMask.fill(self.GC_NONE)
        cv2.line(partialMask, (x1, y1), (x2, y2), mode, self.penSize)

        indices = np.where(partialMask != self.GC_NONE)
        if indices[0].size == 0:
            # nothing new in partialMask
            return
        self.imgWithMask[indices] = (1 - self.alpha)*self.img[indices] + \
            self.alpha*self.mode2color[mode]

        self.repaint()

    def pushMask(self):
        """push a mask to history list masks for undo.
        """
        # if mask hasn't changed
        if len(self.masks) > 0 and np.array_equal(self.masks[-1], self.mask):
            return

        self.masks.append(self.mask.copy())

    def drawPartialImgWithMask(self, curMask, newMask):
        """draw partial imgWithMask.

        mask changed from curMask to newMask, only draw the changed part.
        """
        # redraw partial imgWithMask
        indices = np.where(curMask != newMask)
        if indices[0].size == 0:
            # two masks are equal
            return
        self.imgWithMask[indices] = (1-self.alpha)*self.img[indices] + \
            self.alpha*np.array([self.mode2color[m] for m in newMask[indices]])

    def getResult(self):
        """use mask cuf off forground area as final result.
        """
        result_mask = np.where((self.mask == 2) | (
            self.mask == 0), 0, 1).astype('uint8')
        return self.img*result_mask[:, :, np.newaxis]

    def repaint(self):
        """repaint cavans.
        """
        if self.img is None:
            self.showMessage("No image")
            return

        if self.ui.displayResultAction.isChecked():
            img = self.getResult()
        elif self.ui.hiddenMaskAction.isChecked():
            img = self.img
        else:
            img = self.imgWithMask

        # convert opencv image to qt image
        height, width, _ = img.shape
        bytesOfLine = 3*width
        image = QImage(img.data, width, height,
                       bytesOfLine, QImage.Format_RGB888).rgbSwapped()
        self.canvas.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        # merge designer ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # right box on toolbar
        rightBox = QWidget(self.ui.toolBar)
        boxLayout = QHBoxLayout()

        # grabcut iterCount spinbox
        boxLayout.addWidget(QLabel("iterCount"))
        self.iterCountSpinBox = QSpinBox(self)
        self.iterCountSpinBox.setRange(1, 100)
        self.iterCountSpinBox.setValue(5)
        boxLayout.addWidget(self.iterCountSpinBox)

        boxLayout.addStretch(1)

        # pen size spinbox
        boxLayout.addWidget(QLabel("pen"))
        self.penSizeSpinBox = QSpinBox(self)
        self.penSizeSpinBox.setRange(1, 500)
        self.penSizeSpinBox.setSingleStep(5)
        self.penSizeSpinBox.setValue(40)
        boxLayout.addWidget(self.penSizeSpinBox)

        rightBox.setLayout(boxLayout)
        self.ui.toolBar.addWidget(rightBox)

        self.canvas = Canvas(self)
        self.ui.scrollArea.setWidget(self.canvas)
        # canvas align center in scroll area
        self.ui.scrollArea.setAlignment(Qt.AlignCenter)
        # fixed canvas that make it easier to select mask layer
        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # 4 types of mask layer flags
        actionGroup = QActionGroup(self)
        actionGroup.addAction(self.ui.fgdAction)
        actionGroup.addAction(self.ui.bgdAction)
        actionGroup.addAction(self.ui.prFgdAction)
        actionGroup.addAction(self.ui.prBgdAction)

        # handle events
        self.ui.displayResultAction.triggered.connect(self.repaint)
        self.ui.hiddenMaskAction.triggered.connect(self.repaint)
        self.ui.exitAction.triggered.connect(self.close)
        self.penSizeSpinBox.valueChanged.connect(self.setPenSize)
        self.iterCountSpinBox.valueChanged.connect(self.setIterCount)

        self.ui.opencvAction.triggered.connect(lambda: webbrowser.open(
            'https://opencv-python-tutroals.readthedocs.io/en/'
            'latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html'
        ))

        self.canvas.mousePressed.connect(self.pushMask)
        self.canvas.mouseMoved.connect(self.drawingMask)

        self.resetUiToDrawMaskMode()

    def resetUiToDrawMaskMode(self):
        """reset ui to draw mask mode.
        """
        self.ui.prFgdAction.setChecked(True)
        self.ui.displayResultAction.setChecked(False)
        self.ui.hiddenMaskAction.setChecked(False)

    def setPenSize(self, v):
        self.penSize = v

    def setIterCount(self, v):
        self.iterCount = v

    def showMessage(self, msg):
        self.ui.statusbar.showMessage(msg)

    @pyqtSlot(name="on_openAction_triggered")
    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open File", str(self.imgPath))
        if not fileName:
            return

        self.imgPath = Path(fileName).parent

        self.img = cv2.imread(fileName)
        self.reset()

    @pyqtSlot(name="on_saveAction_triggered")
    def saveResult(self):
        if self.img is None:
            self.showMessage("no result to save")
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save File", str(self.imgPath))
        if not fileName:
            return

        self.imgPath = Path(fileName).parent

        result = self.getResult()
        cv2.imwrite(fileName, result)

    @pyqtSlot(name="on_exportMaskAction_triggered")
    def exportMask(self):
        if self.mask is None or not self.mask.any():
            self.showMessage("no mask")
            return
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Mask", str(self.imgPath))
        if not fileName:
            return

        self.imgPath = Path(fileName).parent
        cv2.imwrite(fileName, self.mask)

    @pyqtSlot(name="on_undoAction_triggered")
    def undo(self):
        if len(self.masks) == 0:
            self.showMessage("undo stack is empty")
            return

        prevMask = self.masks.pop()
        self.drawPartialImgWithMask(self.mask, prevMask)
        self.mask = prevMask

        # after undo, uncheck display result and hidden mask
        self.resetUiToDrawMaskMode()
        self.repaint()

    @pyqtSlot(name="on_resetAction_triggered")
    def reset(self):
        if self.img is None:
            self.showMessage("No image")
            return

        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.mask.fill(cv2.GC_PR_BGD)
        self.masks = []

        # re-create imgWidthMask
        self.imgWithMask = np.zeros(self.img.shape, np.uint8)
        self.imgWithMask[...] = (1-self.alpha)*self.img + \
            self.alpha*self.mode2color[cv2.GC_PR_BGD]

        self.resetUiToDrawMaskMode()
        self.repaint()

    @pyqtSlot(name="on_grabCutAction_triggered")
    def runGrabCut(self):
        self.grabCut(self.iterCount)

    @pyqtSlot(name="on_singleStepAction_triggered")
    def runGrabCutSingleStep(self):
        self.grabCut(1)

    def closeEvent(self, evt):
        # maybe popup a dialog to ask user accept or ignore
        evt.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
