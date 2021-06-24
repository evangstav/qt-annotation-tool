import glob
import os
import shutil
import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QRadioButton,
)

import torch

sys.path.append("../")
from models.models import UNet

MODEL_PATH = "trained_models/segmentation_mixed_augmented_UNet_1604044090.034343.pth"

def load_model(path):
    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(input_filters=1, filters=64, N=2).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model


# os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
IMAGE_WIDTH = 2048 // 2
IMAGE_HEIGHT = 1536 // 2

BOUNDARY_COLOR = (0, 255, 0)
CENTER_COLOR = (255, 255, 255)


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(800, 600)
        self.counter = 0
        self.usealgo = True
        self.model = load_model(MODEL_PATH)

        self.image_frame = QtWidgets.QLabel()
        self.image_frame.setScaledContents(True)
        self.image_frame.setFixedSize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.image_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.image_frame.mousePressEvent = self.getPixel

        layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        self.button = QPushButton("Next")
        self.button.clicked.connect(self.get_next_and_show)
        top_layout.addWidget(self.button)
        self.radio_button = QRadioButton("Use Algo")
        self.radio_button.setChecked(True)
        self.radio_button.toggled.connect(self.btnstate)
        self.button_dir = QPushButton("Select Directory")
        self.button_dir.clicked.connect(self.get_dir)
        top_layout.addWidget(self.button_dir)
        top_layout.addWidget(self.radio_button)

        layout.addLayout(top_layout)
        layout.addWidget(self.image_frame)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowTitle("Annotation Tool")

    @QtCore.pyqtSlot()
    def btnstate(self):
        if self.radio_button.isChecked():
            self.usealgo = True
        else:
            self.usealgo = False

    @QtCore.pyqtSlot()
    def get_next_and_show(self):
        choice = QMessageBox.question(
            self,
            "next?",
            "Save labelled image and go to next?",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if choice == QMessageBox.Yes:
            # self.save_image()
            self.get_nex_image()
            self.show_image()

    @QtCore.pyqtSlot()
    def save_image(self):
        self._image.save_image_and_data(self.dest_dir)

    def get_nex_image(self):
        if self.counter >= len(self.files):
            QMessageBox.about(self, "", "You labelled all files in this folder!")
            return

        print(self.files[self.counter])
        self._image = Image(self.files[self.counter], self.model)
        self.counter += 1

    def show_image(self):
        self.image = self._image.overlay_circles_on_top()
        self.image = QtGui.QImage(
            self.image.data,
            self.image.shape[1],
            self.image.shape[0],
            QtGui.QImage.Format_RGB888,
        )

        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

    def getPixel(self, event):
        x = event.pos().x() * 2
        y = event.pos().y() * 2
        if event.button() == Qt.LeftButton:
            self._image.find_closes_circle(x, y)
        if event.button() == Qt.RightButton:
            self._image.add_circle(x, y)

        self.show_image()

    def get_dir(self):
        dest_dir = QFileDialog.getExistingDirectory(
            None,
            "Open working directory",
            os.getcwd(),
            QFileDialog.ShowDirsOnly,
        )
        self.src_dir = dest_dir
        self.dest_dir = os.path.join(
            self.src_dir,
            os.pardir,
            f"labelled_{os.path.split(self.src_dir)[-1]}",
        )
        self.files = glob.glob(os.path.join(self.src_dir, "*.jpg"))

        if not os.path.exists(self.dest_dir):
            print(f"Creating directory{self.dest_dir}")
            os.makedirs(self.dest_dir)
        else:
            self.files = self.remove_labelled_files(self.files, self.dest_dir)

        print(f"Saving labelled images in {self.dest_dir}")
        self.get_nex_image()
        self.show_image()

    @staticmethod
    def remove_labelled_files(files, dest_dir):
        labelled_files = [
            os.path.basename(os.path.splitext(lf)[0])
            for lf in glob.glob(os.path.join(dest_dir, "*.npy"))
        ]
        return [
            f
            for f in files
            if os.path.basename(os.path.splitext(f)[0]) not in labelled_files
        ]


class Image:
    def __init__(self, file_path, model):
        self.file_path = file_path
        self.image = cv2.imread(self.file_path, 0)
        self.model = model
        self.circles = self.detect_circles()
        self.radius = 30

    def detect_circles(self):
        # resize image
        resized_image = cv2.resize(self.image, (256, 256))
        # to Tensor and unsqueeze batch dimension
        tensor_image = (
            torch.unsqueeze(torch.unsqueeze(torch.Tensor(resized_image), 0), 0) / 255
        )
        # predict with model
        prediction = self.model(tensor_image)[0, 0].cpu().detach().numpy() > 0.5
        # get centers from resized mask
        # resize prediction to original size

        resized_prediction = cv2.resize(prediction.astype(np.uint8) * 255, (2048, 1536))

        # find centers
        contours, hierarchy = cv2.findContours(resized_prediction, 1, 2)

        def find_center(contour):
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

        circles = [find_center(contour) for contour in contours]

        return np.array(circles)

    def overlay_circles_on_top(self, alpha=0.5):

        circles = np.uint16(np.around(self.circles))
        img_over_lay = self.image.copy()
        img_over_lay = np.stack((img_over_lay,) * 3, axis=-1)

        for no, i in enumerate(circles):
            img_over_lay = cv2.circle(
                img_over_lay, (i[0], i[1]), 30, BOUNDARY_COLOR, 10
            )
            img_over_lay = cv2.circle(img_over_lay, (i[0], i[1]), 2, CENTER_COLOR, 5)

        return cv2.addWeighted(
            img_over_lay,
            alpha,
            np.stack((self.image,) * 3, axis=-1),
            1 - alpha,
            0,
        )

    def find_closes_circle(self, x, y):
        x_ys = self.circles
        distances = np.linalg.norm(x_ys - np.array([x, y]), axis=1)
        idx = np.argmin(distances)
        if np.min(distances).item() < self.radius / 2:
            self.circles = np.concatenate(
                [self.circles[:idx], self.circles[idx + 1 :]], axis=0
            )

    def add_circle(self, x, y):
        self.circles = np.concatenate([self.circles, np.array([[x, y]])], axis=0)

    def save_image_and_data(self, dest_dir):
        file_name = os.path.basename(self.file_path)
        shutil.copyfile(self.file_path, os.path.join(dest_dir, file_name))
        circles = self.circles.tolist()
        np.save(os.path.join(dest_dir, os.path.splitext(file_name)[0]), circles)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    window = MainWindow()
    window.show()
    app.exec_()
