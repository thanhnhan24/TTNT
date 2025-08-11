import sys, time, cv2, numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage, QPixmap

import UI  # UI.py của bạn
from common_ipc import make_queues

class MainApp(QMainWindow, UI.Ui_MainWindow):
    def __init__(self, q):
        super().__init__()
        self.ui = UI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("XArm – GUI (multi-process)")
        self.q = q

        self.robot_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.fx = None
        self.fy = None
        self.auto_center = False
        self.last_aruco = None

        # Button mapping
        self.ui.initButton.clicked.connect(self.load_yolo)
        self.ui.zeroPos.clicked.connect(lambda: self.q["gui_to_robot"].put({"type": "home"}))
        self.ui.clearErrorbutton.clicked.connect(lambda: self.q["gui_to_robot"].put({"type": "clear_error"}))
        self.ui.checkPickupPos.clicked.connect(self.check_pickup_pos)
        self.ui.palletView.clicked.connect(self.move_to_pallet)
        self.ui.calibButton.clicked.connect(lambda: self.q["gui_to_robot"].put({"type": "calib"}))
        self.ui.checkDropPos.clicked.connect(self.check_drop_pos)

        # Poll queues
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_queues)
        self.timer.start(30)

        # mở camera vision
        self.q["gui_to_vision"].put({"type": "camera_open"})

    def load_yolo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn YOLOv8 (.pt)", "", "YOLOv8 (*.pt)")
        if path:
            self.ui.yoloPath.setPlainText(path)
            self.q["gui_to_vision"].put({"type": "load_model", "path": path})

    def poll_queues(self):
        # Vision
        try:
            while True:
                m = self.q["vision_to_gui"].get_nowait()
                t = m.get("type")
                if t == "frame":
                    jpg = m["jpeg"]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.show_img(img)
                elif t == "vision_status":
                    txt = "\n".join(f"{k}: {v}" for k, v in m["counts"].items())
                    self.ui.packageAvailable.setPlainText(txt)
                elif t == "aruco_pose":
                    # pose ArUco 3D
                    self.ui.pickUpYaw.setPlainText(f"{m['yaw_img']:.2f}")
                    cam_offset_x = self._get_plain_float("cameraXoffsets") or 0.0
                    cam_offset_y = self._get_plain_float("cameraYoffsets") or 0.0
                    pickX = self.robot_pose["x"] - (m["tvec"][0]*1000.0) + cam_offset_x
                    pickY = self.robot_pose["y"] + (m["tvec"][1]*1000.0) + cam_offset_y
                    self.ui.pickUpX.setPlainText(f"{pickX:.1f}")
                    self.ui.pickUpY.setPlainText(f"{pickY:.1f}")
                    self.ui.pickUpZ.setPlainText(f"{m['tvec'][2]*1000.0:.1f}")
                elif t == "vision_log":
                    self.log(m["msg"])
        except:
            pass

        # Robot
        try:
            while True:
                m = self.q["robot_to_gui"].get_nowait()
                t = m.get("type")
                if t == "robot_pose":
                    self.robot_pose.update(x=m["x"], y=m["y"], z=m["z"], yaw=m["yaw"])
                    self.ui.xarm_X.setPlainText(f"{m['x']:.2f}")
                    self.ui.xarm_Y.setPlainText(f"{m['y']:.2f}")
                    self.ui.xarm_Z.setPlainText(f"{m['z']:.2f}")
                    self.ui.xarm_theta.setPlainText(f"{m['yaw']:.2f}")
                elif t == "robot_log":
                    self.log(m["msg"])
        except:
            pass

    def show_img(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(self.ui.imgOut.width(), self.ui.imgOut.height(),
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.imgOut.setPixmap(pix)

    def _get_plain_float(self, name):
        if not hasattr(self.ui, name): return None
        w = getattr(self.ui, name)
        try:
            if hasattr(w, "toPlainText"):
                s = w.toPlainText().strip()
            else:
                s = w.text().strip()
            return float(s) if s else None
        except:
            return None

    def check_pickup_pos(self):
        x = self._get_plain_float("pickUpX")
        y = self._get_plain_float("pickUpY")
        yaw = self._get_plain_float("pickUpYaw") or self.robot_pose["yaw"]
        if x is None or y is None:
            self.log("Thiếu tọa độ pickUpX/Y")
            return
        self.q["gui_to_robot"].put({"type": "move", "params": {"x": x, "y": y, "z": 175.0, "yaw": yaw}})

    def move_to_pallet(self):
        x = self._get_plain_float("palletX")
        y = self._get_plain_float("palletY")
        z = self._get_plain_float("palletZ")
        if None in (x, y, z):
            self.log("Thiếu tọa độ pallet")
            return
        self.q["gui_to_robot"].put({"type": "move", "params": {"x": x, "y": y, "z": z, "yaw": 0.0}})

    def check_drop_pos(self):
        x = self._get_plain_float("currDropX")
        y = self._get_plain_float("currDropY")
        z = self._get_plain_float("currDropZ")
        yaw = self._get_plain_float("currDropYaw") or 0.0
        if None in (x, y, z):
            self.log("Thiếu tọa độ drop")
            return
        self.q["gui_to_robot"].put({"type": "move", "params": {"x": x, "y": y, "z": z, "yaw": yaw}})

    def log(self, msg):
        try:
            self.ui.logView.addItem(str(msg))
            self.ui.logView.scrollToBottom()
        except:
            print(msg)
