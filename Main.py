import UI
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QThread, Signal, Qt
from PySide2.QtGui import QImage, QPixmap
import sys, os, queue
import pyrealsense2 as rs
import numpy as np
from xarm.wrapper import XArmAPI

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
arm = None  # biến robot chung

class RobotControlThread(QThread):
    def __init__(self, arm, lock, app=None):
        super().__init__()
        self.arm = arm
        self.lock = lock
        self.app = app
        self.command_queue = queue.Queue()
        self.running = True

    def run(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=1)
                if command == "stop":
                    self.running = False
                    break
                elif command == "connect":
                    self.connect_arm()
                elif command == "disconnect":
                    self.disconnect_arm()
            except queue.Empty:
                continue

    def connect_arm(self): pass
    def disconnect_arm(self): pass


class CameraThread(QThread):
    frame_signal = Signal(object)  # emits np.ndarray (color, HxWx3, uint8 BGR)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline = None

    def run(self):
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            # LẤY ẢNH MÀU
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._pipeline.start(config)

            while not self.isInterruptionRequested():
                frames = self._pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())  # uint8 BGR
                self.frame_signal.emit(color_image)
        except Exception as e:
            print(f"[CameraThread] error: {e}")
        finally:
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception as e:
                print(f"[CameraThread] stop error: {e}")
            self._pipeline = None


class MainApp(QMainWindow, UI.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = UI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("XArm Control")

        # Khởi tạo camera thread
        self.camera_thread = CameraThread(self)
        self.camera_thread.frame_signal.connect(self.on_color_frame)
        self.camera_thread.start()

        # Giữ tham chiếu để tránh dữ liệu bị thu hồi sớm
        self._last_qimage = None
        self._last_pixmap = None

    def on_color_frame(self, bgr_image: np.ndarray):
        """
        bgr_image: np.ndarray dtype=uint8, shape (H, W, 3)
        Convert to RGB QPixmap and display on ui.imgOut (QLabel).
        """
        if bgr_image is None or bgr_image.size == 0:
            return

        # BGR -> RGB, đảm bảo bộ nhớ liên tục
        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1])  # (H,W,3) RGB

        h, w, ch = rgb.shape
        print(f"[MainApp] Received frame: {h}x{w}, channels: {ch}")
        bytes_per_line = ch * w

        # Tạo QImage (tham chiếu bộ đệm numpy), rồi copy để tách bộ nhớ an toàn
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qimg = qimg.copy()  # detach để qimg sống độc lập sau khi hàm kết thúc
        self._last_qimage = qimg

        pix = QPixmap.fromImage(qimg)
        self._last_pixmap = pix

        # Scale khớp QLabel (giữ tỉ lệ, làm mượt)
        if hasattr(self.ui, "imgOut") and self.ui.imgOut is not None:
            target_w = self.ui.imgOut.width()
            target_h = self.ui.imgOut.height()
            pix_scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.imgOut.setPixmap(pix_scaled)

    def start_camera(self):
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = CameraThread(self)
            self.camera_thread.frame_signal.connect(self.on_color_frame)
            self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.requestInterruption()
            self.camera_thread.wait()

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
