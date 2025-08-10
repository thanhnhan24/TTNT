import UI
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QThread, Signal, Qt, QTimer
from PySide2.QtGui import QImage, QPixmap
import sys, os, queue, threading, time, cv2, math
import pyrealsense2 as rs
import numpy as np
from xarm.wrapper import XArmAPI

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# =========================
# Camera thread (RealSense)
# =========================

class CameraThread(QThread):
    # Truyền ndarray BGR ra ngoài để main xử lý/YOLO
    frame_signal = Signal(object)  # emits: np.ndarray (HxWx3, uint8 BGR)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline = None
        self.app = parent

    def _log(self, msg):
        if self.app and hasattr(self.app, "log"):
            self.app.log(msg)
        else:
            print(msg)

    def run(self):
        try:
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self._pipeline.start(cfg)

            # === LẤY INTRINSICS MÀU, EMIT 1 LẦN ===
            try:
                cprof = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                intr = cprof.get_intrinsics()
                # K, dist theo OpenCV
                K = np.array([[intr.fx, 0, intr.ppx],
                              [0, intr.fy, intr.ppy],
                              [0,       0,        1]], dtype=np.float32)
                dist = np.array(intr.coeffs, dtype=np.float32)
                self.intrinsics_signal.emit({"K": K, "dist": dist})
            except Exception as e:
                self._log(f"[CameraThread] intrinsics error: {e}")

            while not self.isInterruptionRequested():
                frames = self._pipeline.wait_for_frames(5000)
                color = frames.get_color_frame()
                if not color:
                    continue
                bgr = np.asanyarray(color.get_data())
                self.frame_signal.emit(bgr)

        except Exception as e:
            self._log(f"[CameraThread] {e}")
        finally:
            try:
                if self._pipeline:
                    self._pipeline.stop()
            except Exception as e:
                self._log(f"[CameraThread] stop error: {e}")
            self._pipeline = None


# =========================
# xArm connect worker (không block UI)
# =========================
class XArmWorker(QThread):
    ready = Signal(object)
    log_sig = Signal(str)

    def __init__(self, ip: str):
        super().__init__()
        self.ip = ip

    def run(self):
        try:
            self.log_sig.emit(f"Đang kết nối xArm tại {self.ip} ...")
            arm = XArmAPI(self.ip)
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            arm.clean_error()
            time.sleep(0.8)
            arm.move_gohome(wait=True)
            self.log_sig.emit("Kết nối xArm thành công.")
            self.ready.emit(arm)
        except Exception as e:
            self.log_sig.emit(f"Không thể kết nối xArm: {e}")

# =========================
# Robot control thread
# =========================
class RobotControlThread(QThread):
    log_sig = Signal(str)

    def __init__(self, arm=None, lock=None, app=None):
        super().__init__()
        self.arm = arm
        self.lock = lock or threading.Lock()
        self.app = app
        self.q = queue.Queue()
        self._running = True

    def run(self):
        while self._running and (self.app is None or not self.app.stop_event.is_set()):
            try:
                cmd = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if cmd['type'] == 'move':
                    p = cmd['params']
                    self._move(p['x'], p['y'], p['z'], p['yaw'])
                elif cmd['type'] == 'home':
                    self._home()
            except Exception as e:
                self.log_sig.emit(f"[RobotControl] {e}")

    def _move(self, x, y, z, yaw):
        if self.arm is None:
            self.log_sig.emit("Chưa có kết nối xArm.")
            return

        # speed lấy thẳng từ UI (mm/s). Ép tối thiểu 1 cho chắc.
        spd = None
        try:
            if hasattr(self.app, "ui") and hasattr(self.app.ui, "speed"):
                spd = float(self.app.ui.speed.value())
                if spd <= 0:
                    spd = 39 # default
        except Exception:
            spd = None  # để SDK dùng mặc định nếu đọc lỗi

        # Lấy orientation hiện tại để KHÔNG xoay khi move
        r0 = p0 = None
        try:
            with self.lock:
                pos = self.arm.get_position(is_radian=False)
            if pos and pos[0] == 0:
                r0, p0, _y0 = pos[1][3], pos[1][4], pos[1][5]
        except Exception:
            r0 = p0 = None  # nếu đọc lỗi thì để None

        with self.lock:
            code = self.arm.set_position(
                x=x, y=y, z=z,
                roll=r0, pitch=p0, yaw=yaw,   # giữ nguyên roll/pitch hiện tại
                speed=spd,                    # dùng speed từ UI
                wait=True
            )
        if code != 0:
            self.log_sig.emit(f"[xArm] set_position lỗi code={code}, speed={spd}, keep r/p={r0},{p0}")


    def _home(self):
        if self.arm is None:
            self.log_sig.emit("Chưa có kết nối xArm.")
            return

        spd = None
        try:
            if hasattr(self.app, "ui") and hasattr(self.app.ui, "speed"):
                spd = int(self.app.ui.speed.value())
        except Exception:
            spd = None

        if spd <= 0:
            spd = 39 # default
        with self.lock:
            # move_gohome có thể nhận speed (tuỳ bản SDK). Nếu không hỗ trợ, bỏ tham số đi.
            try:
                self.arm.move_gohome(speed=spd, wait=True)
            except TypeError:
                self.arm.move_gohome(wait=True)


    # Enqueue APIs
    def move(self, x, y, z, yaw):
        self.q.put({'type': 'move', 'params': {'x': x, 'y': y, 'z': z, 'yaw': yaw}})

    def go_home(self):
        self.q.put({'type': 'home'})

    def stop(self):
        self._running = False
        self.wait()

# =========================
# Robot monitor (tùy chọn)
# =========================
class RobotMonitorThread(QThread):
    update_sig = Signal(float, float, float, float)
    log_sig = Signal(str)

    def __init__(self, arm=None, lock=None, app=None):
        super().__init__()
        self.arm = arm
        self.lock = lock or threading.Lock()
        self.app = app
        self._running = True

    def run(self):
        while self._running and (self.app is None or not self.app.stop_event.is_set()):
            try:
                if self.arm is None:
                    time.sleep(0.5); continue
                with self.lock:
                    pos = self.arm.get_position(is_radian=False)
                if pos and pos[0] == 0:
                    x, y, z, roll, pitch, yaw = pos[1]
                    self.update_sig.emit(x, y, z, yaw)
            except Exception as e:
                self.log_sig.emit(f"[RobotMonitor] {e}")
            time.sleep(0.5)

    def stop(self):
        self._running = False
        self.wait()

# =========================
# Main window
# =========================
class MainApp(QMainWindow, UI.Ui_MainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.ui = UI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("XArm Control (Camera + Robot)")
        # trong __init__ của MainApp
        self.detect_counts = {}
        # trạng thái chung
        self.stop_event = threading.Event()
        self._arm = None
        self.arm_lock = threading.Lock()

        # logging
        self.log_signal.connect(self.print_to_listview)
        self.cam_K = None        # camera matrix 3x3 (numpy)
        self.cam_dist = None     # distortion (numpy)
        self.aruco_size_m = 0.035  # kích thước cạnh tag (m) – chỉnh theo tag thực tế
        # Robot threads
        self.robot_ctrl = RobotControlThread(arm=None, lock=self.arm_lock, app=self)
        self.robot_ctrl.log_sig.connect(self.log)
        self.robot_ctrl.start()

        self.robot_mon = RobotMonitorThread(arm=None, lock=self.arm_lock, app=self)
        self.robot_mon.update_sig.connect(self._on_robot_pose)
        self.robot_mon.log_sig.connect(self.log)
        self.robot_mon.start()
        self.camera_thread.frame_signal.connect(self._on_frame)
        self.camera_thread.intrinsics_signal.connect(self._on_intrinsics)

        # Camera thread (khởi động sau khi UI lên)
        self.camera_thread = None
        self.yolo_model = None
        self.yolo_names = None
        QTimer.singleShot(0, self.start_camera)

        # Gán nút theo tên mới
        if hasattr(self.ui, "initButton"):
            self.ui.initButton.clicked.connect(self.connect_hardware)
        if hasattr(self.ui, "zeroPos"):
            self.ui.zeroPos.clicked.connect(self.go_home)
        if hasattr(self.ui, "palletView"):
            self.ui.palletView.clicked.connect(self.move_to_pallet)
        if hasattr(self.ui, "clearErrorbutton"):
            self.ui.clearErrorbutton.clicked.connect(self.clear_error)

    # ===== log helpers =====
    def log(self, msg: str):
        self.log_signal.emit(str(msg))

    def print_to_listview(self, msg: str):
        # hỗ trợ QListWidget hoặc QListView tên logView (nếu có)
        try:
            from PySide2.QtWidgets import QListWidget, QListView
            if hasattr(self.ui, "logView") and self.ui.logView is not None:
                if isinstance(self.ui.logView, QListWidget):
                    self.ui.logView.addItem(str(msg))
                    self.ui.logView.scrollToBottom()
                    return
                elif isinstance(self.ui.logView, QListView):
                    from PySide2.QtCore import QStringListModel
                    if not hasattr(self, '_log_model'):
                        self._log_model = QStringListModel()
                        self._log_data = []
                        self.ui.logView.setModel(self._log_model)
                    self._log_data.append(str(msg))
                    if len(self._log_data) > 1000:
                        self._log_data = self._log_data[-1000:]
                    self._log_model.setStringList(self._log_data)
                    idx = self._log_model.index(len(self._log_data) - 1)
                    self.ui.logView.scrollTo(idx)
                    return
        except Exception:
            pass
        print(msg)

    # ===== Camera =====
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return
        self.camera_thread = CameraThread(self)
        self.camera_thread.frame_signal.connect(self._on_frame)
        self.camera_thread.start()
        self.log("Camera started.")

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.requestInterruption()
            self.camera_thread.wait()
            self.log("Camera stopped.")

    def _on_frame(self, bgr: np.ndarray):
        """Nhận frame BGR từ CameraThread. Nếu đã load YOLO -> chạy detect rồi hiển thị."""
        if bgr is None or bgr.size == 0 or not hasattr(self.ui, "imgOut"):
            return

        vis_bgr = bgr
        count_box = None  # sẽ cập nhật cho UI

        if self.yolo_model is not None:
            try:
                rgb = bgr[:, :, ::-1]
                results = self.yolo_model.predict(rgb, verbose=False, imgsz=640, conf=0.65, iou=0.01, device=0)
                if results:
                    r = results[0]
                    # --- ĐẾM SỐ LƯỢNG THEO CLASS ---
                    names = getattr(r, "names", None) or getattr(self.yolo_model, "names", {})
                    cls = None
                    if hasattr(r, "boxes") and hasattr(r.boxes, "cls") and r.boxes.cls is not None:
                        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
                    counts = {}
                    if cls is not None and len(cls) > 0:
                        for cid in cls:
                            name = names[cid] if isinstance(names, (list, tuple)) else names.get(int(cid), str(cid))
                            counts[name] = counts.get(name, 0) + 1
                    self.detect_counts = counts
                    count_box = counts.get("box", 0)
                    # vẫn trong _on_frame, sau khi tính count_box
                    if count_box is not None and hasattr(self.ui, "packageAvailable"):
                        w = self.ui.packageAvailable
                        text = str(count_box)
                        if hasattr(w, "setPlainText"):
                            w.setPlainText(text)
                        elif hasattr(w, "setText"):
                            w.setText(text)

                    # vẽ ảnh đã detect
                    if hasattr(r, "plot"):
                        vis_bgr = r.plot()  # BGR uint8

            except Exception as e:
                self.log(f"YOLO infer error: {e}")
                vis_bgr = bgr

        # Convert BGR -> RGB -> QImage -> QPixmap và show
        rgb = vis_bgr[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.ui.imgOut.width(),
                                            self.ui.imgOut.height(),
                                            Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
        self.ui.imgOut.setPixmap(pix)

    # ===== xArm =====
    def connect_hardware(self):
        ip = "192.168.1.165"   # đổi nếu cần
        self.log(f"Bắt đầu kết nối xArm {ip}")
        self.xarm_worker = XArmWorker(ip)
        self.xarm_worker.log_sig.connect(self.log)
        self.xarm_worker.ready.connect(self._on_arm_ready)
        self.xarm_worker.start()

    def _on_arm_ready(self, arm_obj):
        self._arm = arm_obj
        self.robot_ctrl.arm = self._arm
        self.robot_mon.arm = self._arm
        self.log("xArm sẵn sàng.")
        self.choose_and_load_yolo()

    def go_home(self):
        self.robot_ctrl.go_home()

    def _get_float_from_plain(self, widget_name: str):
        """
        Đọc số thực từ PlainTextEdit/Label có tên widget_name trong self.ui.
        Hỗ trợ cả dạng '123.45' hoặc 'X = 123.45'.
        Trả về float hoặc None nếu không lấy được.
        """
        if not hasattr(self.ui, widget_name):
            return None
        w = getattr(self.ui, widget_name)
        text = ""
        # Thử PlainTextEdit / TextEdit
        try:
            if hasattr(w, "toPlainText"):
                text = w.toPlainText().strip()
            elif hasattr(w, "text"):
                text = w.text().strip()
            else:
                return None
        except Exception:
            return None

        if not text:
            return None
        # Cắt dạng 'X = 123.45'
        if "=" in text:
            text = text.split("=")[-1].strip()
        try:
            return float(text)
        except Exception:
            return None

    def _on_robot_pose(self, x, y, z, yaw):
        # Nếu UI có các ô hiển thị toạ độ thì cập nhật, nếu không thì log
        updated = False
        self._current_yaw = float(yaw) 
        if hasattr(self.ui, "xarm_X"):
            self.ui.xarm_X.setPlainText(f"{x:.2f}"); updated = True
        if hasattr(self.ui, "xarm_Y"):
            self.ui.xarm_Y.setPlainText(f"{y:.2f}"); updated = True
        if hasattr(self.ui, "xarm_Z"):
            self.ui.xarm_Z.setPlainText(f"{z:.2f}"); updated = True
        if hasattr(self.ui, "xarm_theta"):
            self.ui.xarm_theta.setPlainText(f"{yaw:.2f}"); updated = True
        if not updated:
            self.log(f"Pose: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")

    def move_to_pallet(self):
        # Đọc X/Y/Z từ self.ui.palletX, palletY, palletZ
        x = self._get_float_from_plain("palletX")
        y = self._get_float_from_plain("palletY")
        z = self._get_float_from_plain("palletZ")

        if x is None or y is None or z is None:
            self.log("Lỗi: Thiếu hoặc sai định dạng toạ độ pallet (cần palletX, palletY, palletZ). Về Home.")
            self.go_home()
            return

        if self._arm is None:
            self.log("Chưa kết nối xArm. Vui lòng bấm Init trước.")
            return

        # Dùng yaw hiện tại nếu có, mặc định 0.0
        yaw = getattr(self, "_current_yaw", 0.0)
        self.log(f"Đi tới pallet: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
        self.robot_ctrl.move(x, y, z, yaw)

    def clear_error(self):
        if self._arm is None:
            self.log("Chưa kết nối xArm.")
            return
        try:
            with self.arm_lock:
                self._arm.clean_error()
                self._arm.clean_warn()
                self._arm.set_state(0)  # đưa về trạng thái sẵn sàng
            self.log("Đã gửi lệnh Clear Error cho xArm, robot sẵn sàng.")
        except Exception as e:
            self.log(f"Lỗi khi clear error: {e}")

    def choose_and_load_yolo(self):
        """Mở hộp thoại chọn .pt và load YOLOv8. Trả về True nếu OK."""
        try:
            from PySide2.QtWidgets import QFileDialog
            p, _ = QFileDialog.getOpenFileName(self, "Chọn model YOLOv8 (.pt)",
                                            "", "YOLOv8 weights (*.pt)")
            if not p:
                self.log("Bỏ qua: chưa chọn model YOLO.")
                return False
            try:
                from ultralytics import YOLO
            except Exception as e:
                self.log(f"Chưa cài ultralytics: {e}. pip install ultralytics"); return False

            t0 = time.time()
            self.ui.yoloPath.setPlainText(p)
            self.yolo_model = YOLO(p)
            # Lấy tên lớp nếu có
            try:
                self.yolo_names = self.yolo_model.names
            except Exception:
                self.yolo_names = None
            self.log(f"Đã load YOLOv8: {os.path.basename(p)} ({time.time()-t0:.2f}s)")
            self.move_to_pallet()
            return True
        except Exception as e:
            self.log(f"Lỗi load YOLO: {e}")
            return False

    # ===== lifecycle =====
    def closeEvent(self, event):
        self.stop_event.set()
        try: self.stop_camera()
        except Exception: pass
        try:
            if self.robot_ctrl.isRunning(): self.robot_ctrl.stop()
        except Exception: pass
        try:
            if self.robot_mon.isRunning(): self.robot_mon.stop()
        except Exception: pass
        try:
            if self._arm:
                self._arm.set_state(4)
                self._arm.disconnect()
        except Exception: pass
        super().closeEvent(event)

# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainApp()
    w.show()
    sys.exit(app.exec_())
