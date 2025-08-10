import UI
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide2.QtGui import QImage, QPixmap
import sys, os, queue, threading, time
import pyrealsense2 as rs
import numpy as np
from xarm.wrapper import XArmAPI

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
arm = None  # biến robot chung

# =========================
# Robot control thread
# =========================
class RobotControlThread(QThread):
    """
    Thread điều khiển robot bằng hàng đợi lệnh.
    Hỗ trợ: move, home, gripper(open/close), sequence.
    """
    def __init__(self, arm, lock, app=None):
        super().__init__()
        self.arm = arm
        self.lock = lock
        self.app = app  # để đọc app.stop_event & log
        self.command_queue = queue.Queue()
        self.running = True
        self.speed = None  # speed mặc định (nhận từ UI)

    # ---------- tiện ích log ----------
    def _log(self, msg: str):
        if self.app and hasattr(self.app, "log"):
            self.app.log(msg)
        else:
            # fallback nếu chưa có app
            print(msg)

    # ---------- nhận speed từ UI (queued connection an toàn) ----------
    @Slot(int)
    def on_speed_changed(self, value: int):
        self.speed = int(value)
        self._log(f"[Robot] speed={self.speed}")

    # ---------- API đẩy lệnh vào queue ----------
    def set_position(self, x, y, z, yaw, speed=None, acc=None):
        params = {"x": x, "y": y, "z": z, "yaw": yaw}
        if speed is not None:
            params["speed"] = speed
        if acc is not None:
            params["acc"] = acc
        self.command_queue.put({"type": "move", "params": params})

    def go_home(self, speed=None):
        self.command_queue.put({"type": "home", "speed": speed})

    def open_gripper(self):
        self.command_queue.put({"type": "gripper", "action": "open"})

    def close_gripper(self):
        self.command_queue.put({"type": "gripper", "action": "close"})

    def run_sequence(self, steps):
        self.command_queue.put({"type": "sequence", "steps": steps})

    def stop(self):
        self.running = False
        self.command_queue.put({"type": "stop"})
        self.wait()

    # ---------- Vòng lặp chính ----------
    def run(self):
        def should_stop():
            if not self.running:
                return True
            if self.app is not None and hasattr(self.app, "stop_event"):
                return self.app.stop_event.is_set()
            return False

        while not should_stop():
            try:
                command = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not isinstance(command, dict):
                continue

            ctype = command.get("type", "")
            self._log(f"[RobotThread] recv: {ctype}")

            if ctype == "stop":
                break

            if ctype == "move":
                params = command.get("params", {})
                self._do_move(**params)

            elif ctype == "home":
                self._do_move_home(command.get("speed", None))

            elif ctype == "gripper":
                action = command.get("action", "")
                if action == "open":
                    self._do_gripper_open()
                elif action == "close":
                    self._do_gripper_close()

            elif ctype == "sequence":
                steps = command.get("steps", [])
                for step in steps:
                    if should_stop():
                        break
                    if "move" in step:
                        self._do_move(**step["move"])
                    g = step.get("gripper")
                    if g == "open":
                        self._do_gripper_open()
                    elif g == "close":
                        self._do_gripper_close()
                    time.sleep(step.get("delay", 0.5))

        self.running = False

    # ---------- tiện ích: đảm bảo READY ----------
    def _ensure_ready(self, retry=True):
        if self.arm is None:
            return False
        try:
            code, state = self.arm.get_state()
            if code == 0 and state == 0:
                return True
            if retry:
                self.arm.clean_error()
                self.arm.motion_enable(True)
                self.arm.set_mode(0)
                self.arm.set_state(0)
                time.sleep(0.25)
                code, state = self.arm.get_state()
                return code == 0 and state == 0
        except Exception as e:
            self._log(f"[Robot] ensure_ready error: {e}")
        return False

    # ---------- Thao tác phần cứng ----------
    def _resolve_speed(self, speed):
        s = speed if speed is not None else (self.speed if self.speed is not None else 100)
        # ép min speed > 0
        if s is None or s <= 0:
            s = 100
        return int(s)

    def _do_move(self, x, y, z, yaw, speed=None, acc=None):
        try:
            s = self._resolve_speed(speed)
            if self.arm is None:
                self._log(f"[Robot] (mock) Move to x={x:.1f} y={y:.1f} z={z:.1f} yaw={yaw:.1f} speed={s}")
                return
            if not self._ensure_ready(retry=True):
                self._log("[Robot] Bỏ qua move vì xArm chưa READY")
                return
            with self.lock:
                # Lưu ý roll=180, pitch=0 theo setup đầu kẹp của bạn
                self.arm.set_position(x=x, y=y, z=z, roll=180, pitch=0, yaw=yaw,
                                      speed=s, wait=True, is_radian=False)
                self._log(f"[Robot] Move (real) → x={x}, y={y}, z={z}, yaw={yaw}, speed={s}")
        except Exception as e:
            self._log(f"[Robot] Move error: {e}")

    def _do_move_home(self, speed=None):
        try:
            s = self._resolve_speed(speed)
            if self.arm is None:
                self._log(f"[Robot] (mock) Home speed={s}")
                return
            if not self._ensure_ready(retry=True):
                self._log("[Robot] Bỏ qua home vì xArm chưa READY")
                return
            with self.lock:
                # Toạ độ home tuỳ theo bạn định nghĩa
                self.arm.set_position(x=87, y=0, z=154.2, roll=180, pitch=0, yaw=0,
                                      speed=s, wait=True, is_radian=False)
                self._log(f"[Robot] GoHome (real) speed={s}")
        except Exception as e:
            self._log(f"[Robot] Move home error: {e}")

    def _do_gripper_open(self):
        try:
            if self.arm is None:
                self._log("[Robot] (mock) Open gripper")
                return
            if not self._ensure_ready(retry=False):
                self._log("[Robot] Bỏ qua gripper vì xArm chưa READY")
                return
            with self.lock:
                # Lite6 gripper
                self.arm.open_lite6_gripper(1)
                time.sleep(0.4)
                self.arm.stop_lite6_gripper(1)
                self._log("[Robot] Open gripper (real)")
        except Exception as e:
            self._log(f"[Robot] Gripper open error: {e}")

    def _do_gripper_close(self):
        try:
            if self.arm is None:
                self._log("[Robot] (mock) Close gripper")
                return
            if not self._ensure_ready(retry=False):
                self._log("[Robot] Bỏ qua gripper vì xArm chưa READY")
                return
            with self.lock:
                self.arm.close_lite6_gripper(1)
                time.sleep(0.2)
                self._log("[Robot] Close gripper (real)")
        except Exception as e:
            self._log(f"[Robot] Gripper close error: {e}")


# =========================
# Camera thread
# =========================
class CameraThread(QThread):
    frame_signal = Signal(object)  # emits np.ndarray (color, HxWx3, uint8 BGR)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline = None
        # parent chính là MainApp → dùng để log
        self.app = parent

    def _log(self, msg: str):
        if self.app and hasattr(self.app, "log"):
            self.app.log(msg)
        else:
            print(msg)

    def run(self):
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
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
            self._log(f"[CameraThread] error: {e}")
        finally:
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception as e:
                self._log(f"[CameraThread] stop error: {e}")
            self._pipeline = None


# =========================
# Main window
# =========================
class MainApp(QMainWindow, UI.Ui_MainWindow):
    # Signal để log an toàn từ mọi thread
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.ui = UI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("XArm Control")

        # Kết nối signal log → ListView
        self.log_signal.connect(self.print_to_listview)

        # Event dừng an toàn cho thread robot
        self.stop_event = threading.Event()

        # Khởi tạo lock & tham chiếu arm
        self._arm_lock = threading.Lock()
        self._arm = None

        # Khởi tạo robot thread (arm hiện tại = None, sẽ gán sau khi connect)
        self.robot_thread = RobotControlThread(self._arm, self._arm_lock, self)
        self.robot_thread.start()

        # Khởi tạo camera thread
        self.camera_thread = CameraThread(self)
        self.camera_thread.frame_signal.connect(self.on_color_frame)
        self.camera_thread.start()

        self._last_qimage = None
        self._last_pixmap = None

        # Kết nối xArm ngay sau khi UI lên (không block)
        QTimer.singleShot(0, self._connect_xarm_on_start)

        # Handle UI events
        if hasattr(self.ui, "speed"):
            self.ui.speed.valueChanged.connect(self.robot_thread.on_speed_changed)
        if hasattr(self.ui, "zeroPos"):
            # enqueue lệnh home (không gọi _do_*)
            self.ui.zeroPos.clicked.connect(self._on_zero_clicked)

    # Helper để các thread gọi log
    def log(self, msg: str):
        self.log_signal.emit(msg)

    def _on_zero_clicked(self):
        ui_speed = None
        try:
            if hasattr(self.ui, "speed"):
                ui_speed = int(self.ui.speed.value())
        except Exception:
            ui_speed = None
        self.robot_thread.go_home(ui_speed)

    def _connect_xarm_on_start(self):
        """Kết nối xArm bằng IP, gọi khi GUI vừa mở (sau event loop)."""
        try:
            ip = None
            if hasattr(self.ui, "ip_addr"):
                ip = (self.ui.ip_addr.toPlainText().strip()
                      if hasattr(self.ui.ip_addr, "toPlainText")
                      else self.ui.ip_addr.text().strip())
            if not ip:
                ip = "192.168.1.165"  # fallback

            self._arm = XArmAPI(ip)
            self._arm.motion_enable(True)
            self._arm.clean_error()
            self._arm.set_mode(0)   # position mode
            self._arm.set_state(0)  # ready
            time.sleep(0.2)

            # Gán arm vào robot thread
            self.robot_thread.arm = self._arm

            # Cập nhật UI & log
            if hasattr(self.ui, "ip_addr"):
                if hasattr(self.ui.ip_addr, "setPlainText"):
                    self.ui.ip_addr.setPlainText(ip)
                else:
                    self.ui.ip_addr.setText(ip)
            self.log(f"Kết nối xArm OK: {ip}")

            # Auto về home nhẹ nhàng sau khi connect
            QTimer.singleShot(200, lambda: self.robot_thread.go_home())

        except Exception as e:
            self.log(f"Không thể kết nối xArm ngay khi mở GUI: {e}")

    # ===== Hiển thị camera từ RealSense lên QLabel imgOut =====
    def on_color_frame(self, bgr_image: np.ndarray):
        if bgr_image is None or bgr_image.size == 0:
            return
        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1])
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qimg = qimg.copy()
        self._last_qimage = qimg
        pix = QPixmap.fromImage(qimg)
        self._last_pixmap = pix
        if hasattr(self.ui, "imgOut") and self.ui.imgOut is not None:
            target_w = self.ui.imgOut.width()
            target_h = self.ui.imgOut.height()
            pix_scaled = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.imgOut.setPixmap(pix_scaled)

    # ===== Start/Stop camera thread =====
    def start_camera(self):
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.camera_thread = CameraThread(self)
            self.camera_thread.frame_signal.connect(self.on_color_frame)
            self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.requestInterruption()
            self.camera_thread.wait()

    # ===== Log → ListView/ListView =====
    def print_to_listview(self, msg: str):
        from PySide2.QtCore import QStringListModel
        lv = getattr(self.ui, "logView", None)
        if lv is None:
            print(str(msg))
            return
        # Nếu là QListWidget → dùng item-based
        try:
            from PySide2.QtWidgets import QListWidget
            if isinstance(lv, QListWidget):
                lv.addItem(str(msg))
                lv.scrollToBottom()
                return
        except Exception:
            pass
        # Nếu là QListView → dùng model-based
        try:
            from PySide2.QtWidgets import QListView
            if isinstance(lv, QListView):
                if not hasattr(self, 'log_model'):
                    self.log_model = QStringListModel()
                    self.log_data = []
                    lv.setModel(self.log_model)
                self.log_data.append(str(msg))
                # giữ tối đa 1000 dòng
                if len(self.log_data) > 1000:
                    self.log_data = self.log_data[-1000:]
                self.log_model.setStringList(self.log_data)
                try:
                    index = self.log_model.index(len(self.log_data) - 1)
                    lv.scrollTo(index)
                except Exception:
                    pass
                return
        except Exception:
            pass
        # fallback
        print(str(msg))

    # ===== Đóng app =====
    def closeEvent(self, event):
        self.stop_camera()
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "robot_thread") and self.robot_thread.isRunning():
            self.robot_thread.stop()
        # ngắt xArm
        try:
            if self._arm is not None:
                self._arm.set_state(4)
                self._arm.disconnect()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
