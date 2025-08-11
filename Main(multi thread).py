import UI
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QThread, Signal, Qt, QTimer
from PySide2.QtGui import QImage, QPixmap
import sys, os, queue, threading, time, cv2, math
import pyrealsense2 as rs
import numpy as np
from xarm.wrapper import XArmAPI
from cv2 import aruco

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# =========================
# Camera thread (RealSense)
# =========================

class CameraThread(QThread):
    # Truyền ndarray BGR ra ngoài để main xử lý/YOLO
    frame_signal = Signal(object)        # ndarray BGR
    intrinsics_signal = Signal(object)   # {"K": K, "dist": dist}
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

            # Emit intrinsics một lần
            try:
                cprof = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                intr = cprof.get_intrinsics()
                K = np.array([[intr.fx, 0, intr.ppx],
                            [0, intr.fy, intr.ppy],
                            [0,       0,        1]], dtype=np.float32)
                dist = np.array(intr.coeffs, dtype=np.float32)
                self.intrinsics_signal.emit({"K": K, "dist": dist})
            except Exception as e:
                self._log(f"[CameraThread] intrinsics error: {e}")

            last_ok = time.time()
            while not self.isInterruptionRequested():
                # Không block 5s – poll nhanh 10–15ms
                frames = None
                for _ in range(5):  # ~50–75ms
                    fs = self._pipeline.poll_for_frames()
                    if fs:
                        frames = fs
                        break
                    self.msleep(15)

                if frames:
                    color = frames.get_color_frame()
                    if color:
                        bgr = np.asanyarray(color.get_data())
                        self.frame_signal.emit(bgr)
                        last_ok = time.time()
                    continue

                # Không có frame quá 2 giây -> restart pipeline
                if time.time() - last_ok > 2.0:
                    self._log("[CameraThread] no frames for 2s, restarting pipeline…")
                    try:
                        self._pipeline.stop()
                    except Exception:
                        pass
                    self._pipeline = rs.pipeline()
                    self._pipeline.start(cfg)
                    last_ok = time.time()

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
# ArUco detection thread
# =========================
class ArucoDetectThread(QThread):
    # dx, dy, cx, cy, t (timestamp), ids (numpy or None)
    result_sig = Signal(float, float, float, float, float, object)

    def __init__(self, img_w=640, img_h=480, parent=None):
        super().__init__(parent)
        self.img_w = img_w
        self.img_h = img_h
        self.cx0 = img_w * 0.5
        self.cy0 = img_h * 0.5
        self.q = queue.Queue(maxsize=1)
        self._running = True

        # OpenCV ArUco detector (4x4_50)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def submit(self, bgr):
        # Không block UI: nếu queue đầy thì thay frame cũ bằng frame mới
        try:
            if self.q.full():
                _ = self.q.get_nowait()
            self.q.put_nowait(bgr)
        except queue.Full:
            pass
        except queue.Empty:
            pass

    def run(self):
        while self._running:
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                corners, ids, _ = self.detector.detectMarkers(frame)
                if ids is None or len(ids) == 0:
                    continue

                # Chọn marker gần tâm ảnh nhất
                best_i = None
                best_d2 = 1e18
                best_c = None
                for i in range(len(ids)):
                    pts = corners[i][0]
                    cx = float(pts[:, 0].mean())
                    cy = float(pts[:, 1].mean())
                    d2 = (cx - self.cx0) ** 2 + (cy - self.cy0) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_i = i
                        best_c = (cx, cy)

                if best_i is None or best_c is None:
                    continue

                cx, cy = best_c
                dx = cx - self.cx0
                dy = cy - self.cy0
                self.result_sig.emit(float(dx), float(dy), float(cx), float(cy), time.time(), ids)
            except Exception:
                # nuốt lỗi để thread không chết
                pass

    def stop(self):
        self._running = False
        self.wait()

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
        self.camera_thread = None
        self._last_aruco = None
                # --- ArUco thread ---
        self.aruco_thread = ArucoDetectThread(img_w=640, img_h=480, parent=self)
        self.aruco_thread.result_sig.connect(self._on_aruco_result)
        self.aruco_thread.start()
        self._aruco_last = None  # lưu (dx, dy, cx, cy, t)



        # Kết nối xArm
        self.connect_hardware() 

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
        if hasattr(self.ui, "checkPickupPos"):
            self.ui.checkPickupPos.clicked.connect(self.check_pickup_pos)
        if hasattr(self.ui, "calibButton"):
            self.ui.calibButton.clicked.connect(self.calib_aruco_center)



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
        self.camera_thread.intrinsics_signal.connect(self._on_intrinsics)
        self.camera_thread.start()
        self.log("Camera started.")


    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.requestInterruption()
            self.camera_thread.wait()
            self.log("Camera stopped.")

    def _on_frame(self, bgr):
        yaw_offsets = 90
        try:
            vis_bgr = bgr.copy()

            # ===== Nếu đã load YOLO thì detect trước =====
            if hasattr(self, "yolo_model") and self.yolo_model is not None:
                results = self.yolo_model.predict(vis_bgr, iou=0.5, conf=0.25, verbose=False)
                counts = {}
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        cls_name = self.yolo_model.names[cls_id]
                        counts[cls_name] = counts.get(cls_name, 0) + 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_bgr, f"{cls_name} {box.conf[0]:.2f}",
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)

                # Cập nhật số lượng vào packageAvailable
                pkg_text = "\n".join(f"{k}: {v}" for k, v in counts.items())
                self._set_text_safe("packageAvailable", pkg_text)

            # ===== Phát hiện ArUco 4x4_50 =====
            try:
                dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(dictionary, parameters)

                corners, ids, rejected = detector.detectMarkers(bgr)
                if ids is not None and len(ids) > 0:
                    idx = np.random.randint(len(ids))  # chọn ngẫu nhiên 1 marker
                    cv2.aruco.drawDetectedMarkers(vis_bgr, corners, ids)

                    # Góc 2D trên ảnh
                    pts = corners[idx][0]
                    cx = float(pts[:,0].mean()); cy = float(pts[:,1].mean())
                    self._last_aruco = {
                        "u": cx, "v": cy,
                        "img_wh": (bgr.shape[1], bgr.shape[0]),
                        "t": time.time()
                    }
                    tl, tr = pts[0], pts[1]
                    v = tr - tl
                    yaw_img_deg = float(math.degrees(math.atan2(v[1], v[0]))) + yaw_offsets 
                    self._set_text_safe("pickUpYaw", f"{yaw_img_deg:.2f}")

                    if hasattr(self, "cam_K") and hasattr(self, "cam_dist") and hasattr(self, "aruco_size_m"):
                        if self.cam_K is not None and self.aruco_size_m is not None:
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                corners, self.aruco_size_m, self.cam_K, self.cam_dist
                            )
                            rvec, tvec = rvecs[idx].reshape(3, 1), tvecs[idx].reshape(3, 1)

                            # cv2.aruco.drawAxis(vis_bgr, self.cam_K, self.cam_dist, rvec, tvec, self.aruco_size_m * 0.5)

                            X_mm = float(tvec[0] * 1000.0)
                            Y_mm = float(tvec[1] * 1000.0)
                            Z_mm = float(tvec[2] * 1000.0)

                            # Sau khi lấy X_mm, Y_mm, Z_mm từ tvec
                            robot_X = getattr(self, "_current_x", 0.0)
                            robot_Y = getattr(self, "_current_y", 0.0)
                            cam_offset_x = self._get_float_from_plain("cameraXoffsets") or 0.0
                            cam_offset_y = self._get_float_from_plain("cameraYoffsets") or 0.0

                            pickUpX_val = robot_X - X_mm + cam_offset_x
                            pickUpY_val = robot_Y + Y_mm + cam_offset_y

                            self._set_text_safe("pickUpX", f"{pickUpX_val:.1f}")
                            self._set_text_safe("pickUpY", f"{pickUpY_val:.1f}")
                            # self._set_text_safe("pickUpZ", f"{Z_mm:.1f}")

                    else:
                        # Nếu chưa có intrinsics thì fill toạ độ ảnh
                        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                        self._set_text_safe("pickUpX", f"{cx:.1f}")
                        self._set_text_safe("pickUpY", f"{cy:.1f}")
                        self._set_text_safe("pickUpZ", "")
            except Exception as e:
                self.log(f"ArUco error: {e}")

            # ===== Đưa ảnh ra imgOut =====
            rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            target_w = self.ui.imgOut.width()
            target_h = self.ui.imgOut.height()
            self.ui.imgOut.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            self.log(f"_on_frame error: {e}")


            # ===== YOLO (giữ như bạn đang có) =====
            if self.yolo_model is not None:
                try:
                    rgb = vis_bgr[:, :, ::-1]
                    results = self.yolo_model.predict(rgb, verbose=False)
                    if results and hasattr(results[0], "plot"):
                        vis_bgr = results[0].plot()
                except Exception as e:
                    self.log(f"YOLO infer error: {e}")
                    vis_bgr = vis_bgr

        # ===== Show lên imgOut =====
        rgb = vis_bgr[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.ui.imgOut.width(),
                                            self.ui.imgOut.height(),
                                            Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
        self.ui.imgOut.setPixmap(pix)

    def _set_text_safe(self, widget_name, text):
        if not hasattr(self.ui, widget_name): return
        w = getattr(self.ui, widget_name)
        if hasattr(w, "setPlainText"): w.setPlainText(str(text))
        elif hasattr(w, "setText"):    w.setText(str(text))

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
        self._current_x = float(x)
        self._current_y = float(y)

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
        yaw = 0
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

    def _on_intrinsics(self, data: dict):
        self.cam_K = data.get("K")
        self.cam_dist = data.get("dist")
        self.log(f"Got intrinsics: fx={self.cam_K[0,0]:.1f}, fy={self.cam_K[1,1]:.1f}")

    def check_pickup_pos(self):
        # Lấy X, Y, Yaw từ các ô đã điền (ArUco tính được trước đó)
        x   = self._get_float_from_plain("pickUpX")
        y   = self._get_float_from_plain("pickUpY")
        yaw = self._get_float_from_plain("pickUpYaw")

        if x is None or y is None:
            self.log("Thiếu pickUpX/pickUpY. Kiểm tra lại ArUco hoặc các ô nhập.")
            return
        if yaw is None:
            yaw = getattr(self, "_current_yaw", 0.0)  # fallback: yaw hiện tại của xArm

        z = 175.0  # theo yêu cầu cố định Z

        # Log cho rõ + hiển thị vào cụm 'Tọa độ gắp' nếu có
        self.log(f"Đi tới tọa độ gắp: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}, speed={getattr(self.ui,'speed',None).value() if hasattr(self.ui,'speed') else 'UI'}")
        for name, val in (("grabX", x), ("grabY", y), ("grabZ", z), ("grabYaw", yaw)):
            if hasattr(self.ui, name):
                w = getattr(self.ui, name)
                if hasattr(w, "setPlainText"): w.setPlainText(f"{val:.2f}")
                elif hasattr(w, "setText"):    w.setText(f"{val:.2f}")

        # Gửi lệnh cho thread robot (thread sẽ tự lấy speed từ self.ui.speed)
        self.robot_ctrl.move(x, y, z, yaw)

    def calib_aruco_center(self):
        """
        Đưa tâm ArUco về trùng tâm ảnh bằng visual-servo trên X,Y.
        Dựa trên intrinsics (fx, fy) và cao độ hover_z.
        """
        if self._arm is None:
            self.log("Chưa kết nối xArm."); return
        if self._last_aruco is None or time.time() - self._last_aruco.get("t", 0) > 1.0:
            self.log("Không thấy ArUco gần đây. Đưa tag vào khung hình."); return

        # --- tham số điều khiển ---
        hover_z   = 320.0      # cao độ căn XY (mm) – chỉnh theo thực tế
        px_th     = 3.0        # ngưỡng lỗi ảnh (px) để coi là trùng
        kP        = 0.7        # hệ số tỉ lệ (giảm nếu rung)
        max_step  = 15.0       # mm: giới hạn mỗi bước
        max_iter  = 15
        # nếu thấy đi ngược, đổi dấu ở đây:
        sign_x, sign_y = +1.0, +1.0

        # pose hiện tại
        with self.arm_lock:
            ret = self._arm.get_position(is_radian=False)
        if not ret or ret[0] != 0:
            self.log("Không đọc được pose xArm."); return
        cur_x, cur_y, cur_z, _, _, cur_yaw = ret[1]

        # lên/ xuống độ cao hover_z
        if abs(cur_z - hover_z) > 1.0:
            self.robot_ctrl.move(cur_x, cur_y, hover_z, cur_yaw)

        for it in range(max_iter):
            data = self._last_aruco
            if data is None or time.time() - data.get("t", 0) > 1.0:
                self.log("Mất ArUco trong lúc căn."); break

            u, v = float(data["u"]), float(data["v"])
            img_w, img_h = data.get("img_wh", (640, 480))
            u0, v0 = img_w * 0.5, img_h * 0.5

            ex = u - u0   # + nếu tag ở bên phải tâm ảnh
            ey = v - v0   # + nếu tag ở bên dưới tâm ảnh

            if abs(ex) <= px_th and abs(ey) <= px_th:
                self.log(f"Calib XY xong trong {it} bước (|e|≤{px_th}px).")
                break

            # map px -> mm tại cao độ hover_z (mô hình pinhole)
            if self.cam_K is not None:
                fx = float(self.cam_K[0,0]); fy = float(self.cam_K[1,1])
                sx_m_per_px = (hover_z/1000.0) / max(fx, 1e-6)  # m/px
                sy_m_per_px = (hover_z/1000.0) / max(fy, 1e-6)
                dx_mm = -ex * sx_m_per_px * 1000.0 * kP
                dy_mm = -ey * sy_m_per_px * 1000.0 * kP
            else:
                # fallback nếu chưa có intrinsics: tỉ lệ gần đúng
                scale = 0.2  # mm/px – chỉnh thực nghiệm
                dx_mm = -ex * scale * kP
                dy_mm = -ey * scale * kP

            # giới hạn bước & đổi dấu theo hướng trục robot
            dx_mm = float(np.clip(dx_mm, -max_step, max_step)) * sign_x
            dy_mm = float(np.clip(dy_mm, -max_step, max_step)) * sign_y

            target_x = cur_x - dx_mm
            target_y = cur_y - dy_mm

            self.log(f"[Calib] it={it} err=({ex:.1f}px,{ey:.1f}px) step=({dx_mm:.1f},{dy_mm:.1f}) → ({target_x:.1f},{target_y:.1f})")
            self.robot_ctrl.move(target_x, target_y, hover_z, cur_yaw)

            # cập nhật pose hiện tại sau khi move
            with self.arm_lock:
                ret = self._arm.get_position(is_radian=False)
            if not ret or ret[0] != 0: break
            cur_x, cur_y, cur_z, _, _, cur_yaw = ret[1]

        # (tuỳ chọn) sau khi căn xong thì hạ xuống Z=175
        # self.robot_ctrl.move(cur_x, cur_y, 175.0, cur_yaw)

    def _on_aruco_result(self, dx, dy, cx, cy, t, ids):
        # Lưu để overlay trong _on_frame
        self._aruco_last = {"dx": dx, "dy": dy, "cx": cx, "cy": cy, "t": t}

        # Nếu UI có ô hiển thị thì cập nhật; nếu không thì log
        updated = False
        if hasattr(self.ui, "arucoDX"):
            self.ui.arucoDX.setPlainText(f"{dx:.1f} px"); updated = True
        if hasattr(self.ui, "arucoDY"):
            self.ui.arucoDY.setPlainText(f"{dy:.1f} px"); updated = True
        if not updated:
            self.log(f"ArUco offset: dx={dx:.1f}px, dy={dy:.1f}px (cx={cx:.1f}, cy={cy:.1f})")


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
