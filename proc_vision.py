import time, cv2, numpy as np

# Các import có thể không sẵn trên máy build CI – giữ try/except cho an toàn
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

ARUCO_SIZE_M = 0.035  # cạnh tag (m) – chỉnh theo thước in của bạn

def _init_camera(width=640, height=480, fps=30):
    if rs is None:
        raise RuntimeError("Thiếu pyrealsense2. Hãy 'pip install pyrealsense2'.")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipe.start(cfg)

    # Lấy intrinsics của stream màu
    cprof = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intr = cprof.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0,       0,       1]], dtype=np.float32)
    dist = np.array(intr.coeffs, dtype=np.float32)
    return pipe, K, dist

def _init_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector

def vision_main(gui_to_vision, vision_to_gui):
    """
    Process Vision: RealSense -> (optional) YOLO -> ArUco -> gửi JPEG + counts + pose ArUco về GUI
    """
    pipe, K, dist = None, None, None
    detector = _init_aruco()
    model = None
    model_names = None

    running = True
    last_frame_sent = 0.0

    def send(msg: dict):
        try:
            vision_to_gui.put_nowait(msg)
        except Exception:
            pass

    send({"type": "vision_log", "msg": "Vision process started."})

    while running:
        # Nhận lệnh từ GUI (non-blocking)
        try:
            while True:
                cmd = gui_to_vision.get_nowait()
                ctype = cmd.get("type")
                if ctype == "shutdown":
                    running = False
                elif ctype == "camera_open":
                    pipe, K, dist = _init_camera()
                    send({"type": "vision_log", "msg": "Camera opened (RealSense)."})
                    # NEW: emit intrinsics (fx, fy) để GUI tính mm/px
                    send({"type": "cam_intr", "fx": float(K[0,0]), "fy": float(K[1,1]), "w": 640, "h": 480})
                elif ctype == "camera_close":
                    if pipe is not None:
                        try: pipe.stop()
                        except Exception: pass
                        pipe = None
                        send({"type": "vision_log", "msg": "Camera closed."})
                elif ctype == "load_model":
                    path = cmd.get("path")
                    if YOLO is None:
                        send({"type": "vision_log", "msg": "Thiếu ultralytics. pip install ultralytics"})
                    else:
                        try:
                            t0 = time.time()
                            model = YOLO(path)
                            model_names = model.names
                            send({"type": "vision_log", "msg": f"Loaded YOLO: {path} ({time.time()-t0:.2f}s)"})
                        except Exception as e:
                            send({"type": "vision_log", "msg": f"YOLO load error: {e}"})
        except Exception:
            pass

        # Nếu chưa mở camera thì chờ
        if pipe is None:
            time.sleep(0.01)
            continue

        # Lấy frame non-blocking
        fs = pipe.poll_for_frames()
        if not fs:
            time.sleep(0.005)
            continue
        color = fs.get_color_frame()
        if not color:
            continue

        bgr = np.asanyarray(color.get_data())
        vis = bgr.copy()

        # YOLO (nếu có)
        if model is not None:
            try:
                res = model.predict(vis, iou=0.5, conf=0.25, verbose=False)
                counts = {}
                for r in res:
                    for box in r.boxes:
                        cls_name = model_names[int(box.cls)]
                        counts[cls_name] = counts.get(cls_name, 0) + 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if counts:
                    send({"type": "vision_status", "counts": counts})
            except Exception as e:
                send({"type": "vision_log", "msg": f"YOLO error: {e}"})

        # ArUco detect + pose
        try:
            corners, ids, _ = detector.detectMarkers(bgr)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                # lấy marker gần tâm ảnh
                cx0, cy0 = 320, 240
                best_i, best_d = -1, 1e18
                for i in range(len(ids)):
                    pts = corners[i][0]
                    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                    d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
                    if d2 < best_d: best_d, best_i = d2, i

                pts = corners[best_i][0]
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                # yaw ảnh 2D (độ)
                yaw_img = float(np.degrees(np.arctan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])))

                # Pose 3D (nếu có K, dist)
                if K is not None and ARUCO_SIZE_M > 0:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_SIZE_M, K, dist)
                    rvec = rvecs[best_i].reshape(3)
                    tvec = tvecs[best_i].reshape(3)  # mét
                    # Vẽ trục để debug (tùy chọn)
                    # cv2.aruco.drawAxis(vis, K, dist, rvecs[best_i], tvecs[best_i], ARUCO_SIZE_M * 0.5)

                    dx, dy = cx - cx0, cy - cy0

                    send({
                        "type": "aruco_pose",
                        "yaw_img": yaw_img,
                        "tvec": tvec.tolist() if (K is not None) else [0.0, 0.0, 0.0],
                        "rvec": rvec.tolist() if (K is not None) else [0.0, 0.0, 0.0],
                        "cx": cx, "cy": cy, "dx": dx, "dy": dy,
                        "t": time.time()
                    })
                else:
                    send({"type": "aruco_pose",
                          "yaw_img": yaw_img,
                          "tvec": [0.0, 0.0, 0.0],
                          "rvec": [0.0, 0.0, 0.0],
                          "cx": cx, "cy": cy, "t": time.time()})

                # Overlay tâm ảnh và đường nối
                cv2.drawMarker(vis, (cx0, cy0), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), 2)
                cv2.line(vis, (cx0, cy0), (int(cx), int(cy)), (255, 0, 0), 2)
                dx, dy = cx - cx0, cy - cy0
                cv2.putText(vis, f"dx={dx:.1f} dy={dy:.1f} yaw={yaw_img:.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            send({"type": "vision_log", "msg": f"ArUco error: {e}"})

        # Gửi frame JPEG (≈15fps)
        now = time.time()
        if now - last_frame_sent > (1 / 15):
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                send({"type": "frame", "jpeg": buf.tobytes()})
            last_frame_sent = now

    # cleanup
    try:
        if pipe is not None: pipe.stop()
    except Exception:
        pass
    send({"type": "vision_log", "msg": "Vision process stopped."})
