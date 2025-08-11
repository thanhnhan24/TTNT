import time, threading

try:
    from xarm.wrapper import XArmAPI
except Exception:
    XArmAPI = None

DEFAULT_SPEED = 39  # mm/s – fallback nếu GUI không gửi speed

def robot_main(gui_to_robot, robot_to_gui, ip="192.168.1.165"):
    """
    Process Robot: kết nối xArm, nhận lệnh move/home/clear_error, gửi pose & log về GUI.
    """
    def send(msg: dict):
        try:
            robot_to_gui.put_nowait(msg)
        except Exception:
            pass

    send({"type": "robot_log", "msg": "Robot process started."})

    arm = None
    lock = threading.Lock()
    running = True
    curr_speed = DEFAULT_SPEED

    if XArmAPI is None:
        send({"type": "robot_log", "msg": "Thiếu xArm SDK. pip install xArm-Python-SDK"})
    else:
        try:
            send({"type": "robot_log", "msg": f"Kết nối xArm tại {ip}..."})
            arm = XArmAPI(ip, is_radian=False)
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            arm.clean_error()
            time.sleep(0.6)
            try:
                arm.move_gohome(wait=True)
            except Exception:
                pass
            send({"type": "robot_log", "msg": "xArm sẵn sàng."})
        except Exception as e:
            send({"type": "robot_log", "msg": f"Lỗi kết nối: {e}"})
            arm = None

    # Thread monitor pose định kỳ
    def monitor():
        while running and arm is not None:
            try:
                with lock:
                    ret = arm.get_position(is_radian=False)
                if ret and ret[0] == 0:
                    x, y, z, r, p, yaw = ret[1]
                    send({"type": "robot_pose", "x": x, "y": y, "z": z, "yaw": yaw})
            except Exception as e:
                send({"type": "robot_log", "msg": f"Monitor err: {e}"})
            time.sleep(0.5)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    # Vòng lặp nhận lệnh
    while running:
        cmd = gui_to_robot.get()  # blocking
        ctype = cmd.get("type")

        if ctype == "shutdown":
            running = False
            break

        if ctype == "set_speed":
            # Cho phép GUI gửi speed xuống process robot (mm/s)
            try:
                val = float(cmd.get("speed", DEFAULT_SPEED))
                curr_speed = max(1.0, val)
                send({"type": "robot_log", "msg": f"Set speed = {curr_speed:.1f} mm/s"})
            except Exception:
                pass
            continue

        if arm is None:
            send({"type": "robot_log", "msg": "Chưa có kết nối xArm."})
            continue

        try:
            if ctype == "move":
                p = cmd.get("params", {})
                x = p.get("x"); y = p.get("y"); z = p.get("z")
                yaw = p.get("yaw", 0.0)
                with lock:
                    code = arm.set_position(x=x, y=y, z=z, yaw=yaw, speed=curr_speed, wait=True)
                if code != 0:
                    send({"type": "robot_log", "msg": f"set_position lỗi code={code}, speed={curr_speed}"})
            elif ctype == "home":
                with lock:
                    try:
                        arm.move_gohome(speed=curr_speed, wait=True)
                    except TypeError:
                        arm.move_gohome(wait=True)
            elif ctype == "clear_error":
                with lock:
                    arm.clean_error()
                    arm.clean_warn()
                    arm.set_state(0)
                send({"type": "robot_log", "msg": "Đã clear error & set_state(0)."})
            elif ctype == "move_delta":   # NEW: dịch tương đối
                dx = float(cmd.get("dx", 0.0))
                dy = float(cmd.get("dy", 0.0))
                z  = cmd.get("z", None)           # nếu None: giữ Z hiện tại
                yaw = cmd.get("yaw", None)        # nếu None: giữ yaw hiện tại

                with lock:
                    ret = arm.get_position(is_radian=False)
                if not ret or ret[0] != 0:
                    send({"type":"robot_log","msg":"Không đọc được pose hiện tại."})
                else:
                    x0, y0, z0, r0, p0, yaw0 = ret[1]
                    tgt_x = x0 + dx
                    tgt_y = y0 + dy
                    tgt_z = z0 if z is None else float(z)
                    tgt_yaw = yaw0 if yaw is None else float(yaw)
                    with lock:
                        code = arm.set_position(x=tgt_x, y=tgt_y, z=tgt_z,
                                                yaw=tgt_yaw, speed=curr_speed, wait=True)
                    if code != 0:
                        send({"type":"robot_log","msg":f"move_delta lỗi code={code}"})
            elif ctype == "calib":
                # Placeholder: nếu muốn chạy visual-servo dưới Robot process thì cài sau
                send({"type": "robot_log", "msg": "Calib (placeholder) – thuật toán ở GUI/Vision."})
        except Exception as e:
            send({"type": "robot_log", "msg": f"Lỗi lệnh '{ctype}': {e}"})

    # cleanup
    try:
        if arm is not None:
            arm.set_state(4)
            arm.disconnect()
    except Exception:
        pass
    send({"type": "robot_log", "msg": "Robot process stopped."})
