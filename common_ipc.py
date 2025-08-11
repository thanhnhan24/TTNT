from multiprocessing import Queue

def make_queues():
    """
    Tạo tất cả các hàng đợi dùng chung cho 3 process.
    Trả về dict chứa 4 queue:
      - gui_to_vision: lệnh từ GUI sang Vision
      - vision_to_gui: dữ liệu từ Vision về GUI
      - gui_to_robot: lệnh từ GUI sang Robot
      - robot_to_gui: dữ liệu từ Robot về GUI
    """
    return {
        "gui_to_vision": Queue(maxsize=10),
        "vision_to_gui": Queue(maxsize=20),
        "gui_to_robot":  Queue(maxsize=50),
        "robot_to_gui":  Queue(maxsize=50),
    }
