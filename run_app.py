import sys
from multiprocessing import set_start_method, Process
from PySide2.QtWidgets import QApplication

from common_ipc import make_queues
from proc_vision import vision_main
from proc_robot import robot_main
from app_control import MainApp
if __name__ == "__main__":
    try:
        set_start_method("spawn")  # cần cho Windows
    except RuntimeError:
        pass

    q = make_queues()

    # Khởi động Vision và Robot ở process riêng
    p_vis = Process(target=vision_main, args=(q["gui_to_vision"], q["vision_to_gui"]), daemon=True)
    p_bot = Process(target=robot_main,  args=(q["gui_to_robot"],  q["robot_to_gui"]),  daemon=True)
    p_vis.start()
    p_bot.start()

    app = QApplication(sys.argv)
    w = MainApp(q)
    w.show()
    code = app.exec_()

    # shutdown các process
    try: q["gui_to_vision"].put({"type": "shutdown"})
    except: pass
    try: q["gui_to_robot"].put({"type": "shutdown"})
    except: pass
    p_vis.join(timeout=2)
    p_bot.join(timeout=2)
    sys.exit(code)
