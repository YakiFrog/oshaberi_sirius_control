import sys
from PySide6.QtWidgets import QApplication
from ui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    try:
        exit_code = app.exec()
    finally:
        # アプリ終了時にウェイクワード検出を停止
        if hasattr(window, 'wake_controller'):
            window.wake_controller.detector.stop_detection()
    
    sys.exit(exit_code)