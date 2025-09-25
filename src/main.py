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
        # アプリ終了時にウェイクワード検出と音声認識を停止
        if hasattr(window, 'wake_controller') and window.wake_controller.detector:
            window.wake_controller.detector.stop_detection()
        if hasattr(window, 'voice_controller'):
            window.voice_controller.stop_recognition()
    
    sys.exit(exit_code)