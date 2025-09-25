import sys
import os
from PySide6.QtWidgets import QApplication
from ui import MainWindow

def check_display():
    """ディスプレイ環境が利用可能かチェック"""
    # DISPLAY環境変数チェック
    if os.environ.get('DISPLAY') is None:
        print("⚠️  DISPLAY環境変数が設定されていません。")
        return False
    
    # X11接続チェック
    try:
        import subprocess
        result = subprocess.run(['xset', 'q'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

if __name__ == "__main__":
    # GUI環境チェック
    if not check_display():
        print("❌ GUI環境が利用できません。")
        print("💡 以下のいずれかの方法でGUI環境を設定してください:")
        print("   1. export DISPLAY=:0  (ローカルディスプレイ)")
        print("   2. ssh -X username@hostname  (X11フォワーディング)")
        print("   3. VNCや仮想ディスプレイを使用")
        sys.exit(1)
    
    try:
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
        
    except Exception as e:
        print(f"❌ アプリケーションエラー: {e}")
        sys.exit(1)