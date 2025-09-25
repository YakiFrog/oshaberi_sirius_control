from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject, Signal
from PySide6.QtWidgets import QTextEdit, QPushButton
from wake_word import WakeWordDetector

class WakeWordController(QObject):
    """ウェイクワード検出のコントローラー（スレッドセーフなUI更新用）"""
    wake_word_detected = Signal(str, float)  # 検出テキストと確信度

    def __init__(self):
        super().__init__()
        self.detector = WakeWordDetector(self._on_wake_word_detected)
        self.is_detecting = False

    def _on_wake_word_detected(self, text, confidence):
        """ウェイクワード検出時のコールバック"""
        self.wake_word_detected.emit(text, confidence)

    def toggle_detection(self):
        """検出の開始/停止を切り替え"""
        if self.is_detecting:
            self.detector.stop_detection()
            self.is_detecting = False
        else:
            self.detector.start_detection()
            self.is_detecting = True
        return self.is_detecting

class MainWindow:
    def __init__(self):
        # .uiファイルをロード
        ui_file = QFile("src/main_window.ui")
        ui_file.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()

        # ウィジェットへの参照を取得
        self.chat_display = self.window.findChild(QTextEdit, "chatDisplay")
        self.send_button = self.window.findChild(QPushButton, "sendButton")
        self.manual_audio_button = self.window.findChild(QPushButton, "manualAudioButton")
        self.wake_word_button = self.window.findChild(QPushButton, "wakeWordButton")

        # ウェイクワードコントローラー初期化
        self.wake_controller = WakeWordController()
        self.wake_controller.wake_word_detected.connect(self._on_wake_word_detected)

        # ボタンのイベント接続
        self.wake_word_button.clicked.connect(self._on_wake_word_button_clicked)

        # 初期状態
        self._update_wake_word_button_text()

    def _on_wake_word_button_clicked(self):
        """ウェイクワード検知ボタンのクリックイベント"""
        is_detecting = self.wake_controller.toggle_detection()
        self._update_wake_word_button_text()
        
        if is_detecting:
            self._add_chat_message("🎤 ウェイクワード検出を開始しました")
        else:
            self._add_chat_message("🛑 ウェイクワード検出を停止しました")

    def _update_wake_word_button_text(self):
        """ウェイクワードボタンのテキストを更新"""
        if self.wake_controller.is_detecting:
            self.wake_word_button.setText("ウェイクワード検知停止")
        else:
            self.wake_word_button.setText("ウェイクワード検知開始")

    def _on_wake_word_detected(self, text, confidence):
        """ウェイクワード検出時のUI更新"""
        message = f"🎯 ウェイクワード検出: 「{text}」 (確信度: {confidence:.1f}%)"
        self._add_chat_message(message)

    def _add_chat_message(self, message):
        """チャットディスプレイにメッセージを追加"""
        current_text = self.chat_display.toPlainText()
        timestamp = time.strftime("%H:%M:%S")
        new_message = f"[{timestamp}] {message}\n"
        
        if current_text:
            self.chat_display.setPlainText(current_text + new_message)
        else:
            self.chat_display.setPlainText(new_message)
        
        # 自動スクロール
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def show(self):
        self.window.show()