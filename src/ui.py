from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject, Signal
from PySide6.QtWidgets import QTextEdit, QPushButton
from wake_word import WakeWordDetector
from voice_synthesis import VoiceSynthesizer
from realtime_recognition import RealtimeRecognizer
import time
import threading

class WakeWordController(QObject):
    """ウェイクワード検出のコントローラー（スレッドセーフなUI更新用）"""
    wake_word_detected = Signal(str, float)  # 検出テキストと確信度
    model_loaded = Signal()  # モデルロード完了シグナル

    def __init__(self):
        super().__init__()
        self.detector = None  # 遅延初期化
        self.is_detecting = False
        self.model_loading = False

    def preload_model(self):
        """アプリ起動時にモデルを事前ロード"""
        if self.model_loading or self.detector is not None:
            return

        self.model_loading = True
        print("🚀 ウェイクワード検出モデルを事前ロード中...")

        # バックグラウンドでモデルをロード
        import threading
        load_thread = threading.Thread(target=self._preload_model_worker, daemon=True)
        load_thread.start()

    def _preload_model_worker(self):
        """モデル事前ロードワーカー"""
        try:
            from wake_word import WakeWordDetector
            self.detector = WakeWordDetector(self._on_wake_word_detected)
            # モデルを初期化
            self.detector._init_model()
            self.model_loaded.emit()
            print("✅ ウェイクワード検出モデルロード完了")
        except Exception as e:
            print(f"❌ モデルロードエラー: {e}")
        finally:
            self.model_loading = False

    def _get_detector(self):
        """検出器を取得（モデルがロード済みの場合のみ）"""
        if self.detector is None:
            if self.model_loading:
                print("⏳ モデルロード中です。しばらくお待ちください...")
                return None
            else:
                # フォールバック: その場でロード
                from wake_word import WakeWordDetector
                self.detector = WakeWordDetector(self._on_wake_word_detected)
        return self.detector

    def _on_wake_word_detected(self, text, confidence):
        """ウェイクワード検出時のコールバック"""
        self.wake_word_detected.emit(text, confidence)

    def toggle_detection(self):
        """検出の開始/停止を切り替え"""
        detector = self._get_detector()
        if detector is None:
            print("❌ モデルがまだロードされていません")
            return False
        
        if self.is_detecting:
            detector.stop_detection()
            self.is_detecting = False
        else:
            detector.start_detection()
            self.is_detecting = True
        return self.is_detecting

class VoiceController(QObject):
    """音声入出力コントローラー"""
    transcription_received = Signal(str, float)  # 認識テキストと確信度
    silence_detected = Signal()  # 沈黙検出
    model_loaded = Signal()  # モデルロード完了シグナル
    speech_completed = Signal()  # 音声合成完了シグナル

    def __init__(self):
        super().__init__()
        self.synthesizer = VoiceSynthesizer()
        self.recognizer = None  # 遅延初期化
        self.is_recognizing = False
        self.is_speaking = False  # 音声合成中フラグ
        self.model_loading = False

    def preload_model(self):
        """音声認識モデルを事前ロード"""
        if self.model_loading or self.recognizer is not None:
            return

        self.model_loading = True
        print("🚀 音声認識モデルを事前ロード中...")

        # バックグラウンドでモデルをロード
        import threading
        load_thread = threading.Thread(target=self._preload_model_worker, daemon=True)
        load_thread.start()

    def _preload_model_worker(self):
        """モデル事前ロードワーカー"""
        try:
            from realtime_recognition import RealtimeRecognizer
            self.recognizer = RealtimeRecognizer(
                transcription_callback=self._on_transcription,
                silence_callback=self._on_silence
            )
            # tqdmの競合を避けるため、モデル初期化を遅らせる
            print("✅ 音声認識モデル初期化準備完了")
            self.model_loaded.emit()
        except Exception as e:
            print(f"❌ 音声認識モデルロードエラー: {e}")
        finally:
            self.model_loading = False

    def _get_recognizer(self):
        """認識器を取得（モデルがロード済みの場合のみ）"""
        if self.recognizer is None:
            if self.model_loading:
                print("⏳ 音声認識モデルロード中です。しばらくお待ちください...")
                return None
            else:
                # フォールバック: その場でロード
                from realtime_recognition import RealtimeRecognizer
                self.recognizer = RealtimeRecognizer(
                    transcription_callback=self._on_transcription,
                    silence_callback=self._on_silence
                )
        return self.recognizer

    def _on_transcription(self, text, confidence):
        """音声認識結果のコールバック"""
        self.transcription_received.emit(text, confidence)

    def _on_silence(self):
        """沈黙検出のコールバック"""
        self.silence_detected.emit()

    def speak_response(self, text: str):
        """応答音声を非同期で再生"""
        if self.is_speaking:
            return  # 既に発話中ならスキップ
            
        self.is_speaking = True
        
        def _speak_async():
            try:
                self.synthesizer.speak_with_lipsync(text)
            finally:
                self.is_speaking = False
                self.speech_completed.emit()
        
        thread = threading.Thread(target=_speak_async, daemon=True)
        thread.start()

    def start_recognition(self):
        """音声認識を開始"""
        recognizer = self._get_recognizer()
        if recognizer is None:
            print("❌ 音声認識モデルがまだロードされていません")
            return
            
        if self.is_speaking:
            print("⏳ 音声合成中です。合成完了まで待機します...")
            # 合成完了を待ってから認識を開始
            def _delayed_start():
                while self.is_speaking:
                    time.sleep(0.1)
                if not self.is_recognizing:
                    recognizer.start_recognition()
                    self.is_recognizing = True
            
            thread = threading.Thread(target=_delayed_start, daemon=True)
            thread.start()
            return
            
        if not self.is_recognizing:
            recognizer.start_recognition()
            self.is_recognizing = True

    def stop_recognition(self):
        """音声認識を停止"""
        if self.is_recognizing and self.recognizer:
            self.recognizer.stop_recognition()
            self.is_recognizing = False

    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            # 音声認識停止
            self.stop_recognition()
            # VOICEVOXクリーンアップ
            if self.synthesizer:
                self.synthesizer.cleanup()
        except Exception as e:
            print(f"⚠️ VoiceControllerクリーンアップ警告: {e}")

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
        self.wake_controller.model_loaded.connect(self._on_model_loaded)

        # アプリ起動時にモデルを事前ロード
        self.wake_controller.preload_model()

        # 音声コントローラー初期化
        self.voice_controller = VoiceController()
        self.voice_controller.transcription_received.connect(self._on_transcription_received)
        self.voice_controller.silence_detected.connect(self._on_silence_detected)
        self.voice_controller.model_loaded.connect(self._on_voice_model_loaded)
        self.voice_controller.speech_completed.connect(self._on_speech_completed)

        # アプリ起動時にモデルを事前ロード
        self.voice_controller.preload_model()

        # ボタンのイベント接続
        self.wake_word_button.clicked.connect(self._on_wake_word_button_clicked)
        self.manual_audio_button.clicked.connect(self._on_manual_audio_button_clicked)

        # ボタンの初期スタイル設定
        self._setup_button_styles()

        # 初期状態
        self._update_wake_word_button_text()

    def _setup_button_styles(self):
        """全ボタンの初期スタイルを設定"""
        # 送信ボタン（青色）
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #339af0;
                color: white;
                border: 2px solid #228be6;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #228be6;
            }
            QPushButton:pressed {
                background-color: #1c7ed6;
            }
        """)

        # 手動音声入力ボタン（オレンジ色）
        self.manual_audio_button.setStyleSheet("""
            QPushButton {
                background-color: #ff922b;
                color: white;
                border: 2px solid #fd7e14;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #fd7e14;
            }
            QPushButton:pressed {
                background-color: #f76707;
            }
        """)

        # ウェイクワードボタンの初期スタイル（停止中）
        self._update_wake_word_button_text()

    def _on_wake_word_button_clicked(self):
        """ウェイクワード検知ボタンのクリックイベント"""
        is_detecting = self.wake_controller.toggle_detection()
        if is_detecting is False and self.wake_controller.model_loading:
            self._add_chat_message("⏳ モデルロード中です。しばらくお待ちください...")
            return
            
        self._update_wake_word_button_text()
        
        if is_detecting:
            self._add_chat_message("🎤 ウェイクワード検出を開始しました")
        else:
            self._add_chat_message("🛑 ウェイクワード検出を停止しました")

    def _on_model_loaded(self):
        """モデルロード完了時の処理"""
        self._add_chat_message("✅ ウェイクワード検出モデルが準備できました")

    def _on_voice_model_loaded(self):
        """音声認識モデルロード完了時の処理"""
        self._add_chat_message("✅ 音声認識モデルが準備できました")

    def _on_speech_completed(self):
        """音声合成完了時の処理"""
        # 音声合成完了後に音声認識を開始
        if not self.voice_controller.is_recognizing:
            self.voice_controller.start_recognition()
            self._add_chat_message("🎤 音声入力を開始しました...")

    def _update_wake_word_button_text(self):
        """ウェイクワードボタンのテキストと色を更新"""
        if self.wake_controller.is_detecting:
            self.wake_word_button.setText("ウェイクワード検知停止")
            self.wake_word_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff6b6b;
                    color: white;
                    border: 2px solid #ff5252;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #ff5252;
                }
                QPushButton:pressed {
                    background-color: #ff3838;
                }
            """)
        else:
            self.wake_word_button.setText("ウェイクワード検知開始")
            self.wake_word_button.setStyleSheet("""
                QPushButton {
                    background-color: #51cf66;
                    color: white;
                    border: 2px solid #40c057;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #40c057;
                }
                QPushButton:pressed {
                    background-color: #37b24d;
                }
            """)

    def _on_wake_word_detected(self, text, confidence):
        """ウェイクワード検出時のUI更新"""
        # 即座に検出メッセージを表示
        message = f"🎯 ウェイクワード検出: 「{text}」 (確信度: {confidence:.1f}%)"
        self._add_chat_message(message)

        # 応答音声を非同期で再生（合成完了後に音声認識を開始）
        self.voice_controller.speak_response("はい，なんですか？")

    def _on_transcription_received(self, text, confidence):
        """音声認識結果受信時の処理"""
        message = f"🎯 認識: 「{text}」 (確信度: {confidence:.1f}%)"
        self._add_chat_message(message)

    def _on_silence_detected(self):
        """沈黙検出時の処理"""
        self._add_chat_message("🔇 沈黙を検知しました。自動送信します...")
        self.voice_controller.stop_recognition()
        self._on_send_button_clicked()

    def _on_manual_audio_button_clicked(self):
        """手動音声入力ボタンのクリックイベント"""
        if self.voice_controller.model_loading:
            self._add_chat_message("⏳ 音声認識モデルロード中です。しばらくお待ちください...")
            return
            
        if self.voice_controller.is_recognizing:
            self.voice_controller.stop_recognition()
            self._add_chat_message("🛑 音声入力を停止しました")
        else:
            self.voice_controller.start_recognition()
            self._add_chat_message("🎤 音声入力を開始しました...")

        self._update_manual_audio_button_text()

    def _update_manual_audio_button_text(self):
        """手動音声入力ボタンのテキストを更新"""
        if self.voice_controller.is_recognizing:
            self.manual_audio_button.setText("音声入力停止")
        else:
            self.manual_audio_button.setText("手動音声入力")

    def _on_send_button_clicked(self):
        """送信ボタンのクリックイベント（仮実装）"""
        self._add_chat_message("📤 メッセージを送信しました（LLM処理は未実装）")

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