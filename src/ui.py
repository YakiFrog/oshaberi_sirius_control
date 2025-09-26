from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject, Signal
from PySide6.QtWidgets import QTextEdit, QPushButton, QLineEdit, QComboBox, QLabel
from wake_word import WakeWordDetector
from voice_synthesis import VoiceSynthesizer
from realtime_recognition import RealtimeRecognizer
import time
import threading
import requests
import json

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
    speech_started = Signal()  # 音声再生開始シグナル
    llm_response_received = Signal(str)  # LLM応答受信シグナル

    # 利用可能なLLMモデルリスト
    AVAILABLE_MODELS = {
        "mistralai/magistral-small-2509": "Mistral Smallモデル",
        "openai/gpt-oss-20b": "OpenAI GPT OSS 20Bモデル"
    }

    def __init__(self):
        super().__init__()
        self.synthesizer = VoiceSynthesizer()
        self.recognizer = None  # 遅延初期化
        self.is_recognizing = False
        
        # 音声再生コマンドを検出
        self.audio_command = self._detect_audio_command()
        print(f"🔊 UI音声再生コマンド: {self.audio_command}")
        
        self.is_speaking = False  # 音声合成中フラグ
        self.model_loading = False
        self.llm_model = "mistralai/magistral-small-2509"  # デフォルトモデル
        
        # 音声再生関連
        self.audio_process = None  # 音声再生プロセス
        self.speech_thread = None  # 音声合成スレッド
        self.lipsync_thread = None  # リップシンクスレッド

    def _detect_audio_command(self):
        """利用可能な音声再生コマンドを検出"""
        import shutil
        
        # プラットフォーム別のコマンド優先順位
        commands = [
            'paplay',  # PulseAudio (Ubuntu/Linux preferred)
            'aplay',   # ALSA (Linux fallback)
            'ffplay',  # ffmpeg (Linux/cross-platform)
            'afplay',  # macOS
        ]
        
        for cmd in commands:
            if shutil.which(cmd):
                return cmd
        
        return None

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

    def chat_with_llm(self, user_message: str):
        """LLMとチャットして応答を取得"""
        def _chat_async():
            try:
                url = "http://localhost:1234/v1/chat/completions"
                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたは親切で役立つAIアシスタントです。日本語で答えてください。"
                        },
                        {
                            "role": "user", 
                            "content": user_message
                        } 
                    ],
                    "temperature": 0.7,
                    "max_tokens": -1,
                    "stream": False
                }

                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                ai_message = result["choices"][0]["message"]["content"]
                self.llm_response_received.emit(ai_message)
            except Exception as e:
                error_msg = f"LLMエラー: {e}"
                self.llm_response_received.emit(error_msg)

        thread = threading.Thread(target=_chat_async, daemon=True)
        thread.start()

    def cancel_speech(self):
        """音声再生をキャンセル"""
        if not self.is_speaking:
            return False
            
        try:
            # 音声再生プロセスを停止
            if self.audio_process and self.audio_process.poll() is None:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=1.0)
                print("🛑 音声再生をキャンセルしました")
            
            # リップシンクを停止（シグナルで通知）
            self._set_mouth_pattern_async(None)
            
            # フラグをリセット
            self.is_speaking = False
            self.audio_process = None
            self.speech_thread = None
            self.lipsync_thread = None
            
            self.speech_completed.emit()
            return True
            
        except Exception as e:
            print(f"⚠️ 音声キャンセルエラー: {e}")
            return False

    def speak_response(self, text: str):
        """応答音声を非同期で再生"""
        if self.is_speaking:
            return  # 既に発話中ならスキップ
            
        self.is_speaking = True
        
        def _speak_async():
            try:
                self._speak_with_lipsync(text)
            finally:
                self.is_speaking = False
                self.audio_process = None
                self.speech_thread = None
                self.lipsync_thread = None
                self.speech_completed.emit()
        
        self.speech_thread = threading.Thread(target=_speak_async, daemon=True)
        self.speech_thread.start()

    def _speak_with_lipsync(self, text: str):
        """音声合成 + リップシンク（キャンセル対応版）"""
        if not self.synthesizer.synthesizer:
            print("❌ Synthesizerが初期化されていません")
            return

        try:
            print(f"🎤 合成: 「{text}」 (速度: {self.synthesizer.speed_scale}x, スタイル: {self.synthesizer.style_id})")

            # AudioQuery作成
            audio_query = self.synthesizer.synthesizer.create_audio_query(text, self.synthesizer.style_id)
            self.synthesizer._set_speed_scale(audio_query, self.synthesizer.speed_scale)

            # 音声合成
            wav_data = self.synthesizer.synthesizer.synthesis(audio_query, self.synthesizer.style_id)
            print("✅ 音声合成成功")

            # リップシンク実行（プロセスを返す）
            self.audio_process = self._perform_lipsync_with_cancel(audio_query, wav_data, self.synthesizer.speed_scale)

        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")

    def _perform_lipsync_with_cancel(self, audio_query, wav_data, speed_scale: float):
        """リップシンク実行（キャンセル対応版）"""
        # 口形状シーケンス生成
        mouth_sequence = self.synthesizer._get_mouth_shape_sequence(audio_query, speed_scale)

        print("📝 口パターンシーケンス:")
        for i, (seq_time, mouth_shape, duration) in enumerate(mouth_sequence[:10]):
            print(".2f")
        if len(mouth_sequence) > 10:
            print(f"  ... 他{len(mouth_sequence) - 10}個")

        # 同期イベント作成
        audio_start_event = threading.Event()

        # 音声再生スレッド開始
        audio_thread = threading.Thread(
            target=self._play_audio_with_process,
            args=(wav_data, audio_start_event),
            daemon=True
        )
        audio_thread.start()

        # 音声再生開始を待機
        audio_start_event.wait()
        actual_audio_start = time.time()

        print(".6f")

        # リップシンク実行
        timing_stats = {'perfect': 0, 'good': 0, 'poor': 0}
        first_mouth_pattern = True

        for seq_time, mouth_shape, duration in mouth_sequence:
            # キャンセルチェック
            if not self.is_speaking:
                break
                
            # 目標時間 = 音声開始時間 + シーケンス時間
            target_time = actual_audio_start + seq_time

            # 高精度タイミング制御
            while self.is_speaking:  # キャンセルチェック
                current_time = time.time()
                time_to_target = target_time - current_time

                if time_to_target <= 0.0001:
                    break
                elif time_to_target > 0:
                    if time_to_target < 0.001:
                        pass
                    else:
                        time.sleep(max(0.0001, time_to_target - 0.0005))
                else:
                    break

            if not self.is_speaking:
                break

            # 口パターン設定
            server_pattern = f"mouth_{mouth_shape}" if mouth_shape else None
            if server_pattern:
                if first_mouth_pattern:
                    success = self.synthesizer._set_mouth_pattern(server_pattern)
                    if not success:
                        print(f"⚠️ 口パターン設定失敗: {server_pattern}")
                    first_mouth_pattern = False
                else:
                    self.synthesizer._set_mouth_pattern_async(server_pattern)

                # タイミング精度評価
                actual_time = time.time()
                timing_error_ms = (actual_time - target_time) * 1000

                if abs(timing_error_ms) <= 5:
                    sync_indicator = "✓"
                    timing_stats['perfect'] += 1
                elif abs(timing_error_ms) <= 15:
                    sync_indicator = "~"
                    timing_stats['good'] += 1
                else:
                    sync_indicator = "⚠"
                    timing_stats['poor'] += 1

                print(".1f")

        # 統計表示
        total_patterns = timing_stats['perfect'] + timing_stats['good'] + timing_stats['poor']
        if total_patterns > 0:
            perfect_rate = timing_stats['perfect'] / total_patterns * 100
            print(".1f")

        # 終了時に口パターンをリセット
        if self.is_speaking:  # キャンセルされていない場合のみ
            time.sleep(0.2)
            self.synthesizer._set_mouth_pattern_async(None)
            print("✅ 発話完了\n")

        return self.audio_process

    def _play_audio_with_process(self, wav_data, start_event):
        """音声を再生してプロセスを保存（クロスプラットフォーム対応）"""
        if not self.audio_command:
            print("❌ 音声再生コマンドが見つかりません")
            if start_event:
                start_event.set()
            return
        
        try:
            import tempfile
            import subprocess
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name

            # 再生開始を通知
            start_event.set()

            # 音声再生開始を通知
            self.speech_started.emit()

            # プラットフォーム別の音声再生
            if self.audio_command == 'ffplay':
                # ffplay（ログ出力を抑制）
                self.audio_process = subprocess.Popen([
                    'ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', temp_file_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # aplay, paplay, afplay
                self.audio_process = subprocess.Popen([self.audio_command, temp_file_path],
                                                    stdout=subprocess.DEVNULL,
                                                    stderr=subprocess.DEVNULL)
            
            self.audio_process.wait()
            
            # クリーンアップ
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
        except Exception as e:
            print(f"❌ 音声再生エラー: {e}")

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
            # 音声合成キャンセル
            self.cancel_speech()
            
            # 音声認識停止とクリーンアップ
            self.stop_recognition()
            if hasattr(self.recognizer, 'cleanup') and self.recognizer:
                self.recognizer.cleanup()
            
            # VOICEVOXクリーンアップ
            if self.synthesizer:
                self.synthesizer.cleanup()
            
            print("✅ VoiceController クリーンアップ完了")
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
        self.message_input = self.window.findChild(QLineEdit, "messageInput")
        self.model_selector = self.window.findChild(QComboBox, "modelSelector")
        self.cancel_speech_button = self.window.findChild(QPushButton, "cancelSpeechButton")

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
        self.voice_controller.speech_started.connect(self._on_speech_started)
        self.voice_controller.llm_response_received.connect(self._on_llm_response_received)

        # アプリ起動時にモデルを事前ロード
        self.voice_controller.preload_model()

        # ボタンのイベント接続
        self.wake_word_button.clicked.connect(self._on_wake_word_button_clicked)
        self.manual_audio_button.clicked.connect(self._on_manual_audio_button_clicked)
        self.send_button.clicked.connect(self._on_send_button_clicked)
        self.message_input.returnPressed.connect(self._on_send_button_clicked)
        self.model_selector.currentTextChanged.connect(self._on_model_changed)
        self.cancel_speech_button.clicked.connect(self._on_cancel_speech_button_clicked)

        # モデルセレクターの初期化
        self._setup_model_selector()

        # ボタンの初期スタイル設定
        self._setup_button_styles()

        # キャンセルボタンの初期状態（無効）
        self.cancel_speech_button.setEnabled(False)

        # 初期状態
        self._update_wake_word_button_text()
        self.current_transcription = ""  # 音声認識中のテキスト

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

        # キャンセルボタンのスタイル
        self.cancel_speech_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: 2px solid #c82333;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                border-color: #545b62;
                color: #adb5bd;
            }
        """)

    def _setup_model_selector(self):
        """モデルセレクターの初期化"""
        self.model_selector.clear()
        for model_name, description in self.voice_controller.AVAILABLE_MODELS.items():
            display_text = f"{description} ({model_name})"
            self.model_selector.addItem(display_text, model_name)
        
        # デフォルトモデルを選択
        default_index = self.model_selector.findData(self.voice_controller.llm_model)
        if default_index >= 0:
            self.model_selector.setCurrentIndex(default_index)

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

    def _on_llm_response_received(self, response: str):
        """LLM応答受信時の処理"""
        # 考え中メッセージを削除して応答を表示
        current_text = self.chat_display.toPlainText()
        lines = current_text.split('\n')
        if lines and "考え中..." in lines[-1]:
            lines = lines[:-1]
            current_text = '\n'.join(lines)
            self.chat_display.setPlainText(current_text)
        
        self._add_chat_message(f"🤖 AI: {response}")
        
        # 応答を音声合成 + リップシンク
        self.voice_controller.speak_response(response)

    def _on_speech_started(self):
        """音声再生開始時の処理"""
        # キャンセルボタンを有効化
        self.cancel_speech_button.setEnabled(True)

    def _on_speech_completed(self):
        
        # キャンセルボタンを有効化
        self.cancel_speech_button.setEnabled(True)

    def _on_speech_completed(self):
        """音声合成完了時の処理"""
        # キャンセルボタンを無効化
        self.cancel_speech_button.setEnabled(False)
        
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
        self.current_transcription = text

    def _on_silence_detected(self):
        """沈黙検出時の処理"""
        self._add_chat_message("🔇 沈黙を検知しました。自動送信します...")
        self.voice_controller.stop_recognition()
        
        if self.current_transcription:
            # 音声認識のテキストをLLMに送信
            self._add_chat_message(f"👤 あなた: {self.current_transcription}")
            self._add_chat_message("🤖 AI: 考え中...")
            self.voice_controller.chat_with_llm(self.current_transcription)
            self.current_transcription = ""  # リセット

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
        """送信ボタンのクリックイベント"""
        message = self.message_input.text().strip()
        if not message:
            self._add_chat_message("⚠️ メッセージを入力してください")
            return

        # ユーザーメッセージを表示
        self._add_chat_message(f"👤 あなた: {message}")
        
        # 入力フィールドをクリア
        self.message_input.clear()
        
        # LLMに送信
        self._add_chat_message("🤖 AI: 考え中...")
        self.voice_controller.chat_with_llm(message)

    def _on_model_changed(self, display_text):
        """モデル選択変更時の処理"""
        if display_text:
            selected_model = self.model_selector.currentData()
            if selected_model:
                self.voice_controller.llm_model = selected_model
                self._add_chat_message(f"🔄 LLMモデルを変更しました: {display_text}")

    def _on_cancel_speech_button_clicked(self):
        """読み上げキャンセルボタンのクリックイベント"""
        if self.voice_controller.cancel_speech():
            self._add_chat_message("🛑 読み上げをキャンセルしました")
        else:
            self._add_chat_message("⚠️ キャンセル可能な読み上げがありません")

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

    def cleanup(self):
        """アプリケーション終了時のクリーンアップ"""
        try:
            # 音声合成キャンセル
            if hasattr(self.voice_controller, 'cancel_speech'):
                self.voice_controller.cancel_speech()
            
            # 音声認識停止
            if hasattr(self.voice_controller, 'stop_recognition'):
                self.voice_controller.stop_recognition()
            
            # ウェイクワード検出停止
            if hasattr(self.wake_controller, 'detector') and self.wake_controller.detector:
                self.wake_controller.detector.stop_detection()
            
            # 各コントローラーのクリーンアップ
            if hasattr(self.voice_controller, 'cleanup'):
                self.voice_controller.cleanup()
            
            print("✅ MainWindow クリーンアップ完了")
        except Exception as e:
            print(f"⚠️ MainWindow クリーンアップ警告: {e}")

    def closeEvent(self, event):
        """ウィンドウクローズイベント"""
        self.cleanup()
        event.accept()