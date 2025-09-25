#!/usr/bin/env python3
"""
リアルタイムウェイクワード検出プログラム
「シリウスくん」をウェイクワードとして検出するプログラム
realtime_whisper_test.pyとfaster_whisper_test.pyの実装を参考に作成
"""

# 必要なモジュールのチェック
try:
    import pyaudio
except ImportError:
    print("❌ pyaudioがインストールされていません。")
    print("インストール方法: pip install pyaudio")
    print("または: conda install pyaudio")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ numpyがインストールされていません。")
    print("インストール方法: pip install numpy")
    exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster_whisperがインストールされていません。")
    print("インストール方法: pip install faster-whisper")
    exit(1)

import time
import threading
import queue
import os

# =============================================================================
# 🚀 高速化設定パラメータ（faster_whisper_test.pyからインポート）
# =============================================================================

# モデル設定（高速化の基盤）
MODEL_CONFIG = {
    "model_size": "small",         # リアルタイム用にsmallモデルを使用
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 6,              # リアルタイム用に適度なスレッド数
    "num_workers": 1
}

# 認識パラメータ（速度重視のチューニング）
TRANSCRIBE_CONFIG = {
    "language": "ja",
    "beam_size": 3,                # 改善: 1から3に上げて精度を向上（速度とのトレードオフ）
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,  # 改善: 2.0から2.4に上げてより自然な音声を処理
    "log_prob_threshold": -1.0,    # 改善: -0.8から-1.0に下げてより多くの候補を処理
    "no_speech_threshold": 0.3,    # 改善: 0.4から0.3に下げてより多くの音声を処理
    "condition_on_previous_text": True,
    "initial_prompt": "以下は日本語の音声です。ウェイクワード「シリウスくん」を聞いてください。",
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 300,   # 改善: 500msから300msに短くして短い発話も検出
        "speech_pad_ms": 50,              # 改善: 30msから50msに増やして発話を確実に捉える
        "threshold": 0.4                  # 改善: 0.5から0.4に下げて感度を上げる
    }
}

# リアルタイム設定
REALTIME_CONFIG = {
    "chunk": 1024,
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 16000,
    "buffer_seconds": 1.5,         # 改善: 1.0秒から1.5秒に増やしてウェイクワードを確実に捉える
    "overlap_seconds": 0.3,        # 改善: 0.2秒から0.3秒に増やして重複を増やす
    "processing_interval": 0.3     # 改善: 0.5秒から0.3秒に短くして検出頻度を上げる
}

# ウェイクワード設定
WAKE_WORD_CONFIG = {
    "wake_word": "シリウスくん",     # ウェイクワード
    "confidence_threshold": 60.0,  # 検出信頼度の閾値（%）- 改善: 60%から30%に下げる
    "cooldown_seconds": 2.0,       # 検出後のクールダウン時間（秒）- 改善: 3秒から2秒に短く
    "max_detection_history": 10,   # 検出履歴の最大保持数
    "debug_mode": False            # デバッグモード - 改善完了後はOFFに
}

# =============================================================================
# ウェイクワード検出クラス
# =============================================================================

class WakeWordDetector:
    def __init__(self, wake_word_callback=None):
        # モデル初期化
        print("🚀 Faster Whisperモデルをロード中...")
        self.model = WhisperModel(
            MODEL_CONFIG["model_size"],
            device=MODEL_CONFIG["device"],
            compute_type=MODEL_CONFIG["compute_type"],
            cpu_threads=MODEL_CONFIG["cpu_threads"],
            num_workers=MODEL_CONFIG["num_workers"]
        )
        print("✅ モデルロード完了")

        # 音声設定
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.last_processed_time = 0

        # バッファ管理
        self.audio_buffer = np.array([], dtype=np.int16)
        self.buffer_lock = threading.Lock()

        # ウェイクワード設定
        self.wake_word = WAKE_WORD_CONFIG["wake_word"]
        self.confidence_threshold = WAKE_WORD_CONFIG["confidence_threshold"]
        self.cooldown_seconds = WAKE_WORD_CONFIG["cooldown_seconds"]
        self.last_detection_time = 0
        self.detection_history = []
        self.debug_mode = WAKE_WORD_CONFIG.get("debug_mode", False)

        # コールバック関数
        self.wake_word_callback = wake_word_callback or self._default_wake_word_callback

    def _default_wake_word_callback(self, detected_text, confidence):
        """デフォルトのウェイクワード検出時のコールバック"""
        print(f"\n🎯 ウェイクワード検出: 「{detected_text}」 (確信度: {confidence:.1f}%)")
        print("🔔 シリウスくんが呼び出されました！")
        # ここに実際のアクションを追加（例: 音声合成、画面表示など）

        # ビープ音を鳴らす（オプション）
        try:
            import platform
            if platform.system() == "Darwin":  # macOS
                os.system("afplay /System/Library/Sounds/Ping.aiff")
            elif platform.system() == "Linux":
                os.system("beep -f 800 -l 200")
        except:
            pass

    def start_detection(self):
        """ウェイクワード検出を開始"""
        print(f"🎤 ウェイクワード検出を開始します...")
        print(f"💡 ウェイクワード: 「{self.wake_word}」")
        print("💡 話しかけると自動で検出されます（Ctrl+Cで終了）")

        # 音声ストリーム開始
        self.stream = self.audio.open(
            format=REALTIME_CONFIG["format"],
            channels=REALTIME_CONFIG["channels"],
            rate=REALTIME_CONFIG["rate"],
            input=True,
            frames_per_buffer=REALTIME_CONFIG["chunk"]
        )

        self.is_running = True

        # 録音スレッド開始
        recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        recording_thread.start()

        # 検出スレッド開始
        detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        detection_thread.start()

        try:
            # メインスレッドは待機
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 終了します...")
            self.stop_detection()

    def stop_detection(self):
        """ウェイクワード検出を停止"""
        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        print("✅ ウェイクワード検出を終了しました")

    def _recording_worker(self):
        """音声録音ワーカー"""
        print("🎙️  録音スレッド開始")

        while self.is_running:
            try:
                # 音声データを取得
                data = self.stream.read(REALTIME_CONFIG["chunk"], exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # バッファに追加
                with self.buffer_lock:
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

                # 音量レベル表示（100msごと）
                current_time = time.time()
                if int(current_time * 10) % 10 == 0:  # 100msごと
                    volume = self._calculate_volume(audio_chunk)
                    print(f"🎵 音量: {volume:.0f}", end='\r')

            except Exception as e:
                print(f"❌ 録音エラー: {e}")
                break

        print("\n📥 録音スレッド終了")

    def _detection_worker(self):
        """ウェイクワード検出ワーカー"""
        print("🧠 検出スレッド開始")

        while self.is_running:
            current_time = time.time()

            # 処理間隔チェック
            if current_time - self.last_processed_time < REALTIME_CONFIG["processing_interval"]:
                time.sleep(0.05)  # 短い待機
                continue

            # クールダウンチェック
            if current_time - self.last_detection_time < self.cooldown_seconds:
                time.sleep(0.05)  # クールダウン中は待機
                continue

            # 十分なバッファがあるかチェック
            with self.buffer_lock:
                buffer_duration = len(self.audio_buffer) / REALTIME_CONFIG["rate"]

                if buffer_duration < REALTIME_CONFIG["buffer_seconds"]:
                    time.sleep(0.05)  # バッファが足りない場合は待機
                    continue

                # 処理する音声データを抽出（オーバーラップ考慮）
                process_samples = int(REALTIME_CONFIG["buffer_seconds"] * REALTIME_CONFIG["rate"])
                overlap_samples = int(REALTIME_CONFIG["overlap_seconds"] * REALTIME_CONFIG["rate"])

                # 最新のデータを処理
                audio_to_process = self.audio_buffer[-process_samples:]

                # バッファをオーバーラップ分残して更新
                keep_samples = overlap_samples
                if len(self.audio_buffer) > keep_samples:
                    self.audio_buffer = self.audio_buffer[-keep_samples:]

            # ウェイクワード検出処理
            try:
                self._process_audio_for_wake_word(audio_to_process)
                self.last_processed_time = current_time

            except Exception as e:
                print(f"❌ 検出エラー: {e}")

        print("\n🔍 検出スレッド終了")

    def _process_audio_for_wake_word(self, audio_chunk):
        """音声チャンクからウェイクワードを検出"""
        # NumPy配列をfloat32に変換
        audio_np = audio_chunk.astype(np.float32) / 32768.0

        # 無音チェック
        volume = self._calculate_volume(audio_chunk)
        if volume < 100:  # 音量が小さすぎる場合はスキップ
            return

        # Whisperで認識
        segments, info = self.model.transcribe(
            audio_np,
            language=TRANSCRIBE_CONFIG["language"],
            beam_size=TRANSCRIBE_CONFIG["beam_size"],
            temperature=TRANSCRIBE_CONFIG["temperature"],
            compression_ratio_threshold=TRANSCRIBE_CONFIG["compression_ratio_threshold"],
            log_prob_threshold=TRANSCRIBE_CONFIG["log_prob_threshold"],
            no_speech_threshold=TRANSCRIBE_CONFIG["no_speech_threshold"],
            condition_on_previous_text=TRANSCRIBE_CONFIG["condition_on_previous_text"],
            initial_prompt=TRANSCRIBE_CONFIG["initial_prompt"],
            word_timestamps=TRANSCRIBE_CONFIG["word_timestamps"],
            vad_filter=TRANSCRIBE_CONFIG["vad_filter"],
            vad_parameters=TRANSCRIBE_CONFIG["vad_parameters"]
        )

        # 結果を取得
        segments_list = list(segments)
        if not segments_list:
            return

        text = "".join(segment.text for segment in segments_list).strip()
        if not text:  # 空でないテキストのみ処理
            return

        confidence = self._calculate_simple_confidence(segments_list, info)
        duration = len(audio_chunk) / REALTIME_CONFIG["rate"]

        # デバッグ情報表示
        if self.debug_mode:
            print(f"\n🔍 認識結果: 「{text}」 (確信度: {confidence:.1f}%, 時間: {duration:.1f}s)")

        # ウェイクワード検出
        if self._check_wake_word(text, confidence):
            # 検出履歴に追加
            detection_info = {
                'timestamp': time.time(),
                'text': text,
                'confidence': confidence,
                'duration': duration
            }
            self.detection_history.append(detection_info)

            # 履歴を制限
            if len(self.detection_history) > WAKE_WORD_CONFIG["max_detection_history"]:
                self.detection_history.pop(0)

            # コールバック呼び出し
            self.wake_word_callback(text, confidence)
            self.last_detection_time = time.time()

        else:
            # 通常の認識結果を表示（デバッグモード時のみ詳細表示）
            if self.debug_mode:
                if len(text) <= 30:  # 少し長いテキストも表示
                    print(f"💭 通常認識: 「{text}」 (確信度: {confidence:.1f}%)")
            else:
                # 非デバッグモードでは短いテキストのみ表示
                if len(text) <= 20:
                    print(f"🎯 [{duration:.1f}s] {text}", end='\r')

    def _check_wake_word(self, text, confidence):
        """テキストにウェイクワードが含まれているかチェック"""
        # 信頼度チェック（デバッグモードでは緩和）
        if not self.debug_mode and confidence < self.confidence_threshold:
            return False

        # ノイズフィルタリング（明らかに無意味なテキストを除外）
        if len(text.strip()) < 1:  # 1文字未満は無視（デバッグ時は緩和）
            return False

        # 英語のみのテキストは無視（日本語のウェイクワード検出用）
        if not any(ord(char) > 127 for char in text):  # ASCII文字のみの場合は無視
            return False

        # ウェイクワードチェック（より柔軟なマッチング）
        text_lower = text.lower()

        # 完全一致を優先
        if self.wake_word in text:
            return True

        # より柔軟なマッチング（「シリウスくん」のバリエーション）
        wake_word_variants = [
            "シリウスくん",
            "シリウス",
            "しりうすくん",
            "しりうす",
            "シリウス くん",  # スペース入り
            "しりうす くん",
            "シリウスさん",   # より柔軟な表現
            "しりうすさん",
            "ねえ，シリウスくん",  # 自然な呼びかけ
            "ねえシリウスくん",
            "ねえ、シリウスくん",
            "ねえ シリウスくん",
            "おい、シリウスくん",  # 他の呼びかけも追加
            "おいシリウスくん",
            "ちょっと、シリウスくん",
            "ちょっとシリウスくん"
        ]

        for variant in wake_word_variants:
            if variant in text_lower:
                return True

        # 部分一致（「シリウス」が含まれていれば検出）
        if "シリウス" in text or "しりうす" in text:
            return True

        return False

    def _calculate_volume(self, audio_chunk):
        """音量を計算"""
        if len(audio_chunk) == 0:
            return 0
        return np.sqrt(np.mean(audio_chunk.astype(np.float64)**2))

    def _calculate_simple_confidence(self, segments, info):
        """簡易的な信頼度計算"""
        try:
            if hasattr(info, 'language_probability') and info.language_probability:
                return info.language_probability * 100

            # セグメントの平均確率を使用
            confidences = []
            for segment in segments:
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                    # 対数確率をパーセンテージに変換
                    confidence = min(100.0, max(0.0, (segment.avg_logprob + 5.0) / 5.0 * 100))
                    confidences.append(confidence)

            return sum(confidences) / len(confidences) if confidences else 50.0

        except:
            return 50.0

    def get_detection_history(self):
        """検出履歴を取得"""
        return self.detection_history.copy()

    def clear_detection_history(self):
        """検出履歴をクリア"""
        self.detection_history.clear()

# =============================================================================
# カスタムコールバック例
# =============================================================================

def custom_wake_word_callback(detected_text, confidence):
    """カスタムウェイクワード検出時の処理例"""
    print(f"\n🚨 ウェイクワード検出！")
    print(f"📝 検出テキスト: {detected_text}")
    print(f"📊 確信度: {confidence:.1f}%")
    print("🎉 シリウスくん、参上！")

    # ここに実際のアクションを追加
    # 例: 音声合成、画面表示、API呼び出しなど

# =============================================================================
# メイン関数
# =============================================================================

def main():
    print("🎤 リアルタイムウェイクワード検出プログラム")
    print("=" * 50)
    print(f"🎯 ウェイクワード: 「{WAKE_WORD_CONFIG['wake_word']}」")
    print("💡 対応パターン: 「シリウスくん」「ねえ、シリウスくん」「おい、シリウスくん」など")
    print(f"📊 モデル: {MODEL_CONFIG['model_size']} ({MODEL_CONFIG['compute_type']})")
    print(f"⚡ CPUスレッド: {MODEL_CONFIG['cpu_threads']}")
    print(f"⏱️  処理間隔: {REALTIME_CONFIG['processing_interval']}秒")
    print(f"🎵 バッファ: {REALTIME_CONFIG['buffer_seconds']}秒")
    print(f"🔄 クールダウン: {WAKE_WORD_CONFIG['cooldown_seconds']}秒")
    print(f"📈 信頼度閾値: {WAKE_WORD_CONFIG['confidence_threshold']}%")
    print(f"🐛 デバッグモード: {'ON' if WAKE_WORD_CONFIG.get('debug_mode', False) else 'OFF'}")
    print("=" * 50)

    # ウェイクワード検出インスタンス作成
    # detector = WakeWordDetector()  # デフォルトコールバック使用
    detector = WakeWordDetector(custom_wake_word_callback)  # カスタムコールバック使用

    try:
        # ウェイクワード検出開始
        detector.start_detection()

    except KeyboardInterrupt:
        print("\n🛑 プログラムを終了します")
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        detector.stop_detection()

        # 検出履歴を表示
        history = detector.get_detection_history()
        if history:
            print("\n📋 検出履歴:")
            for i, detection in enumerate(history, 1):
                timestamp = time.strftime("%H:%M:%S", time.localtime(detection['timestamp']))
                print(f"  {i}. {timestamp} - 「{detection['text']}」 ({detection['confidence']:.1f}%)")
        else:
            print("\n📋 検出履歴: なし")

if __name__ == "__main__":
    main()
