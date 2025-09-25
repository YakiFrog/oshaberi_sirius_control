#!/usr/bin/env python3
"""
ウェイクワード検出モジュール
PySide6 UI統合用に最適化
"""

# 必要なモジュールのチェック
try:
    import pyaudio
except ImportError:
    print("❌ pyaudioがインストールされていません。")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ numpyがインストールされていません。")
    exit(1)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster_whisperがインストールされていません。")
    exit(1)

import time
import threading
import queue
import os

# =============================================================================
# 🚀 高速化設定パラメータ
# =============================================================================

MODEL_CONFIG = {
    "model_size": "small",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 6,
    "num_workers": 1
}

TRANSCRIBE_CONFIG = {
    "language": "ja",
    "beam_size": 1,  # 高速化のため1に変更
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.3,
    "condition_on_previous_text": True,
    "initial_prompt": "以下は日本語の音声です。ウェイクワード「シリウスくん」を聞いてください。",
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 50,
        "threshold": 0.4
    }
}

REALTIME_CONFIG = {
    "chunk": 1024,
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 16000,
    "buffer_seconds": 1.5,
    "overlap_seconds": 0.3,
    "processing_interval": 0.3
}

WAKE_WORD_CONFIG = {
    "wake_word": "シリウスくん",
    "confidence_threshold": 60.0,
    "cooldown_seconds": 2.0,
    "max_detection_history": 10,
    "debug_mode": False
}

# =============================================================================
# ウェイクワード検出クラス
# =============================================================================

class WakeWordDetector:
    def __init__(self, wake_word_callback=None):
        self.model = None  # 遅延初期化
        self.audio = None
        self.stream = None
        self.is_running = False
        self.last_processed_time = 0
        self.audio_buffer = np.array([], dtype=np.int16)
        self.buffer_lock = threading.Lock()
        self.wake_word = WAKE_WORD_CONFIG["wake_word"]
        self.confidence_threshold = WAKE_WORD_CONFIG["confidence_threshold"]
        self.cooldown_seconds = WAKE_WORD_CONFIG["cooldown_seconds"]
        self.last_detection_time = 0
        self.detection_history = []
        self.debug_mode = WAKE_WORD_CONFIG.get("debug_mode", False)
        self.wake_word_callback = wake_word_callback or self._default_wake_word_callback

    def _init_model(self):
        """モデルを遅延初期化"""
        if self.model is None:
            print("🚀 Faster Whisperモデルをロード中...")
            self.model = WhisperModel(
                MODEL_CONFIG["model_size"],
                device=MODEL_CONFIG["device"],
                compute_type=MODEL_CONFIG["compute_type"],
                cpu_threads=MODEL_CONFIG["cpu_threads"],
                num_workers=MODEL_CONFIG["num_workers"]
            )
            print("✅ モデルロード完了")

    def _default_wake_word_callback(self, detected_text, confidence):
        """デフォルトのウェイクワード検出時のコールバック"""
        print(f"\n🎯 ウェイクワード検出: 「{detected_text}」 (確信度: {confidence:.1f}%)")
        print("🔔 シリウスくんが呼び出されました！")

    def start_detection(self):
        """ウェイクワード検出を開始"""
        if self.is_running:
            return

        self._init_model()

        print(f"🎤 ウェイクワード検出を開始します...")
        print(f"💡 ウェイクワード: 「{self.wake_word}」")

        self.audio = pyaudio.PyAudio()
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

    def stop_detection(self):
        """ウェイクワード検出を停止"""
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio:
            self.audio.terminate()
            self.audio = None

        print("✅ ウェイクワード検出を終了しました")

    def _recording_worker(self):
        """音声録音ワーカー"""
        while self.is_running and self.stream:
            try:
                data = self.stream.read(REALTIME_CONFIG["chunk"], exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                with self.buffer_lock:
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

            except Exception as e:
                if self.is_running:
                    print(f"❌ 録音エラー: {e}")
                break

    def _detection_worker(self):
        """ウェイクワード検出ワーカー"""
        while self.is_running:
            current_time = time.time()

            if current_time - self.last_processed_time < REALTIME_CONFIG["processing_interval"]:
                time.sleep(0.05)
                continue

            if current_time - self.last_detection_time < self.cooldown_seconds:
                time.sleep(0.05)
                continue

            with self.buffer_lock:
                buffer_duration = len(self.audio_buffer) / REALTIME_CONFIG["rate"]

                if buffer_duration < REALTIME_CONFIG["buffer_seconds"]:
                    time.sleep(0.05)
                    continue

                process_samples = int(REALTIME_CONFIG["buffer_seconds"] * REALTIME_CONFIG["rate"])
                overlap_samples = int(REALTIME_CONFIG["overlap_seconds"] * REALTIME_CONFIG["rate"])

                audio_to_process = self.audio_buffer[-process_samples:]

                keep_samples = overlap_samples
                if len(self.audio_buffer) > keep_samples:
                    self.audio_buffer = self.audio_buffer[-keep_samples:]

            try:
                self._process_audio_for_wake_word(audio_to_process)
                self.last_processed_time = current_time

            except Exception as e:
                if self.is_running:
                    print(f"❌ 検出エラー: {e}")

    def _process_audio_for_wake_word(self, audio_chunk):
        """音声チャンクからウェイクワードを検出"""
        audio_np = audio_chunk.astype(np.float32) / 32768.0

        volume = self._calculate_volume(audio_chunk)
        if volume < 100:
            return

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

        segments_list = list(segments)
        if not segments_list:
            return

        text = "".join(segment.text for segment in segments_list).strip()
        if not text:
            return

        confidence = self._calculate_simple_confidence(segments_list, info)

        if self._check_wake_word(text, confidence):
            detection_info = {
                'timestamp': time.time(),
                'text': text,
                'confidence': confidence,
                'duration': len(audio_chunk) / REALTIME_CONFIG["rate"]
            }
            self.detection_history.append(detection_info)

            if len(self.detection_history) > WAKE_WORD_CONFIG["max_detection_history"]:
                self.detection_history.pop(0)

            self.wake_word_callback(text, confidence)
            self.last_detection_time = time.time()

    def _check_wake_word(self, text, confidence):
        """テキストにウェイクワードが含まれているかチェック"""
        if not self.debug_mode and confidence < self.confidence_threshold:
            return False

        if len(text.strip()) < 1:
            return False

        if not any(ord(char) > 127 for char in text):
            return False

        text_lower = text.lower()

        if self.wake_word in text:
            return True

        wake_word_variants = [
            "シリウスくん", "シリウス", "しりうすくん", "しりうす",
            "シリウス くん", "しりうす くん", "シリウスさん", "しりうすさん",
            "ねえ，シリウスくん", "ねえシリウスくん", "ねえ、シリウスくん", "ねえ シリウスくん",
            "おい、シリウスくん", "おいシリウスくん", "ちょっと、シリウスくん", "ちょっとシリウスくん"
        ]

        for variant in wake_word_variants:
            if variant in text_lower:
                return True

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

            confidences = []
            for segment in segments:
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
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