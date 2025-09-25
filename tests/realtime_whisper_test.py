#!/usr/bin/env python3
"""
リアルタイムFaster Whisper音声認識プログラム
faster_whisper_test.pyの最適化設定を適用したリアルタイム版
"""

import pyaudio
import numpy as np
import time
import threading
import queue
from faster_whisper import WhisperModel

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
    "beam_size": 1,                # リアルタイム用にgreedy decoding
    "temperature": 0.0,
    "compression_ratio_threshold": 2.0,
    "log_prob_threshold": -0.8,
    "no_speech_threshold": 0.4,
    "condition_on_previous_text": True,   # リアルタイム用に文脈を維持
    "initial_prompt": "以下は日本語の音声です。",
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 500,   # リアルタイム用に短めに
        "speech_pad_ms": 30,
        "threshold": 0.5
    }
}

# リアルタイム設定
REALTIME_CONFIG = {
    "chunk": 1024,
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 16000,
    "buffer_seconds": 1.0,         # 1秒分のバッファ
    "overlap_seconds": 0.2,        # 0.2秒のオーバーラップ
    "processing_interval": 0.5     # 0.5秒ごとに処理
}

# =============================================================================
# リアルタイム音声認識クラス
# =============================================================================

class RealtimeWhisper:
    def __init__(self):
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

    def start_realtime_recognition(self):
        """リアルタイム認識を開始"""
        print("🎤 リアルタイム音声認識を開始します...")
        print("💡 話しかけると自動で認識されます（Ctrl+Cで終了）")

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

        # 処理スレッド開始
        processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        processing_thread.start()

        try:
            # メインスレッドは待機
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 終了します...")
            self.stop_realtime_recognition()

    def stop_realtime_recognition(self):
        """リアルタイム認識を停止"""
        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        print("✅ リアルタイム認識を終了しました")

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

    def _processing_worker(self):
        """音声処理ワーカー"""
        print("🧠 処理スレッド開始")

        while self.is_running:
            current_time = time.time()

            # 処理間隔チェック
            if current_time - self.last_processed_time < REALTIME_CONFIG["processing_interval"]:
                time.sleep(0.05)  # 短い待機
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

            # 音声認識処理
            try:
                self._process_audio_chunk(audio_to_process)
                self.last_processed_time = current_time

            except Exception as e:
                print(f"❌ 処理エラー: {e}")

        print("\n🔄 処理スレッド終了")

    def _process_audio_chunk(self, audio_chunk):
        """音声チャンクを処理"""
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

        # 結果を表示
        segments_list = list(segments)
        if segments_list:
            text = "".join(segment.text for segment in segments_list).strip()
            if text:  # 空でないテキストのみ表示
                confidence = self._calculate_simple_confidence(segments_list, info)
                duration = len(audio_chunk) / REALTIME_CONFIG["rate"]
                print(f"\n🎯 [{duration:.1f}s] {text} (確信度: {confidence:.1f}%)")

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

# =============================================================================
# メイン関数
# =============================================================================

def main():
    print("🎤 リアルタイム Faster Whisper 音声認識")
    print("=" * 50)
    print(f"📊 モデル: {MODEL_CONFIG['model_size']} ({MODEL_CONFIG['compute_type']})")
    print(f"⚡ CPUスレッド: {MODEL_CONFIG['cpu_threads']}")
    print(f"⏱️  処理間隔: {REALTIME_CONFIG['processing_interval']}秒")
    print(f"🎵 バッファ: {REALTIME_CONFIG['buffer_seconds']}秒")
    print("=" * 50)

    # リアルタイム認識インスタンス作成
    recognizer = RealtimeWhisper()

    try:
        # リアルタイム認識開始
        recognizer.start_realtime_recognition()

    except KeyboardInterrupt:
        print("\n🛑 プログラムを終了します")
    except Exception as e:
        print(f"❌ エラー: {e}")
    finally:
        recognizer.stop_realtime_recognition()

if __name__ == "__main__":
    main()
