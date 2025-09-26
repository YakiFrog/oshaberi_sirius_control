#!/usr/bin/env python3
"""
リアルタイム音声認識モジュール（録音バッファ方式）
PySide6 UI統合用に最適化
faster_whisper_test.pyと同じロジック
"""

import pyaudio
import numpy as np
import time
import threading
import queue
import tempfile
import wave
import os
from faster_whisper import WhisperModel

# モデル設定
MODEL_CONFIG = {
    "model_size": "small",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 6,
    "num_workers": 1
}

# 認識パラメータ
TRANSCRIBE_CONFIG = {
    "language": "ja",
    "beam_size": 1,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.0,
    "log_prob_threshold": -0.8,
    "no_speech_threshold": 0.4,
    "condition_on_previous_text": False,
    "initial_prompt": "以下は日本語の音声です。",
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 800,
        "speech_pad_ms": 50,
        "threshold": 0.5
    }
}

# 音声設定
AUDIO_CONFIG = {
    "chunk": 1024,
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 16000
}

class RealtimeRecognizer:
    def __init__(self, transcription_callback=None, silence_callback=None):
        self.model = None
        self.audio = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.transcription_callback = transcription_callback
        self.silence_callback = silence_callback

        # 沈黙検出設定
        self.last_voice_time = 0
        self.silence_threshold = 2.0  # 2秒
        self.voice_threshold = 100

    def _init_model(self):
        """モデルを遅延初期化（重複回避）"""
        if self.model is None:
            print("🚀 Faster Whisperモデルをロード中...")
            try:
                self.model = WhisperModel(
                    MODEL_CONFIG["model_size"],
                    device=MODEL_CONFIG["device"],
                    compute_type=MODEL_CONFIG["compute_type"],
                    cpu_threads=MODEL_CONFIG["cpu_threads"],
                    num_workers=MODEL_CONFIG["num_workers"]
                )
                print("✅ モデルロード完了")
            except Exception as e:
                print(f"❌ モデルロードエラー: {e}")
                self.model = None
                raise

    def start_recognition(self):
        """音声認識を開始（録音開始）"""
        if self.is_recording:
            return

        self._init_model()

        print("🎤 音声認識を開始します（録音中）...")

        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = self.audio.open(
            format=AUDIO_CONFIG["format"],
            channels=AUDIO_CONFIG["channels"],
            rate=AUDIO_CONFIG["rate"],
            input=True,
            frames_per_buffer=AUDIO_CONFIG["chunk"]
        )

        self.is_recording = True
        self.last_voice_time = time.time()

        # 録音スレッド開始
        recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        recording_thread.start()

        # 沈黙監視スレッド開始
        silence_thread = threading.Thread(target=self._silence_monitor, daemon=True)
        silence_thread.start()

    def stop_recognition(self):
        """音声認識を停止（録音終了して認識実行）"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio:
            self.audio.terminate()
            self.audio = None

        print("✅ 録音を終了しました。音声認識中...")

        # 録音データをファイルに保存して認識
        self._process_recording()

    def _recording_worker(self):
        """録音ワーカー"""
        while self.is_recording and self.stream:
            try:
                data = self.stream.read(AUDIO_CONFIG["chunk"], exception_on_overflow=False)
                self.frames.append(data)

                # 音量チェック
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    volume = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                    if volume > self.voice_threshold:
                        self.last_voice_time = time.time()

            except Exception as e:
                if self.is_recording:
                    print(f"❌ 録音エラー: {e}")
                break

    def _silence_monitor(self):
        """沈黙監視ワーカー"""
        while self.is_recording:
            current_time = time.time()
            silence_duration = current_time - self.last_voice_time

            if silence_duration >= self.silence_threshold:
                print(f"🔇 {silence_duration:.1f}秒の沈黙を検知しました")
                if self.silence_callback:
                    self.silence_callback()
                break

            time.sleep(0.1)

    def _process_recording(self):
        """録音データを処理して認識"""
        if not self.frames:
            print("❌ 録音データがありません")
            return

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(AUDIO_CONFIG["channels"])
            wf.setsampwidth(self.audio.get_sample_size(AUDIO_CONFIG["format"]) if self.audio else 2)
            wf.setframerate(AUDIO_CONFIG["rate"])
            wf.writeframes(b''.join(self.frames))
            wf.close()

        try:
            # ファイルサイズチェック
            file_size = os.path.getsize(temp_filename)
            print(f"音声ファイルサイズ: {file_size} bytes")

            if file_size < 1000:
                print("⚠️ 音声データが短すぎます")
                return

            # 音声認識
            print("🔄 音声認識処理開始...")
            segments, info = self.model.transcribe(
                temp_filename,
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
            print("✅ 音声認識処理完了")

            # 結果処理
            segments_list = list(segments)
            print(f"🔍 認識セグメント数: {len(segments_list)}")
            
            # テキストを安全に結合（エンコーディングエラー対応）
            text_parts = []
            for i, segment in enumerate(segments_list):
                segment_text = segment.text
                print(f"🔍 セグメント{i}: タイプ={type(segment_text)}, 内容={repr(segment_text)}")
                
                if isinstance(segment_text, bytes):
                    # バイト文字列の場合、UTF-8でデコード
                    try:
                        segment_text = segment_text.decode('utf-8', errors='ignore')
                        print(f"✅ バイトからデコード: {segment_text}")
                    except Exception as decode_error:
                        print(f"❌ デコードエラー: {decode_error}")
                        segment_text = str(segment_text, 'utf-8', errors='ignore')
                elif isinstance(segment_text, str):
                    # 文字列の場合、そのまま使用
                    print(f"✅ 文字列として処理: {segment_text}")
                else:
                    # その他の場合、文字列に変換
                    print(f"⚠️ その他の型から変換: {type(segment_text)}")
                    segment_text = str(segment_text)
                
                text_parts.append(segment_text)
            
            text = "".join(text_parts).strip()
            print(f"🎯 結合後テキスト: {repr(text)}")

            if text:
                confidence = self._calculate_confidence(segments_list, info)
                print(f"🎯 認識結果: 「{text}」 (確信度: {confidence:.1f}%)")

                if self.transcription_callback:
                    self.transcription_callback(text, confidence)
            else:
                print("🎯 認識結果: （無音または認識できませんでした）")

        except Exception as e:
            print(f"❌ 認識エラー: {e}")
        finally:
            # 一時ファイル削除
            try:
                os.unlink(temp_filename)
            except:
                pass

    def _calculate_confidence(self, segments, info):
        """信頼度計算"""
        try:
            word_confidences = []

            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        if hasattr(word, 'probability') and word.probability is not None:
                            confidence = min(100.0, max(0.0, (word.probability + 5.0) / 5.0 * 100))
                            word_confidences.append(confidence)

                if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                    segment_confidence = min(100.0, max(0.0, (segment.avg_logprob + 5.0) / 5.0 * 100))
                    word_confidences.append(segment_confidence)

            if word_confidences:
                return sum(word_confidences) / len(word_confidences)
            else:
                return info.language_probability * 100 if hasattr(info, 'language_probability') else 50.0

        except:
            return 50.0

    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            # 録音停止
            if self.is_recording:
                self.stop_recognition()
            
            # ストリームクリーンアップ
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                self.stream = None

            # PyAudioクリーンアップ
            if self.audio:
                try:
                    self.audio.terminate()
                except:
                    pass
                self.audio = None

            # フレームバッファクリア
            self.frames.clear()
            
            print("✅ RealtimeRecognizer リソースクリーンアップ完了")
        except Exception as e:
            print(f"⚠️ RealtimeRecognizer クリーンアップ警告: {e}")

    def __del__(self):
        """デストラクタでリソースをクリーンアップ"""
        self.cleanup()