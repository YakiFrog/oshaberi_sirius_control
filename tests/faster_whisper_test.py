#!/usr/bin/env python3
"""
Faster Whisper 音声認識テストプログラム
コンソールで操作（start: 録音開始, stop: 録音終了・認識, quit: 終了）
"""

import os
import tempfile
import pyaudio
import wave
import numpy as np
import time
import threading
from faster_whisper import WhisperModel

# 設定
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper推奨の16kHz

def calculate_confidence_metrics(segments, info):
    """セグメントから信頼度メトリクスを計算"""
    try:
        word_confidences = []
        word_count = 0
        total_duration = 0
        
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                # 単語レベルの信頼度を取得
                for word in segment.words:
                    if hasattr(word, 'probability') and word.probability is not None:
                        # 対数確率を信頼度パーセンテージに変換
                        confidence = min(100.0, max(0.0, (word.probability + 5.0) / 5.0 * 100))
                        word_confidences.append(confidence)
                        word_count += 1
        
            # セグメントレベルの情報
            if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                # 平均対数確率を信頼度に変換
                segment_confidence = min(100.0, max(0.0, (segment.avg_logprob + 5.0) / 5.0 * 100))
                word_confidences.append(segment_confidence)
        
            total_duration += getattr(segment, 'end', 0) - getattr(segment, 'start', 0)
        
        # 全体的な信頼度を計算
        if word_confidences:
            overall_confidence = sum(word_confidences) / len(word_confidences)
            min_confidence = min(word_confidences)
            max_confidence = max(word_confidences)
            std_confidence = (sum((x - overall_confidence) ** 2 for x in word_confidences) / len(word_confidences)) ** 0.5
        else:
            # フォールバック: 言語確率を使用
            overall_confidence = info.language_probability * 100 if hasattr(info, 'language_probability') else 50.0
            min_confidence = max_confidence = overall_confidence
            std_confidence = 0.0
            word_count = len(segments)
        
        return {
            'overall_confidence': overall_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'std_confidence': std_confidence,
            'word_count': word_count,
            'segment_count': len(segments),
            'audio_duration': getattr(info, 'duration', total_duration),
            'language_probability': getattr(info, 'language_probability', 0.0) * 100,
            'word_confidences': word_confidences
        }
        
    except Exception as e:
        print(f"⚠️ 信頼度計算エラー: {e}")
        # エラー時のデフォルト値
        return {
            'overall_confidence': 50.0,
            'min_confidence': 50.0,
            'max_confidence': 50.0,
            'std_confidence': 0.0,
            'word_count': 0,
            'segment_count': len(segments) if segments else 0,
            'audio_duration': 0.0,
            'language_probability': 50.0,
            'word_confidences': []
        }

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        if self.is_recording:
            print("すでに録音中です。")
            return

        self.frames = []
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.is_recording = True
        print("録音を開始しました。'stop' と入力して終了してください。")
        
        # 録音開始後、バックグラウンドで音声データを継続取得
        import threading
        self.recording_thread = threading.Thread(target=self._continuous_recording, daemon=True)
        self.recording_thread.start()

    def _continuous_recording(self):
        """バックグラウンドで録音を継続する"""
        frame_count = 0
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                frame_count += 1
                
                # 音声レベルをチェック
                import numpy as np
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    volume = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                    if np.isnan(volume):
                        volume = 0
                else:
                    volume = 0
                    
                if frame_count % 10 == 0:  # 10フレームごとに表示
                    print(f"録音中... フレーム数: {frame_count}, 音量: {volume:.0f}")
            except Exception as e:
                print(f"録音エラー: {e}")
                break
            import time
            time.sleep(0.01)  # 10ms待機

    def stop_recording(self):
        if not self.is_recording:
            print("録音中ではありません。")
            return None

        self.is_recording = False
        
        # スレッドの終了を待つ
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        self.stream.stop_stream()
        self.stream.close()

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()

        print("録音を終了しました。音声認識中...")
        return temp_filename

    def record_chunk(self):
        if self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                # 音声レベルをチェック
                import numpy as np
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    volume = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                    if np.isnan(volume):
                        volume = 0
                else:
                    volume = 0
                if len(self.frames) % 10 == 0:  # 10フレームごとに表示
                    print(f"録音中... フレーム数: {len(self.frames)}, 音量: {volume:.0f}")
            except Exception as e:
                print(f"録音エラー: {e}")

    def close(self):
        self.audio.terminate()

def list_audio_devices():
    """利用可能な音声デバイスを表示"""
    print("\n🎤 利用可能な音声入力デバイス:")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  {i}: {info['name']} (チャンネル数: {info['maxInputChannels']}, サンプルレート: {info['defaultSampleRate']}Hz)")
    p.terminate()
    print()

def main():
    print("🎤 Faster Whisper 音声認識テスト")
    print("コマンド: start (録音開始), stop (録音終了・認識), quit (終了)")
    print("-" * 50)
    
    # 音声デバイス一覧を表示
    list_audio_devices()

    # Whisperモデルをロード（高速化設定）
    print("Whisperモデルをロード中...")
    model = WhisperModel(
        "medium",  # ここはゆずれない
        device="cpu",
        compute_type="int8",
        cpu_threads=8,  # CPUスレッド数を8に増加（10コア中）
        num_workers=1
    )
    print("モデルロード完了")

    recorder = AudioRecorder()

    try:
        import time
        
        while True:
            command = input("コマンド: ").strip().lower()

            if command == "start":
                if not recorder.is_recording:
                    recorder.start_recording()
                    print("録音中です... 'stop'と入力して終了してください")
                else:
                    print("すでに録音中です。")
                    
            elif command == "stop":
                if recorder.is_recording:
                    # stopコマンド入力時の時間を記録
                    stop_command_time = time.time()
                    
                    temp_file = recorder.stop_recording()
                    if temp_file:
                        # ファイルサイズをチェック
                        file_size = os.path.getsize(temp_file)
                        print(f"音声ファイルサイズ: {file_size} bytes")
                        
                        if file_size < 1000:  # 1KB未満の場合は警告
                            print("⚠️ 音声データが短すぎます。マイクが正しく動作していない可能性があります。")
                        
                        # 音声認識（sync_siriusface.pyを参考にしたパラメータチューニング）
                        print("音声認識処理中...")
                        
                        try:
                            segments, info = model.transcribe(
                                temp_file,
                                language="ja",              # 日本語指定
                                beam_size=1,                # ビームサーチを1に設定（greedy decoding）で最大速度
                                temperature=0.0,            # 決定論的出力
                                compression_ratio_threshold=2.0,  # 圧縮率閾値を緩く（2.4→2.0）で処理軽減
                                log_prob_threshold=-0.8,    # 確率閾値を緩く（-1.0→-0.8）で高速化
                                no_speech_threshold=0.4,    # 無音判定をさらに緩く（0.3→0.4）
                                condition_on_previous_text=False,
                                initial_prompt="以下は日本語の音声です。",
                                word_timestamps=False,
                                vad_filter=True,
                                vad_parameters=dict(
                                    min_silence_duration_ms=800,  # 無音区間を長く（500→800ms）で処理軽減
                                    speech_pad_ms=50,       # 音声パディングをさらに短縮
                                    threshold=0.5           # VAD閾値を高くしてより積極的なフィルタリング
                                )
                            )
                            
                            # セグメントからテキストを抽出
                            segments_list = list(segments)
                            text = "".join(segment.text for segment in segments_list).strip()
                            
                            # stopコマンドから認識完了までの時間を計算
                            recognition_end = time.time()
                            total_time_from_stop = recognition_end - stop_command_time
                            
                            # 簡潔なデバッグ情報（速度優先）
                            print(f"セグメント数: {len(segments_list)}")
                            
                            # 信頼度情報を計算（簡略化）
                            confidence_info = calculate_confidence_metrics(segments_list, info)
                            
                            print(f"認識結果: {text}")
                            print(f"言語: {info.language} (確信度: {info.language_probability:.2f})")
                            print(f"音声時間: {info.duration:.2f}秒")
                            print(f"認識精度: {confidence_info['overall_confidence']:.1f}% (単語数: {confidence_info['word_count']})")
                            print(f"⏱️  stopコマンドから完了まで: {total_time_from_stop:.2f}秒")
                            
                            # 処理効率の計算
                            if info.duration > 0:
                                processing_ratio = total_time_from_stop / info.duration
                                if processing_ratio < 1.0:
                                    print(f"⚡ リアルタイム係数: {processing_ratio:.2f}x (リアルタイムより {1/processing_ratio:.1f}倍高速)")
                                else:
                                    print(f"🐌 リアルタイム係数: {processing_ratio:.2f}x (リアルタイムより {processing_ratio:.1f}倍低速)")
                            
                        except Exception as transcribe_error:
                            print(f"❌ 音声認識エラー: {transcribe_error}")
                        
                        # 一時ファイルを削除
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    print("-" * 50)
                else:
                    print("録音中ではありません。")
                    
            elif command == "quit":
                if recorder.is_recording:
                    recorder.stop_recording()
                print("プログラムを終了します。")
                break
            else:
                print("無効なコマンドです。start, stop, quit のいずれかを入力してください。")

    except KeyboardInterrupt:
        print("\nプログラムを終了します。")
    finally:
        recorder.close()

if __name__ == "__main__":
    main()