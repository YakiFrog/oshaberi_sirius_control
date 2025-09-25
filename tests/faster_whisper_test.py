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
import threading
import time
import sys
import select
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

    def stop_recording(self):
        if not self.is_recording:
            print("録音中ではありません。")
            return None

        self.is_recording = False
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

    # Whisperモデルをロード（サイズはmediumで高精度）
    print("Whisperモデルをロード中...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("モデルロード完了")

    recorder = AudioRecorder()

    try:
        import threading
        import time
        import sys
        import select
        
        def recording_thread():
            """録音用スレッド"""
            while recorder.is_recording:
                recorder.record_chunk()
                time.sleep(0.01)  # 10ms待機
        
        while True:
            # 録音中でもコマンド入力を受け付ける
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                command = input("コマンド: ").strip().lower()
            else:
                command = input("コマンド: ").strip().lower()

            if command == "start":
                if not recorder.is_recording:
                    recorder.start_recording()
                    # 録音スレッドを開始
                    thread = threading.Thread(target=recording_thread, daemon=True)
                    thread.start()
                else:
                    print("すでに録音中です。")
                    
            elif command == "stop":
                if recorder.is_recording:
                    temp_file = recorder.stop_recording()
                    if temp_file:
                        # ファイルサイズをチェック
                        file_size = os.path.getsize(temp_file)
                        print(f"音声ファイルサイズ: {file_size} bytes")
                        
                        if file_size < 1000:  # 1KB未満の場合は警告
                            print("⚠️ 音声データが短すぎます。マイクが正しく動作していない可能性があります。")
                        
                        # 音声認識（sync_siriusface.pyを参考にしたパラメータチューニング）
                        print("音声認識処理中...")
                        segments, info = model.transcribe(
                            temp_file,
                            language="ja",              # 日本語指定
                            beam_size=5,                # ビームサーチサイズ（精度向上）
                            temperature=0.0,            # 決定論的出力（精度向上）
                            compression_ratio_threshold=2.4,  # 圧縮率閾値（ノイズ除去）
                            log_prob_threshold=-1.0,    # 確率閾値（低信頼度フィルタ）
                            no_speech_threshold=0.2,    # 無音判定閾値を緩く（0.6→0.2）
                            condition_on_previous_text=False,  # 前のテキストに依存しない
                            initial_prompt="以下は日本語の音声です。",  # 日本語コンテキスト
                            word_timestamps=True,       # 単語レベルの信頼度取得のため有効化
                            vad_filter=True,           # Voice Activity Detection（音声区間検出）
                            vad_parameters=dict(min_silence_duration_ms=250)  # 無音区間を短く（500→250ms）
                        )
                        
                        # セグメントからテキストを抽出
                        segments_list = list(segments)
                        text = "".join(segment.text for segment in segments_list).strip()
                        
                        # デバッグ情報を追加
                        print(f"セグメント数: {len(segments_list)}")
                        for i, segment in enumerate(segments_list):
                            print(f"  セグメント {i+1}: '{segment.text}' (開始: {segment.start:.2f}s, 終了: {segment.end:.2f}s)")
                        
                        # 信頼度情報を計算
                        confidence_info = calculate_confidence_metrics(segments_list, info)
                        
                        print(f"認識結果: {text}")
                        print(f"言語: {info.language} (確信度: {info.language_probability:.2f})")
                        print(f"音声時間: {info.duration:.2f}秒")
                        print(f"認識精度: {confidence_info['overall_confidence']:.1f}% (単語数: {confidence_info['word_count']})")
                        
                        # 一時ファイルを削除
                        os.unlink(temp_file)
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