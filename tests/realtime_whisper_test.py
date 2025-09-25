import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# モデル読み込み CPU推奨
model = WhisperModel("small", device="cpu")

# PyAudio初期設定
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

print("マイク入力中... Ctrl+Cで終了")

buffer = bytes()

try:
    while True:
        data = stream.read(1024)
        buffer += data
        
        # バッファが0.5秒分以上たまったら処理
        if len(buffer) > 16000 * 2 // 2:  # 16kHz * 2bytes * 0.5秒
            audio_np = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
            segments, _ = model.transcribe(audio_np, language="ja")

            for segment in segments:
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

            buffer = bytes()

except KeyboardInterrupt:
    print("終了します")

stream.stop_stream()
stream.close()
p.terminate()
