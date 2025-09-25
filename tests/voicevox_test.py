from pprint import pprint
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# 1. Synthesizerの初期化
voicevox_onnxruntime_path = "voicevox_core/onnxruntime/lib/" + Onnxruntime.LIB_VERSIONED_FILENAME
open_jtalk_dict_dir = "voicevox_core/dict/open_jtalk_dic_utf_8-1.11"
synthesizer = Synthesizer(Onnxruntime.load_once(filename=voicevox_onnxruntime_path), OpenJtalk(open_jtalk_dict_dir))

# 2. 音声モデルの読み込み
with VoiceModelFile.open("voicevox_core/models/vvms/0.vvm") as model:
    synthesizer.load_voice_model(model)

# 3. テキスト音声合成
text = "サンプル音声です"
style_id = 0
wav = synthesizer.tts(text, style_id)
with open("output/output.wav", "wb") as f:
    f.write(wav)