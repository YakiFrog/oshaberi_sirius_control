#!/usr/bin/env python3
"""
シンプルリップシンクテスト
VOICEVOX音声合成 + 音韻解析 + 口の形制御
"""

import os
import time as time_module
import threading
import urllib.request
import urllib.parse
import json
from pprint import pprint
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# VOICEVOX Core設定
VOICEVOX_ONNXRUNTIME_PATH = "voicevox_core/onnxruntime/lib/" + Onnxruntime.LIB_VERSIONED_FILENAME
OPEN_JTALK_DICT_DIR = "voicevox_core/dict/open_jtalk_dic_utf_8-1.11"
MODEL_PATH = "voicevox_core/models/vvms/0.vvm"

# シリウス表情制御API
SIRIUS_API_URL = "http://localhost:8080"

class LipSyncController:
    def __init__(self):
        # VOICEVOX初期化
        print("🚀 VOICEVOX初期化中...")
        self.synthesizer = Synthesizer(
            Onnxruntime.load_once(filename=VOICEVOX_ONNXRUNTIME_PATH), 
            OpenJtalk(OPEN_JTALK_DICT_DIR)
        )
        
        # 音声モデル読み込み
        with VoiceModelFile.open(MODEL_PATH) as model:
            self.synthesizer.load_voice_model(model)
        print("✅ VOICEVOX準備完了")

    def phoneme_to_mouth_shape(self, phoneme):
        """音韻から口の形にマッピング"""
        mouth_mapping = {
            # あ系
            'a': 'mouth_a',
            'A': 'mouth_a',
            # い系
            'i': 'mouth_i', 
            'I': 'mouth_i',
            'e': 'mouth_a',
            'E': 'mouth_a',
            # お系
            'o': 'mouth_o',
            'O': 'mouth_o',
            'u': 'mouth_o',
            'U': 'mouth_o',
            # 子音・無音
            'pau': None,
            'sil': None,
        }
        return mouth_mapping.get(phoneme, None)

    def set_mouth_pattern(self, pattern):
        """シリウスの口パターンを設定"""
        try:
            data = json.dumps({"mouth_pattern": pattern}).encode('utf-8')
            req = urllib.request.Request(
                f"{SIRIUS_API_URL}/mouth_pattern",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            print(f"📡 HTTPリクエスト送信: POST {SIRIUS_API_URL}/mouth_pattern")
            print(f"   データ: {{'mouth_pattern': '{pattern}'}}")
            
            with urllib.request.urlopen(req, timeout=0.5) as response:
                status_code = response.getcode()
                # レスポンス詳細は省略（統計情報に集中）
                return status_code == 200
        except Exception as e:
            print(f"❌ HTTPリクエストエラー: {e}")
            return False

    def play_audio(self, wav_data):
        """音声を再生（macOS）"""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name
            
            os.system(f"afplay {temp_file_path}")
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"❌ 音声再生エラー: {e}")

    def speak_with_lipsync(self, text, style_id=0):
        """音声合成 + リップシンク"""
        print(f"🎤 合成: 「{text}」")
        print(f"📏 文字数: {len(text)}文字")
        
        # 1. 音声合成（シンプル版）
        wav_data = self.synthesizer.tts(text, style_id)
        
        # 2. シンプルな音韻解析（文字ベース）
        mouth_sequence = self.text_to_mouth_sequence(text)
        
        print("📝 口パターンシーケンス:")
        mouth_pattern_count = 0
        none_pattern_count = 0
        for i, (time, shape, duration) in enumerate(mouth_sequence[:15]):  # 最初の15個まで表示
            print(f"  {time:.2f}s: {shape} ({duration:.2f}s)")
            if shape is not None:
                mouth_pattern_count += 1
            else:
                none_pattern_count += 1
        
        if len(mouth_sequence) > 15:
            remaining = len(mouth_sequence) - 15
            for time, shape, duration in mouth_sequence[15:]:
                if shape is not None:
                    mouth_pattern_count += 1
                else:
                    none_pattern_count += 1
            print(f"  ... 他{remaining}個")
        
        print(f"📊 統計: 総パターン数{len(mouth_sequence)}, 口パターン{mouth_pattern_count}個, None{none_pattern_count}個")
        print(f"📊 比率: 文字数{len(text)} vs パターン数{len(mouth_sequence)} = {len(mouth_sequence)/len(text):.2f}倍")
        
        # 3. 音声再生開始
        audio_thread = threading.Thread(target=self.play_audio, args=(wav_data,))
        audio_thread.daemon = True
        audio_thread.start()
        
        # 4. リップシンク実行
        start_time = time_module.time()
        
        for seq_time, mouth_shape, duration in mouth_sequence:
            # タイミング待機
            elapsed = time_module.time() - start_time
            wait_time = seq_time - elapsed
            if wait_time > 0:
                time_module.sleep(wait_time)
            
            # 口パターン設定
            self.set_mouth_pattern(mouth_shape)
            
            # デバッグ出力
            if mouth_shape:
                print(f"👄 {seq_time:.2f}s: {mouth_shape}")
        
        # 5. 終了時に口をリセット
        time_module.sleep(0.5)
        self.set_mouth_pattern(None)
        print("✅ 発話完了\n")

    def text_to_mouth_sequence(self, text):
        """テキストから口の動きシーケンスを生成（簡易版）"""
        sequence = []
        current_time = 0.0
        char_duration = 0.15  # 1文字あたりの時間
        
        for char in text:
            mouth_shape = self.char_to_mouth_shape(char)
            sequence.append((current_time, mouth_shape, char_duration))
            current_time += char_duration
        
        return sequence

    def char_to_mouth_shape(self, char):
        """文字から口の形を推定"""
        # ひらがな・カタカナの母音判定
        a_sounds = 'あかがさざただなはばぱまやらわアカガサザタダナハバパマヤラワ'
        i_sounds = 'いきぎしじちぢにひびぴみりイキギシジチヂニヒビピミリ'
        o_sounds = 'おこごそぞとどのほぼぽもよろをンおこごそぞとどのほぼぽもよろをンオコゴソゾトドノホボポモヨロヲン'
        
        if char in a_sounds:
            return 'mouth_a'
        elif char in i_sounds:
            return 'mouth_i' 
        elif char in o_sounds:
            return 'mouth_o'
        else:
            return None

def main():
    print("🎭 シンプルリップシンクテスト")
    print("=" * 40)
    
    # リップシンクコントローラー初期化
    controller = LipSyncController()
    
    # テストセリフ
    test_phrases = [
        # "こんにちは、僕の名前はシリウスです",
        # "今日はいい天気ですね",
        # "ありがとうございます",
        # "さようなら"
        "今日恐々恐々"
    ]
    
    print("🎤 テスト開始（シリウスの表情サーバーが起動している必要があります）")
    print(f"📡 API URL: {SIRIUS_API_URL}")
    
    try:
        for i, text in enumerate(test_phrases, 1):
            print(f"\n--- テスト {i}/{len(test_phrases)} ---")
            controller.speak_with_lipsync(text)
            
            if i < len(test_phrases):
                print("⏳ 3秒待機...")
                time_module.sleep(3)
        
        print("\n🎉 全テスト完了！")
        
    except KeyboardInterrupt:
        print("\n🛑 テスト中止")
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()
