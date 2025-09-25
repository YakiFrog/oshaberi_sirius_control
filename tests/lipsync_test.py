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
try:
    import pykakasi
except ImportError:
    pykakasi = None
    print("⚠️  pykakasiがインストールされていません。漢字の読み変換は利用できません。")

try:
    import requests
    from typing import Optional
except ImportError:
    requests = None
    Optional = object  # ダミーオブジェクト
    print("⚠️  requestsがインストールされていません。おしゃべりモード制御は利用できません。")

# VOICEVOX Core設定
VOICEVOX_ONNXRUNTIME_PATH = "voicevox_core/onnxruntime/lib/" + Onnxruntime.LIB_VERSIONED_FILENAME
OPEN_JTALK_DICT_DIR = "voicevox_core/dict/open_jtalk_dic_utf_8-1.11"
MODEL_PATH = "voicevox_core/models/vvms/13.vvm"  # 13.vvmを使用

# シリウス表情制御API
SIRIUS_API_URL = "http://localhost:8080"

class AudioQueryPhonemeAnalyzer:
    """AudioQueryから音韻情報を抽出してリップシンク用に変換"""
    
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        
        # 日本語音韻から口形状への詳細マッピング
        self.phoneme_to_mouth = {
            # 母音
            'a': 'a',    # あ
            'i': 'i',    # い
            'u': 'o',    # う（oに統合）
            'e': 'a',    # え（aに近い）
            'o': 'o',    # お
            
            # 子音（口の形に影響を与えるもの）
            'k': 'a',    # か行（aに近い）
            'g': 'a',    # が行
            's': 'i',    # さ行（iに近い）
            'z': 'i',    # ざ行
            't': 'a',    # た行
            'd': 'a',    # だ行
            'n': 'o',    # な行（oに近い）
            'h': 'o',    # は行
            'b': 'o',    # ば行
            'p': 'o',    # ぱ行
            'm': 'o',    # ま行
            'y': 'a',    # や行
            'r': 'a',    # ら行
            'w': 'o',    # わ行
            'f': 'o',    # ふ
            'v': 'o',    # ヴ
            'ch': 'i',   # ち（iに近い）
            'sh': 'i',   # し
            'j': 'i',    # じ
            'ts': 'a',   # つ
            
            # 子音＋母音の組み合わせ
            # あ系
            'ka': 'a', 'ga': 'a', 'sa': 'a', 'za': 'a', 'ta': 'a', 'da': 'a',
            'na': 'a', 'ha': 'a', 'ba': 'a', 'pa': 'a', 'ma': 'a', 'ya': 'a',
            'ra': 'a', 'wa': 'a', 'fa': 'a', 'va': 'a',
            
            # い系  
            'ki': 'i', 'gi': 'i', 'si': 'i', 'shi': 'i', 'zi': 'i', 'ji': 'i',
            'ti': 'i', 'chi': 'i', 'di': 'i', 'ni': 'i', 'hi': 'i', 'bi': 'i',
            'pi': 'i', 'mi': 'i', 'ri': 'i', 'wi': 'i', 'fi': 'i', 'vi': 'i',
            
            # う系（oに統合）
            'ku': 'o', 'gu': 'o', 'su': 'o', 'zu': 'o', 'tu': 'o', 'tsu': 'o',
            'du': 'o', 'nu': 'o', 'hu': 'o', 'fu': 'o', 'bu': 'o', 'pu': 'o',
            'mu': 'o', 'yu': 'o', 'ru': 'o', 'wu': 'o',
            
            # え系（aに近い）
            'ke': 'a', 'ge': 'a', 'se': 'a', 'ze': 'a', 'te': 'a', 'de': 'a',
            'ne': 'a', 'he': 'a', 'be': 'a', 'pe': 'a', 'me': 'a', 're': 'a',
            'we': 'a', 'fe': 'a', 've': 'a',
            
            # お系
            'ko': 'o', 'go': 'o', 'so': 'o', 'zo': 'o', 'to': 'o', 'do': 'o',
            'no': 'o', 'ho': 'o', 'bo': 'o', 'po': 'o', 'mo': 'o', 'yo': 'o',
            'ro': 'o', 'wo': 'o', 'fo': 'o', 'vo': 'o',
            
            # 特殊音韻
            'sil': None,    # 無音
            'pau': None,    # ポーズ
            'cl': None,     # 閉鎖音
            'q': None,      # 促音
            'N': 'o',       # ん
            
            # 長音・その他
            'ー': None,
            'っ': None,
            ',': None,      # 句読点
            '、': None,
            '。': None,
            '.': None,
            ' ': None,      # スペース
        }
    
    def analyze_from_audio_query(self, text: str, style_id: int, speed_scale: float = 1.0):
        """AudioQueryから音韻情報を抽出してリップシンク用に変換"""
        try:
            print(f"🔍 AudioQuery音韻解析開始: '{text}' (速度: {speed_scale}x)")
            
            # AudioQueryを作成
            audio_query = self.synthesizer.create_audio_query(text, style_id)
            
            # 速度スケールを設定（AudioQueryオブジェクトに直接設定）
            if hasattr(audio_query, 'speed_scale'):
                audio_query.speed_scale = speed_scale
                print(f"✅ 速度スケールを設定: {speed_scale}x")
            else:
                print(f"⚠️  AudioQueryにspeed_scale属性が見つからないため、デフォルト速度を使用")
            
            phoneme_timeline = []
            current_time = 0.0
            
            # accent_phrasesから音韻情報を抽出
            if hasattr(audio_query, 'accent_phrases'):
                for accent_phrase in audio_query.accent_phrases:
                    if hasattr(accent_phrase, 'moras'):
                        for mora in accent_phrase.moras:
                            # 子音処理
                            if hasattr(mora, 'consonant') and mora.consonant:
                                consonant_phoneme = mora.consonant
                                consonant_duration = getattr(mora, 'consonant_length', 0.1) or 0.1
                                
                                # 速度スケールを適用
                                consonant_duration /= speed_scale
                                
                                mouth_shape = self.phoneme_to_mouth.get(consonant_phoneme, 'a')  # デフォルトを'a'に
                                phoneme_timeline.append((current_time, mouth_shape, consonant_duration))
                                current_time += consonant_duration
                            
                            # 母音処理
                            if hasattr(mora, 'vowel') and mora.vowel:
                                vowel_phoneme = mora.vowel
                                vowel_duration = getattr(mora, 'vowel_length', 0.1) or 0.1
                                
                                # 速度スケールを適用
                                vowel_duration /= speed_scale
                                
                                mouth_shape = self.phoneme_to_mouth.get(vowel_phoneme, 'a')
                                phoneme_timeline.append((current_time, mouth_shape, vowel_duration))
                                current_time += vowel_duration
                
                # ポーズ処理
                if hasattr(accent_phrase, 'pause_mora') and accent_phrase.pause_mora:
                    pause_duration = getattr(accent_phrase.pause_mora, 'vowel_length', 0.0) or 0.0
                    if pause_duration > 0:
                        # 速度スケールを適用
                        pause_duration /= speed_scale
                        phoneme_timeline.append((current_time, None, pause_duration))
                        current_time += pause_duration
            
            print(f"✅ AudioQuery音韻解析完了: {len(phoneme_timeline)}音韻, 総時間: {current_time:.2f}秒")
            return phoneme_timeline
            
        except Exception as e:
            print(f"❌ AudioQuery音韻解析エラー: {e}")
            return []
    
    def get_mouth_shape_sequence(self, text: str, style_id: int, speed_scale: float = 1.0):
        """テキストから口形状シーケンスを生成（AudioQuery版）"""
        phoneme_timeline = self.analyze_from_audio_query(text, style_id, speed_scale)
        mouth_sequence = []
        
        for time_pos, mouth_shape, duration in phoneme_timeline:
            if mouth_shape:  # Noneでない場合のみ追加
                mouth_sequence.append((time_pos, mouth_shape, duration))
            else:
                print(f"⚠️  無音音韻をスキップ: {time_pos:.2f}s (duration: {duration:.2f}s)")
        
        return mouth_sequence

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
        """音声合成 + リップシンク（AudioQuery使用版）"""
        print(f"🎤 合成: 「{text}」")
        print(f"📏 文字数: {len(text)}文字")
        
        # 1. AudioQueryで音韻情報を取得
        try:
            audio_query = self.synthesizer.create_audio_query(text, style_id)
            print("✅ AudioQuery取得成功")
        except Exception as e:
            print(f"❌ AudioQuery取得エラー: {e}")
            print("🔄 文字ベース解析にフォールバック")
            return self.speak_with_lipsync_fallback(text, style_id)
        
        # AudioQuery音韻解析を使用
        try:
            mouth_sequence = self.analyzer.get_mouth_shape_sequence(text, style_id)
            print("✅ AudioQuery音韻解析成功")
        except Exception as e:
            print(f"❌ AudioQuery音韻解析エラー: {e}")
            print("🔄 文字ベース解析にフォールバック")
            return self.speak_with_lipsync_fallback(text, style_id)
        
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
        
        # 3. 音声合成
        wav_data = self.synthesizer.synthesis(audio_query, style_id)
        
        # 4. 音声再生開始
        audio_thread = threading.Thread(target=self.play_audio, args=(wav_data,))
        audio_thread.daemon = True
        audio_thread.start()
        
        # 5. リップシンク実行
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
        
        # 6. 終了時に口をリセット（元の表情の自然な口パターンに戻す）
        time_module.sleep(0.2)
        
        # 明示的に口パターンをNoneに設定して表情の自然な口の形に戻る
        success = self.set_mouth_pattern(None)
        if success:
            print("✅ 発話完了（口を元の表情の自然な口パターンに戻しました）")
        else:
            print("⚠️ 口パターンのリセットに失敗しました")
        print()

    def audioquery_to_mouth_sequence(self, audio_query):
        """AudioQueryから口の動きシーケンスを生成（廃止予定）"""
        # このメソッドは使われていないので、get_mouth_shape_sequenceを使う
        return self.analyzer.get_mouth_shape_sequence("", 0)  # ダミー

    def speak_with_lipsync_fallback(self, text, style_id=0):
        """フォールバック: 文字ベースリップシンク"""
        print(f"🔄 フォールバック処理: 「{text}」")
        
        # 1. 音声合成（シンプル版）
        wav_data = self.synthesizer.tts(text, style_id)
        
        # 2. シンプルな音韻解析（文字ベース）
        mouth_sequence = self.text_to_mouth_sequence(text)
        
        print("📝 口パターンシーケンス (フォールバック):")
        mouth_pattern_count = 0
        none_pattern_count = 0
        for i, (time, shape, duration) in enumerate(mouth_sequence[:15]):
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
        
        # 5. 終了時に口をリセット（元の表情の自然な口パターンに戻す）
        time_module.sleep(0.2)
        success = self.set_mouth_pattern(None)
        if success:
            print("✅ 発話完了（口を元の表情の自然な口パターンに戻しました）")
        else:
            print("⚠️ 口パターンのリセットに失敗しました")
        print()

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
        """文字から口の形を推定（pykakasi漢字読み対応版）"""
        # 漢字の場合はpykakasiで読みに変換
        if self.kakasi_converter and self._is_kanji(char):
            try:
                # 新しいpykakasiのAPI使用
                converted = self.kakasi_converter.convert(char)
                if converted:
                    # 変換結果から'hira'（ひらがな）を取得
                    hiragana_reading = ''.join([item['hira'] for item in converted])
                    if hiragana_reading and hiragana_reading != char:
                        # 読みの最初の文字で口の形を判定
                        first_char = hiragana_reading[0]
                        print(f"🔤 漢字変換: '{char}' → '{hiragana_reading}' → 判定文字:'{first_char}'")
                        return self._hiragana_to_mouth_shape(first_char)
            except Exception as e:
                print(f"⚠️  漢字読み変換エラー: {char} - {e}")
        
        # ひらがな・カタカナ・その他の処理
        return self._hiragana_to_mouth_shape(char)
    
    def _is_kanji(self, char):
        """文字が漢字かどうか判定"""
        return '\u4e00' <= char <= '\u9faf'
    
    def _hiragana_to_mouth_shape(self, char):
        """ひらがな・カタカナから口の形を判定"""
        # ひらがな・カタカナの母音判定
        a_sounds = 'あかがさざただなはばぱまやらわアカガサザタダナハバパマヤラワ'
        i_sounds = 'いきぎしじちぢにひびぴみりイキギシジチヂニヒビピミリ'
        o_sounds = 'うえおこごそぞとどのほぼぽもよろをンウエオコゴソゾトドノホボポモヨロヲン'
        
        if char in a_sounds:
            return 'mouth_a'
        elif char in i_sounds:
            return 'mouth_i' 
        elif char in o_sounds:
            return 'mouth_o'
        else:
            return None

class TalkingModeController:
    """おしゃべりモード制御クラス"""
    
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.is_talking_mode_active = False
        self.last_mouth_pattern = None  # 冗長リクエストを防ぐ
        
        # 高速化のためのHTTPセッション設定
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'  # Keep-Aliveを有効にする
        })
        
        # コネクションプールの設定
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,  # プール内の接続数
            pool_maxsize=1,      # プールの最大サイズ
            max_retries=0        # リトライしない（高速化のため）
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def set_talking_mode(self, enabled: bool) -> bool:
        """おしゃべりモードを設定"""
        if self.is_talking_mode_active == enabled:
            return True  # ログ出力も省略して高速化
        
        try:
            response = self.session.post(
                f"{self.server_url}/talking_mouth_mode",
                json={'talking_mouth_mode': enabled},
                timeout=0.1  # タイムアウトを極短に
            )
            
            if response.status_code == 200:
                self.is_talking_mode_active = enabled
                return True
            else:
                print(f"❌ おしゃべりモード設定失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ おしゃべりモード設定エラー: {e}")
            return False
    
    def set_mouth_pattern_fast(self, pattern) -> bool:
        """高速口形状設定（冗長リクエスト排除）"""
        # 同じパターンの場合はスキップ（ただし、Noneの場合は必ず実行）
        if self.last_mouth_pattern == pattern and pattern is not None:
            print(f"🔧 同じ口パターン ({pattern}) のためスキップ")
            return True
        
        try:
            print(f"🔧 口パターン設定リクエスト: {pattern}")
            response = self.session.post(
                f"{self.server_url}/mouth_pattern",
                json={'mouth_pattern': pattern},
                timeout=0.1  # タイムアウトを少し長くして確実に処理
            )
            
            if response.status_code == 200:
                self.last_mouth_pattern = pattern
                print(f"✅ 口パターン設定成功: {pattern}")
                return True
            else:
                print(f"❌ 口パターン設定失敗: HTTP {response.status_code}, レスポンス: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 口パターン設定エラー: {e}")
            return False
    
    def reset_to_neutral(self):
        """main.pyのリセット機能を使用して全設定をリセット"""
        try:
            response = self.session.post(
                f"{self.server_url}/api/reset",
                json={},
                timeout=0.1
            )
            
            if response.status_code == 200:
                self.is_talking_mode_active = False
                self.last_mouth_pattern = None
                print("🔄 main.pyリセット機能により全設定をリセットしました")
                return True
            else:
                print(f"❌ リセット失敗: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ リセット機能エラー: {e}")
            return False
    
    def cleanup_session(self):
        """セッションのクリーンアップ（メモリリーク防止）"""
        try:
            self.session.close()
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            })
            # アダプターも再設定
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1,
                pool_maxsize=1,
                max_retries=0
            )
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
            self.last_mouth_pattern = None
        except Exception as e:
            print(f"セッションクリーンアップエラー: {e}")

class LipSyncController:
    def __init__(self):
        # VOICEVOX初期化
        print("🚀 VOICEVOX初期化中...")
        self.synthesizer = Synthesizer(
            Onnxruntime.load_once(filename=VOICEVOX_ONNXRUNTIME_PATH), 
            OpenJtalk(OPEN_JTALK_DICT_DIR)
        )
        
        # 音声モデル読み込み（13.vvmを使用）
        with VoiceModelFile.open(MODEL_PATH) as model:
            self.synthesizer.load_voice_model(model)
        print("✅ VOICEVOX準備完了")
        
        # pykakasi初期化（漢字読み変換用）
        self.kakasi_converter = None
        if pykakasi:
            try:
                # 新しいpykakasiのAPI使用
                kks = pykakasi.kakasi()
                self.kakasi_converter = kks
                print("✅ pykakasi漢字読み変換準備完了")
            except Exception as e:
                print(f"⚠️  pykakasi初期化エラー: {e}")
                self.kakasi_converter = None
        
        # AudioQuery音韻解析器を初期化
        self.analyzer = AudioQueryPhonemeAnalyzer(self.synthesizer)
        
        # おしゃべりモードコントローラーを初期化
        self.talking_controller = TalkingModeController() if requests else None
        
        # 音声パラメータ（ハードコードされた設定）
        self.style_id = 54
        self.speed_scale = 1.0
        self.pitch_scale = 0.0
        self.intonation_scale = 0.9
        
        print(f"✅ 音声設定: style_id={self.style_id}, speed={self.speed_scale}, pitch={self.pitch_scale}, intonation={self.intonation_scale}")
    
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

    def play_audio_precise(self, wav_data, start_event):
        """音声を再生（精密同期版）"""
        try:
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name
            
            # 再生開始を通知
            start_event.set()
            
            # afplayで再生
            process = subprocess.Popen(['afplay', temp_file_path], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
            process.wait()
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"❌ 音声再生エラー: {e}")

    def get_current_mouth_pattern(self):
        """現在のシリウス口パターンを取得"""
        try:
            req = urllib.request.Request(f"{SIRIUS_API_URL}/mouth_pattern")
            with urllib.request.urlopen(req, timeout=0.1) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get('mouth_pattern')
        except Exception as e:
            print(f"⚠️ 現在の口パターン取得エラー: {e}")
        return None  # 取得できない場合はNone

    def set_mouth_pattern(self, pattern):
        """シリウスの口パターンを設定（同期版）"""
        try:
            data = json.dumps({"mouth_pattern": pattern}).encode('utf-8')
            req = urllib.request.Request(
                f"{SIRIUS_API_URL}/mouth_pattern",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=0.5) as response:
                return response.getcode() == 200
        except Exception as e:
            print(f"❌ 口パターン設定エラー: {e}")
            return False

    def reset_to_neutral(self):
        """全設定をニュートラルにリセット（口パターンを元の表情の自然な口に戻す）"""
        try:
            # /resetエンドポイントが利用できない場合は口パターンをNoneに設定
            success = self.set_mouth_pattern(None)
            if success:
                print("🔄 口パターンを元の表情の自然な口にリセットしました")
            return success
        except Exception as e:
            print(f"🔄 リセットエラー: {e}")
            return False

    def set_mouth_pattern_async(self, pattern):
        """シリウスの口パターンを非同期設定"""
        def _set_pattern():
            try:
                data = json.dumps({"mouth_pattern": pattern}).encode('utf-8')
                req = urllib.request.Request(
                    f"{SIRIUS_API_URL}/mouth_pattern",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=0.05) as response:
                    return response.getcode() == 200
            except:
                return False
        
        # 非同期実行
        thread = threading.Thread(target=_set_pattern, daemon=True)
        thread.start()

    def speak_with_lipsync(self, text, style_id=None, speed_scale=None, restore_original_mouth=True):
        """音声合成 + リップシンク（超精密同期版）
        
        Args:
            text: 合成するテキスト
            style_id: VOICEVOXスタイルID（Noneの場合はデフォルトを使用）
            speed_scale: 速度スケール（Noneの場合はデフォルトを使用）
            restore_original_mouth: 発話後に口パターンをリセットして表情の自然な口パターンに戻すかどうか
        """
        # パラメータのデフォルト設定
        style_id = style_id or self.style_id
        speed_scale = speed_scale or self.speed_scale
        
        print(f"�🎤 合成: 「{text}」 (速度: {speed_scale}x, スタイル: {style_id})")
        print(f"📏 文字数: {len(text)}文字")
        
        # 1. 発話前の元の口パターンを保存
        original_mouth_pattern = None
        if restore_original_mouth:
            original_mouth_pattern = self.get_current_mouth_pattern()
            print(f"💾 元の口パターン保存: {original_mouth_pattern}")
        
        # AudioQuery音韻解析を使用
        try:
            mouth_sequence = self.analyzer.get_mouth_shape_sequence(text, style_id, speed_scale)
            print("✅ AudioQuery音韻解析成功")
        except Exception as e:
            print(f"❌ AudioQuery音韻解析エラー: {e}")
            print("🔄 文字ベース解析にフォールバック")
            return self.speak_with_lipsync_fallback(text, style_id, speed_scale, restore_original_mouth)
        
        print("📝 口パターンシーケンス:")
        mouth_pattern_count = 0
        none_pattern_count = 0
        for i, (seq_time, mouth_shape, duration) in enumerate(mouth_sequence[:10]):  # 最初の10個まで表示
            print(f"  {seq_time:.2f}s: {mouth_shape} ({duration:.2f}s)")
            if mouth_shape is not None:
                mouth_pattern_count += 1
            else:
                none_pattern_count += 1
        
        if len(mouth_sequence) > 10:
            remaining = len(mouth_sequence) - 10
            for seq_time, mouth_shape, duration in mouth_sequence[10:]:
                if mouth_shape is not None:
                    mouth_pattern_count += 1
                else:
                    none_pattern_count += 1
            print(f"  ... 他{remaining}個")
        
        print(f"📊 統計: 総パターン数{len(mouth_sequence)}, 口パターン{mouth_pattern_count}個, None{none_pattern_count}個")
        
        # 音声合成
        try:
            audio_query = self.synthesizer.create_audio_query(text, style_id)
            
            # 速度スケールをAudioQueryに設定
            if hasattr(audio_query, 'speed_scale'):
                audio_query.speed_scale = speed_scale
                print(f"✅ 速度スケールを設定: {speed_scale}x")
            else:
                print(f"⚠️  AudioQueryにspeed_scale属性が見つからないため、デフォルト速度を使用")
            
            wav_data = self.synthesizer.synthesis(audio_query, style_id)
            print("✅ 音声合成成功")
        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            return
        
        # 超精密同期モード
        print("🎯 超精密同期モード開始...")
        
        # 2. 同期イベントを作成
        audio_start_event = threading.Event()
        
        # 3. 音声再生スレッドを開始
        audio_thread = threading.Thread(target=self.play_audio_precise, args=(wav_data, audio_start_event))
        audio_thread.daemon = True
        audio_thread.start()
        
        # 4. 音声再生開始を待機
        audio_start_event.wait()
        actual_audio_start = time_module.time()
        
        print(f"🔊 音声再生開始検知: {actual_audio_start:.6f}")
        
        # 5. 音声再生開始直後におしゃべりモードを有効化（これで自然なタイミング）
        if self.talking_controller:
            self.talking_controller.set_talking_mode(True)
            print("🎭 おしゃべりモード有効化（音声再生開始直後）")
        
        # 6. リップシンク実行（音声開始と完全同期）
        timing_stats = {'perfect': 0, 'good': 0, 'poor': 0}
        
        for seq_time, mouth_shape, duration in mouth_sequence:
            # 目標時間 = 音声開始時間 + シーケンス時間
            target_time = actual_audio_start + seq_time
            
            # 高精度なタイミング制御
            while True:
                current_time = time_module.time()
                time_to_target = target_time - current_time
                
                if time_to_target <= 0.0001:  # 0.1ms以内の精度
                    break
                elif time_to_target > 0:
                    # 短い時間はスピンウェイト
                    if time_to_target < 0.001:
                        pass  # スピン
                    else:
                        time_module.sleep(max(0.0001, time_to_target - 0.0005))
                else:
                    # 遅延が発生
                    break
            
            # 口パターン設定（非同期）
            server_pattern = f"mouth_{mouth_shape}" if mouth_shape else None
            if server_pattern:
                self.set_mouth_pattern_async(server_pattern)
                
                # タイミング精度評価
                actual_time = time_module.time()
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
                
                print(f"{sync_indicator} {seq_time:.2f}s: {server_pattern} (誤差:{timing_error_ms:+.1f}ms)")
        
        # 7. 同期統計を表示
        total_patterns = timing_stats['perfect'] + timing_stats['good'] + timing_stats['poor']
        if total_patterns > 0:
            perfect_rate = timing_stats['perfect'] / total_patterns * 100
            print(f"📈 同期精度: ✓{timing_stats['perfect']} ~{timing_stats['good']} ⚠{timing_stats['poor']} "
                  f"({perfect_rate:.1f}% が5ms以内の精度)")
        
        # 8. 終了時に口パターンをリセット（表情の自然な口パターンに戻す）
        time_module.sleep(0.2)
        
        # おしゃべりモードを無効化（audioqueryの実装と同じ）
        if self.talking_controller:
            self.talking_controller.set_talking_mode(False)
            print("🎭 おしゃべりモード無効化")
            # 明示的に口パターンをクリア
            time_module.sleep(0.1)  # おしゃべりモード無効化の処理を待つ
            success = self.talking_controller.set_mouth_pattern_fast(None)
            if success:
                print("✅ 口パターンをクリアしました（元の表情の自然な口パターンに戻る）")
            else:
                # フォールバック: 直接設定
                self.set_mouth_pattern(None)
                print("✅ 口パターンをクリアしました（フォールバック処理で戻る）")
        else:
            # talking_controllerがない場合は従来の方法
            if restore_original_mouth:
                print("🔄 口パターンをリセット（元の表情の自然な口パターンに戻す）")
                self.set_mouth_pattern_async(None)
            else:
                print("🔄 口パターンリセット")
                self.set_mouth_pattern_async(None)
        
        print("✅ 発話完了\n")

    def speak_with_word_based_lipsync(self, text, style_id=0):
        """単語ベースリップシンク（各単語の発音時間を測定して調整）"""
        print(f"🎤 単語ベース合成: 「{text}」")
        print(f"📏 文字数: {len(text)}文字")
        
        # 1. 音声合成（AudioQueryから総時間を取得）
        try:
            audio_query = self.synthesizer.create_audio_query(text, style_id)
            wav_data = self.synthesizer.synthesis(audio_query, style_id)
            print("✅ 音声合成成功")
        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            return
        
        # 2. 総発音時間を取得
        total_duration = self._get_audio_duration_from_wav(wav_data)
        print(f"📊 総発音時間: {total_duration:.2f}秒")
        
        # 3. テキストを単語に分割
        words = self._split_text_into_words(text)
        print(f"📝 単語分割: {words}")
        
        # 4. 各単語の発音時間を推定
        word_durations = self._estimate_word_durations(words, total_duration)
        print("📝 単語タイミング:")
        current_time = 0.0
        for word, duration in zip(words, word_durations):
            print(f"  {current_time:.2f}s - {current_time + duration:.2f}s: '{word}' ({duration:.2f}s)")
            current_time += duration
        
        # 5. 各単語の口パターンを決定
        word_mouth_patterns = []
        for word in words:
            pattern = self._get_word_mouth_pattern(word)
            word_mouth_patterns.append(pattern)
            print(f"🔤 単語 '{word}' → 口パターン: {pattern}")
        
        # 6. 音声再生開始
        audio_thread = threading.Thread(target=self.play_audio, args=(wav_data,))
        audio_thread.daemon = True
        audio_thread.start()
        
        # 7. 単語ベースリップシンク実行
        start_time = time_module.time()
        current_time = 0.0
        
        for word, duration, mouth_pattern in zip(words, word_durations, word_mouth_patterns):
            # 単語の発音期間中に口パターンを設定
            pattern_start_time = current_time
            
            # 口パターンがNoneの場合はスキップ（句読点など）
            if mouth_pattern is None:
                print(f"⏭️  {pattern_start_time:.2f}s: スキップ (単語: '{word}', 期間: {duration:.2f}s)")
            else:
                server_pattern = f"mouth_{mouth_pattern}"
                success = self.set_mouth_pattern(server_pattern)
                print(f"👄 {pattern_start_time:.2f}s: {server_pattern} (単語: '{word}', 期間: {duration:.2f}s)")
            
            # 次の単語まで待機
            current_time += duration
            elapsed = time_module.time() - start_time
            wait_time = current_time - elapsed
            if wait_time > 0:
                time_module.sleep(wait_time)
        
        # 8. 終了時に口をリセット
        time_module.sleep(0.5)
        self.set_mouth_pattern(None)
        print("✅ 発話完了\n")

    def speak_with_lipsync_fallback(self, text, style_id=0, speed_scale=1.0, restore_original_mouth=True):
        """フォールバック: 文字ベースリップシンク
        
        Args:
            text: 合成するテキスト
            style_id: VOICEVOXスタイルID
            speed_scale: 速度スケール
            restore_original_mouth: 発話後に口パターンをリセットして表情の自然な口パターンに戻すかどうか
        """
        print(f"🔄 フォールバック処理: 「{text}」 (速度: {speed_scale}x)")
        
        # 1. 発話前の元の口パターンを保存
        original_mouth_pattern = None
        if restore_original_mouth:
            original_mouth_pattern = self.get_current_mouth_pattern()
            print(f"💾 元の口パターン保存: {original_mouth_pattern}")
        
        # 音声合成（シンプル版）
        try:
            wav_data = self.synthesizer.tts(text, style_id)
        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            return
        
        # シンプルな音韻解析（文字ベース）
        mouth_sequence = self.text_to_mouth_sequence(text)
        
        print("📝 口パターンシーケンス (フォールバック):")
        mouth_pattern_count = 0
        none_pattern_count = 0
        for i, (time, shape, duration) in enumerate(mouth_sequence[:15]):
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
        
        # 音声再生開始
        audio_thread = threading.Thread(target=self.play_audio, args=(wav_data,))
        audio_thread.daemon = True
        audio_thread.start()
        
        # リップシンク実行
        start_time = time_module.time()
        
        for seq_time, mouth_shape, duration in mouth_sequence:
            # タイミング待機
            elapsed = time_module.time() - start_time
            wait_time = seq_time - elapsed
            if wait_time > 0:
                time_module.sleep(wait_time)
            
            # 口パターン設定（正しい形式に変換）
            server_pattern = mouth_shape  # 既にmouth_形式になっているはず
            
            self.set_mouth_pattern(server_pattern)
            
            # デバッグ出力
            if server_pattern:
                print(f"👄 {seq_time:.2f}s: {server_pattern}")
        
        # 終了時に口パターンをリセット（元の表情の自然な口パターンに戻す）
        time_module.sleep(0.2)
        
        # 明示的に口パターンをNoneに設定
        success = self.set_mouth_pattern(None)
        if success:
            if restore_original_mouth:
                print("🔄 口パターンをリセット（元の表情の自然な口パターンに戻す）")
            else:
                print("🔄 口パターンリセット")
            print("✅ 発話完了（口を元の表情の自然な口パターンに戻しました）")
        else:
            print("⚠️ 口パターンのリセットに失敗しました")
        print()

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
        """文字から口の形を推定（pykakasi漢字読み対応版）"""
        # 漢字の場合はpykakasiで読みに変換
        if self.kakasi_converter and self._is_kanji(char):
            try:
                # 新しいpykakasiのAPI使用
                converted = self.kakasi_converter.convert(char)
                if converted:
                    # 変換結果から'hira'（ひらがな）を取得
                    hiragana_reading = ''.join([item['hira'] for item in converted])
                    if hiragana_reading and hiragana_reading != char:
                        # 読みの最初の文字で口の形を判定
                        first_char = hiragana_reading[0]
                        print(f"🔤 漢字変換: '{char}' → '{hiragana_reading}' → 判定文字:'{first_char}'")
                        return self._hiragana_to_mouth_shape(first_char)
            except Exception as e:
                print(f"⚠️  漢字読み変換エラー: {char} - {e}")
        
        # ひらがな・カタカナ・その他の処理
        return self._hiragana_to_mouth_shape(char)
    
    def _is_kanji(self, char):
        """文字が漢字かどうか判定"""
        return '\u4e00' <= char <= '\u9faf'
    
    def _hiragana_to_mouth_shape(self, char):
        """ひらがな・カタカナから口の形を判定"""
        # ひらがな・カタカナの母音判定
        a_sounds = 'あかがさざただなはばぱまやらわアカガサザタダナハバパマヤラワ'
        i_sounds = 'いきぎしじちぢにひびぴみりイキギシジチヂニヒビピミリ'
        o_sounds = 'うえおこごそぞとどのほぼぽもよろをンウエオコゴソゾトドノホボポモヨロヲン'
        
        if char in a_sounds:
            return 'mouth_a'
        elif char in i_sounds:
            return 'mouth_i' 
        elif char in o_sounds:
            return 'mouth_o'
        else:
            return None

    def _get_audio_duration_from_wav(self, wav_data):
        """WAVデータから音声の長さを取得"""
        try:
            import io
            import wave
            
            # WAVデータをメモリから読み込み
            wav_buffer = io.BytesIO(wav_data)
            with wave.open(wav_buffer, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"⚠️  WAV時間取得エラー: {e}")
            # フォールバック: 文字数ベースの推定
            return len(wav_data) / 44100.0  # 仮定のサンプリングレート

    def _split_text_into_words(self, text):
        """テキストを単語に分割（日本語対応）"""
        import re
        
        # 句読点で分割し、空の要素を除去
        words = re.split(r'([、。,.！？!?\s]+)', text)
        words = [word.strip() for word in words if word.strip()]
        
        # さらに細かく分割（漢字や長い単語を分割）
        refined_words = []
        for word in words:
            if len(word) > 3:  # 3文字以上の単語はさらに分割
                # 漢字とひらがなの境界で分割
                parts = re.split(r'([^\u3040-\u309f\u30a0-\u30ff]+)', word)
                parts = [part for part in parts if part]
                refined_words.extend(parts)
            else:
                refined_words.append(word)
        
        return refined_words

    def _estimate_word_durations(self, words, total_duration):
        """各単語の発音時間を推定"""
        if not words:
            return []
        
        # 各単語の文字数に基づいて時間を分配
        total_chars = 0
        punctuation_words = []
        
        for word in words:
            if word in [',', '、', '。', '.', '!', '！', '?', '？']:
                punctuation_words.append(word)
            else:
                total_chars += len(word)
        
        # 句読点の時間（固定で短く）
        punctuation_duration = 0.15  # 句読点1つあたり0.15秒
        punctuation_total_time = len(punctuation_words) * punctuation_duration
        
        # 実単語の総時間
        word_total_time = total_duration - punctuation_total_time
        if word_total_time < 0:
            word_total_time = total_duration * 0.8  # 最低でも80%は単語に
            punctuation_total_time = total_duration * 0.2
        
        # 文字数に比例して時間を分配（ベース時間 + 文字数ボーナス）
        base_duration_per_char = 0.08  # 1文字あたりの基本時間
        min_duration_per_word = 0.2    # 単語の最小時間
        
        word_durations = []
        for word in words:
            if word in [',', '、', '。', '.', '!', '！', '?', '？']:
                # 句読点は固定時間
                word_durations.append(punctuation_duration)
            else:
                char_based_duration = len(word) * base_duration_per_char
                duration = max(char_based_duration, min_duration_per_word)
                word_durations.append(duration)
        
        # 実単語の時間を調整してフィットさせる
        current_word_total = sum(d for w, d in zip(words, word_durations) 
                               if w not in [',', '、', '。', '.', '!', '！', '?', '？'])
        
        if current_word_total > 0 and word_total_time > 0:
            scale_factor = word_total_time / current_word_total
            word_durations = [d * scale_factor if w not in [',', '、', '。', '.', '!', '！', '?', '？'] else d 
                            for w, d in zip(words, word_durations)]
        
        return word_durations

    def _get_word_mouth_pattern(self, word):
        """単語から口パターンを決定"""
        if not word or word in [',', '、', '。', '.', '!', '！', '?', '？']:
            return None  # 句読点や空文字の場合は口パターンを設定しない
        
        # 単語の最初の文字でパターンを決定
        first_char = word[0]
        
        # 漢字の場合は読みに変換
        if self.kakasi_converter and self._is_kanji(first_char):
            try:
                converted = self.kakasi_converter.convert(first_char)
                if converted:
                    hiragana_reading = ''.join([item['hira'] for item in converted])
                    if hiragana_reading:
                        first_char = hiragana_reading[0]
            except Exception as e:
                pass  # 変換失敗時は元の文字を使用
        
        # 母音でパターンを決定
        if first_char in 'あかがさざただなはばぱまやらわアカガサザタダナハバパマヤラワ':
            return 'a'
        elif first_char in 'いきぎしじちぢにひびぴみりイキギシジチヂニヒビピミリ':
            return 'i'
        elif first_char in 'うえおこごそぞとどのほぼぽもよろをンウエオコゴソゾトドノホボポモヨロヲン':
            return 'o'
        else:
            # その他の文字の場合は'a'をデフォルトに
            return 'a'

def main():
    import sys
    
    print("🎭 精密リップシンクテスト（ハードコード設定・元の口パターン復元機能付き）")
    print("=" * 60)
    
    # リップシンクコントローラー初期化（JSONファイル不要）
    controller = LipSyncController()
    
    print("🎤 テスト開始（シリウスの表情サーバーが起動している必要があります）")
    print(f"📡 API URL: {SIRIUS_API_URL}")
    
    try:
        # テスト前に特定の口パターンを設定（元の表情をシミュレート）
        print("\n--- テスト準備 ---")
        print("🎭 発話前の口パターンを 'mouth_a' に設定（元の表情をシミュレート）")
        controller.set_mouth_pattern("mouth_a")
        time_module.sleep(0.5)  # 設定が反映されるまで待機
        
        print("\n--- 精密同期テスト（ハードコード設定・元の口パターン復元機能） ---")
        
        # テストフレーズを使用
        test_text = "こんにちは、シリウスです。リップシンクのテストを行っています。"
        
        print(f"🎭 テスト発話: '{test_text}'")
        controller.speak_with_lipsync(test_text, restore_original_mouth=True)
        
        print("\n--- 復元機能テスト完了 ---")
        print("💡 発話後に口パターンがリセットされ、元の表情の自然な口パターンに戻っているはずです")
        
        # 最終確認: 口パターンが確実にクリアされているかチェック
        print("\n--- 最終確認 ---")
        current_mouth = controller.get_current_mouth_pattern()
        print(f"🔍 現在の口パターン: {current_mouth}")
        
        if current_mouth is not None and current_mouth != "":
            print("⚠️ 口パターンがまだ設定されています。再度クリアを実行...")
            success = controller.set_mouth_pattern(None)
            if success:
                time_module.sleep(0.3)
                final_mouth = controller.get_current_mouth_pattern()
                print(f"🔍 クリア後の口パターン: {final_mouth}")
                if final_mouth is None or final_mouth == "":
                    print("✅ 口パターンが正常にクリアされました")
                else:
                    print("⚠️ 口パターンのクリアが完全ではありません")
            else:
                print("❌ 口パターンのクリアに失敗しました")
        else:
            print("✅ 口パターンは正常にクリアされています")
        
        print("\n🎉 全テスト完了！")
        
    except KeyboardInterrupt:
        print("\n🛑 テスト中止")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
