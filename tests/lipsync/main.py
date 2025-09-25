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

try:
    import requests
    from typing import Optional
except ImportError:
    requests = None
    Optional = object  # ダミーオブジェクト
    print("⚠️  requestsがインストールされていません。おしゃべりモード制御は利用できません。")

# 分割されたモジュールをインポート
from speech_synthesis import SpeechSynthesizer
from phoneme_analysis import PhonemeAnalyzer
from mouth_control import MouthController, TalkingModeController, AudioPlayer

# シリウス表情制御API
SIRIUS_API_URL = "http://localhost:8080"

class LipSyncController:
    def __init__(self):
        # 分割されたモジュールの初期化
        self.speech_synthesizer = SpeechSynthesizer()
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.mouth_controller = MouthController()
        self.audio_player = AudioPlayer()
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
            original_mouth_pattern = self.mouth_controller.get_current_mouth_pattern()
            print(f"💾 元の口パターン保存: {original_mouth_pattern}")
        
        # AudioQuery音韻解析を使用
        try:
            audio_query = self.speech_synthesizer.create_audio_query(text, style_id)
            self.speech_synthesizer.set_speed_scale(audio_query, speed_scale)
            mouth_sequence = self.phoneme_analyzer.get_mouth_shape_sequence(audio_query, speed_scale)
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
            wav_data = self.speech_synthesizer.synthesize_speech(audio_query, style_id)
            print("✅ 音声合成成功")
        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            return
        
        # 超精密同期モード
        print("🎯 超精密同期モード開始...")
        
        # 2. 同期イベントを作成
        audio_start_event = threading.Event()
        
        # 3. 音声再生スレッドを開始
        audio_thread = threading.Thread(target=self.audio_player.play_audio_precise, args=(wav_data, audio_start_event))
        audio_thread.daemon = True
        audio_thread.start()
        
        # 4. 音声再生開始を待機
        audio_start_event.wait()
        actual_audio_start = time_module.time()
        
        print(f"🔊 音声再生開始検知: {actual_audio_start:.6f}")
        
        # 6. リップシンク実行（音声開始と完全同期）
        timing_stats = {'perfect': 0, 'good': 0, 'poor': 0}
        first_mouth_pattern = True  # 最初の口パターン設定フラグ
        
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
            
            # 最初の口パターン設定前におしゃべりモードを有効化
            if first_mouth_pattern and self.talking_controller:
                self.talking_controller.set_talking_mode(True)
                print("🎭 おしゃべりモード有効化（最初の口パターン設定前）")
                # おしゃべりモード有効化の処理を待つ
                time_module.sleep(0.02)  # 20msの短い待機
            
            # 口パターン設定（同期版で確実に設定）
            server_pattern = f"mouth_{mouth_shape}" if mouth_shape else None
            if server_pattern:
                # 最初のパターン設定時は同期版を使用してタイミングを確実にする
                if first_mouth_pattern or seq_time < 0.5:  # 最初の0.5秒間は同期版
                    success = self.mouth_controller.set_mouth_pattern(server_pattern)
                    if not success:
                        print(f"⚠️ 口パターン設定失敗: {server_pattern}")
                    if first_mouth_pattern:
                        first_mouth_pattern = False
                else:
                    # その後は非同期版で高速化
                    self.mouth_controller.set_mouth_pattern_async(server_pattern)
                
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
        
        # 5. 同期統計を表示
        total_patterns = timing_stats['perfect'] + timing_stats['good'] + timing_stats['poor']
        if total_patterns > 0:
            perfect_rate = timing_stats['perfect'] / total_patterns * 100
            print(f"📈 同期精度: ✓{timing_stats['perfect']} ~{timing_stats['good']} ⚠{timing_stats['poor']} "
                  f"({perfect_rate:.1f}% が5ms以内の精度)")
        
        # 6. 終了時に口パターンをリセット（表情の自然な口パターンに戻す）
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
                self.mouth_controller.set_mouth_pattern(None)
                print("✅ 口パターンをクリアしました（フォールバック処理で戻る）")
        else:
            # talking_controllerがない場合は従来の方法
            if restore_original_mouth:
                print("🔄 口パターンをリセット（元の表情の自然な口パターンに戻す）")
                self.mouth_controller.set_mouth_pattern_async(None)
            else:
                print("🔄 口パターンリセット")
                self.mouth_controller.set_mouth_pattern_async(None)
        
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
            original_mouth_pattern = self.mouth_controller.get_current_mouth_pattern()
            print(f"💾 元の口パターン保存: {original_mouth_pattern}")
        
        # 音声合成（シンプル版）
        try:
            wav_data = self.speech_synthesizer.synthesize_simple(text, style_id)
        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            return
        
        # シンプルな音韻解析（文字ベース）
        mouth_sequence = self.phoneme_analyzer.text_to_mouth_sequence(text)
        
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
        audio_thread = threading.Thread(target=self.audio_player.play_audio, args=(wav_data,))
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
            
            self.mouth_controller.set_mouth_pattern(server_pattern)
            
            # デバッグ出力
            if server_pattern:
                print(f"👄 {seq_time:.2f}s: {server_pattern}")
        
        # 終了時に口パターンをリセット（元の表情の自然な口パターンに戻す）
        time_module.sleep(0.2)
        
        # 明示的に口パターンをNoneに設定
        success = self.mouth_controller.set_mouth_pattern(None)
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
        # テスト前に口パターンをクリアして自然な状態から始める
        print("\n--- テスト準備 ---")
        print("🎭 発話前の状態確認（自然な表情の口のまま開始）")
        current_mouth = controller.mouth_controller.get_current_mouth_pattern()
        print(f"🔍 現在の口パターン: {current_mouth}")
        
        # 口パターンがある場合はクリアして自然な状態にする
        if current_mouth is not None and current_mouth != "":
            print("🔄 口パターンをクリアして自然な状態にします")
            controller.mouth_controller.set_mouth_pattern(None)
            time_module.sleep(0.3)
            final_mouth = controller.mouth_controller.get_current_mouth_pattern()
            print(f"🔍 クリア後の口パターン: {final_mouth}")
        else:
            print("✅ 既に自然な状態です")
        
        time_module.sleep(0.2)  # 状態安定化のための短い待機
        
        print("\n--- 精密同期テスト（ハードコード設定・元の口パターン復元機能） ---")
        
        # テストフレーズを使用
        test_text = "こんにちは、シリウスです。リップシンクのテストを行っています。"
        
        print(f"🎭 テスト発話: '{test_text}'")
        controller.speak_with_lipsync(test_text, restore_original_mouth=True)
        
        print("\n--- 復元機能テスト完了 ---")
        print("💡 発話後に口パターンがリセットされ、元の表情の自然な口パターンに戻っているはずです")
        
        # 最終確認: 口パターンが確実にクリアされているかチェック
        print("\n--- 最終確認 ---")
        current_mouth = controller.mouth_controller.get_current_mouth_pattern()
        print(f"🔍 現在の口パターン: {current_mouth}")
        
        if current_mouth is not None and current_mouth != "":
            print("⚠️ 口パターンがまだ設定されています。再度クリアを実行...")
            success = controller.mouth_controller.set_mouth_pattern(None)
            if success:
                time_module.sleep(0.3)
                final_mouth = controller.mouth_controller.get_current_mouth_pattern()
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
