#!/usr/bin/env python3
"""
VOICEVOX音声合成モジュール（リップシンク対応）
PySide6 UI統合用に最適化
"""

import os
import tempfile
import subprocess
import threading
import time
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# シリウス表情制御API
SIRIUS_API_URL = "http://localhost:8080"

class VoiceSynthesizer:
    def __init__(self):
        self.synthesizer = None
        self._init_synthesizer()

        # リップシンク関連の初期化
        self.style_id = 54
        self.speed_scale = 1.0
        self.pitch_scale = 0.0
        self.intonation_scale = 0.9

    def _init_synthesizer(self):
        """Synthesizerを初期化"""
        try:
            # VOICEVOX COREの初期化
            voicevox_onnxruntime_path = "voicevox_core/onnxruntime/lib/" + Onnxruntime.LIB_VERSIONED_FILENAME
            open_jtalk_dict_dir = "voicevox_core/dict/open_jtalk_dic_utf_8-1.11"

            self.synthesizer = Synthesizer(
                Onnxruntime.load_once(filename=voicevox_onnxruntime_path),
                OpenJtalk(open_jtalk_dict_dir)
            )

            # 音声モデルの読み込み（13.vvmを使用）
            with VoiceModelFile.open("voicevox_core/models/vvms/13.vvm") as model:
                self.synthesizer.load_voice_model(model)

            print("✅ VOICEVOX初期化完了")

        except Exception as e:
            print(f"❌ VOICEVOX初期化エラー: {e}")
            self.synthesizer = None

    def speak_with_lipsync(self, text: str):
        """音声合成 + リップシンク"""
        if not self.synthesizer:
            print("❌ Synthesizerが初期化されていません")
            return False

        try:
            print(f"🎤 合成: 「{text}」 (速度: {self.speed_scale}x, スタイル: {self.style_id})")

            # AudioQuery作成
            audio_query = self.synthesizer.create_audio_query(text, self.style_id)
            self._set_speed_scale(audio_query, self.speed_scale)

            # 音声合成
            wav_data = self.synthesizer.synthesis(audio_query, self.style_id)
            print("✅ 音声合成成功")

            # リップシンク実行
            self._perform_lipsync(audio_query, wav_data, self.speed_scale)

            return True

        except Exception as e:
            print(f"❌ 音声合成エラー: {e}")
            return False

    def _set_speed_scale(self, audio_query, speed_scale: float):
        """AudioQueryに速度スケールを設定"""
        if hasattr(audio_query, 'speed_scale'):
            audio_query.speed_scale = speed_scale

    def _perform_lipsync(self, audio_query, wav_data, speed_scale: float):
        """リップシンク実行"""
        # 口形状シーケンス生成
        mouth_sequence = self._get_mouth_shape_sequence(audio_query, speed_scale)

        print("📝 口パターンシーケンス:")
        for i, (seq_time, mouth_shape, duration) in enumerate(mouth_sequence[:10]):
            print(".2f")
        if len(mouth_sequence) > 10:
            print(f"  ... 他{len(mouth_sequence) - 10}個")

        # 同期イベント作成
        audio_start_event = threading.Event()

        # 音声再生スレッド開始
        audio_thread = threading.Thread(
            target=self._play_audio_precise,
            args=(wav_data, audio_start_event),
            daemon=True
        )
        audio_thread.start()

        # 音声再生開始を待機
        audio_start_event.wait()
        actual_audio_start = time.time()

        print(".6f")

        # リップシンク実行
        timing_stats = {'perfect': 0, 'good': 0, 'poor': 0}
        first_mouth_pattern = True

        for seq_time, mouth_shape, duration in mouth_sequence:
            # 目標時間 = 音声開始時間 + シーケンス時間
            target_time = actual_audio_start + seq_time

            # 高精度タイミング制御
            while True:
                current_time = time.time()
                time_to_target = target_time - current_time

                if time_to_target <= 0.0001:
                    break
                elif time_to_target > 0:
                    if time_to_target < 0.001:
                        pass
                    else:
                        time.sleep(max(0.0001, time_to_target - 0.0005))
                else:
                    break

            # 口パターン設定
            server_pattern = f"mouth_{mouth_shape}" if mouth_shape else None
            if server_pattern:
                if first_mouth_pattern:
                    success = self._set_mouth_pattern(server_pattern)
                    if not success:
                        print(f"⚠️ 口パターン設定失敗: {server_pattern}")
                    first_mouth_pattern = False
                else:
                    self._set_mouth_pattern_async(server_pattern)

                # タイミング精度評価
                actual_time = time.time()
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

                print(".1f")

        # 統計表示
        total_patterns = timing_stats['perfect'] + timing_stats['good'] + timing_stats['poor']
        if total_patterns > 0:
            perfect_rate = timing_stats['perfect'] / total_patterns * 100
            print(".1f")

        # 終了時に口パターンをリセット
        time.sleep(0.2)
        self._set_mouth_pattern_async(None)
        print("✅ 発話完了\n")

    def _get_mouth_shape_sequence(self, audio_query, speed_scale: float):
        """AudioQueryから口形状シーケンスを生成"""
        try:
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
                                consonant_duration /= speed_scale

                                mouth_shape = self._phoneme_to_mouth_shape(consonant_phoneme)
                                phoneme_timeline.append((current_time, mouth_shape, consonant_duration))
                                current_time += consonant_duration

                            # 母音処理
                            if hasattr(mora, 'vowel') and mora.vowel:
                                vowel_phoneme = mora.vowel
                                vowel_duration = getattr(mora, 'vowel_length', 0.1) or 0.1
                                vowel_duration /= speed_scale

                                mouth_shape = self._phoneme_to_mouth_shape(vowel_phoneme)
                                phoneme_timeline.append((current_time, mouth_shape, vowel_duration))
                                current_time += vowel_duration

                    # ポーズ処理
                    if hasattr(accent_phrase, 'pause_mora') and accent_phrase.pause_mora:
                        pause_duration = getattr(accent_phrase.pause_mora, 'vowel_length', 0.0) or 0.0
                        if pause_duration > 0:
                            pause_duration /= speed_scale
                            phoneme_timeline.append((current_time, None, pause_duration))
                            current_time += pause_duration

            return phoneme_timeline

        except Exception as e:
            print(f"❌ 音韻解析エラー: {e}")
            return []

    def _phoneme_to_mouth_shape(self, phoneme):
        """音韻から口形状にマッピング"""
        phoneme_to_mouth = {
            'a': 'a', 'i': 'i', 'u': 'o', 'e': 'a', 'o': 'o',
            'k': 'a', 'g': 'a', 's': 'i', 'z': 'i', 't': 'a', 'd': 'a',
            'n': 'o', 'h': 'o', 'b': 'o', 'p': 'o', 'm': 'o', 'y': 'a',
            'r': 'a', 'w': 'o', 'f': 'o', 'v': 'o', 'ch': 'i', 'sh': 'i',
            'j': 'i', 'ts': 'a', 'sil': None, 'pau': None, 'cl': None,
            'q': None, 'N': 'o'
        }
        return phoneme_to_mouth.get(phoneme, 'a')

    def _set_mouth_pattern(self, pattern):
        """シリウスの口パターンを設定（同期版）"""
        try:
            import urllib.request
            import json

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

    def _set_mouth_pattern_async(self, pattern):
        """シリウスの口パターンを非同期設定"""
        def _set_pattern():
            try:
                import urllib.request
                import json

                data = json.dumps({"mouth_pattern": pattern}).encode('utf-8')
                req = urllib.request.Request(
                    f"{SIRIUS_API_URL}/mouth_pattern",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=0.05) as response:
                    pass
            except:
                pass

        thread = threading.Thread(target=_set_pattern, daemon=True)
        thread.start()

    def _play_audio_precise(self, wav_data, start_event):
        """音声を再生（精密同期版）"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file_path = temp_file.name

            # 再生開始を通知
            start_event.set()

            # afplayで再生
            process = subprocess.Popen(
                ['afplay', temp_file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            process.wait()
            os.unlink(temp_file_path)
        except Exception as e:
            print(f"❌ 音声再生エラー: {e}")

    def speak_response(self, text: str):
        """応答音声を再生（リップシンク付き）"""
        return self.speak_with_lipsync(text)

    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            if self.synthesizer:
                # VOICEVOX Coreのクリーンアップ
                # Synthesizerオブジェクトの明示的なクリーンアップ
                self.synthesizer = None
                print("✅ VOICEVOXリソースクリーンアップ完了")
        except Exception as e:
            print(f"⚠️ VOICEVOXクリーンアップ警告: {e}")