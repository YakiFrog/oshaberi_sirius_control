#!/usr/bin/env python3
"""
音声合成プログラム
VOICEVOX Core を使用した音声合成機能
"""

import os
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# VOICEVOX Core設定
VOICEVOX_ONNXRUNTIME_PATH = "voicevox_core/onnxruntime/lib/" + Onnxruntime.LIB_VERSIONED_FILENAME
OPEN_JTALK_DICT_DIR = "voicevox_core/dict/open_jtalk_dic_utf_8-1.11"
MODEL_PATH = "voicevox_core/models/vvms/13.vvm"  # 13.vvmを使用

class SpeechSynthesizer:
    """VOICEVOX を使用した音声合成クラス"""

    def __init__(self):
        print("🚀 VOICEVOX初期化中...")
        self.synthesizer = Synthesizer(
            Onnxruntime.load_once(filename=VOICEVOX_ONNXRUNTIME_PATH),
            OpenJtalk(OPEN_JTALK_DICT_DIR)
        )

        # 音声モデル読み込み（13.vvmを使用）
        with VoiceModelFile.open(MODEL_PATH) as model:
            self.synthesizer.load_voice_model(model)
        print("✅ VOICEVOX準備完了")

        # デフォルト音声パラメータ
        self.default_style_id = 54
        self.default_speed_scale = 1.0
        self.default_pitch_scale = 0.0
        self.default_intonation_scale = 0.9

    def create_audio_query(self, text: str, style_id: int = None):
        """AudioQuery を作成"""
        style_id = style_id or self.default_style_id
        return self.synthesizer.create_audio_query(text, style_id)

    def synthesize_speech(self, audio_query, style_id: int = None):
        """音声合成を実行"""
        style_id = style_id or self.default_style_id
        return self.synthesizer.synthesis(audio_query, style_id)

    def synthesize_simple(self, text: str, style_id: int = None):
        """シンプル音声合成（TTS）"""
        style_id = style_id or self.default_style_id
        return self.synthesizer.tts(text, style_id)

    def set_speed_scale(self, audio_query, speed_scale: float):
        """AudioQuery に速度スケールを設定"""
        if hasattr(audio_query, 'speed_scale'):
            audio_query.speed_scale = speed_scale
            return True
        return False