#!/usr/bin/env python3
"""
口の形制御プログラム
シリウスの口パターンを制御する機能
"""

import os
import urllib.request
import json
import threading
import time as time_module

try:
    import requests
    from typing import Optional
except ImportError:
    requests = None
    Optional = object  # ダミーオブジェクト
    print("⚠️  requestsがインストールされていません。おしゃべりモード制御は利用できません。")

# シリウス表情制御API
SIRIUS_API_URL = "http://localhost:8080"

class MouthController:
    """シリウスの口パターンを制御するクラス"""

    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.current_mouth_pattern = None

    def get_current_mouth_pattern(self):
        """現在のシリウス口パターンを取得"""
        try:
            req = urllib.request.Request(f"{self.server_url}/mouth_pattern")
            with urllib.request.urlopen(req, timeout=0.1) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    return data.get('mouth_pattern')
        except Exception as e:
            print(f"⚠️ 現在の口パターン取得エラー: {e}")
        return None

    def set_mouth_pattern(self, pattern):
        """シリウスの口パターンを設定（同期版）"""
        try:
            data = json.dumps({"mouth_pattern": pattern}).encode('utf-8')
            req = urllib.request.Request(
                f"{self.server_url}/mouth_pattern",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=0.5) as response:
                success = response.getcode() == 200
                if success:
                    self.current_mouth_pattern = pattern
                return success
        except Exception as e:
            print(f"❌ 口パターン設定エラー: {e}")
            return False

    def set_mouth_pattern_async(self, pattern):
        """シリウスの口パターンを非同期設定"""
        def _set_pattern():
            try:
                data = json.dumps({"mouth_pattern": pattern}).encode('utf-8')
                req = urllib.request.Request(
                    f"{self.server_url}/mouth_pattern",
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=0.05) as response:
                    if response.getcode() == 200:
                        self.current_mouth_pattern = pattern
            except:
                pass

        # 非同期実行
        thread = threading.Thread(target=_set_pattern, daemon=True)
        thread.start()

    def reset_to_neutral(self):
        """全設定をニュートラルにリセット（口パターンを元の表情の自然な口に戻す）"""
        try:
            # 口パターンをNoneに設定
            success = self.set_mouth_pattern(None)
            if success:
                print("🔄 口パターンを元の表情の自然な口にリセットしました")
            return success
        except Exception as e:
            print(f"🔄 リセットエラー: {e}")
            return False

class TalkingModeController:
    """おしゃべりモード制御クラス"""

    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.is_talking_mode_active = False
        self.last_mouth_pattern = None  # 冗長リクエストを防ぐ

        # 高速化のためのHTTPセッション設定
        if requests:
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
        else:
            self.session = None

    def set_talking_mode(self, enabled: bool) -> bool:
        """おしゃべりモードを設定"""
        if self.is_talking_mode_active == enabled:
            return True  # ログ出力も省略して高速化

        if not self.session:
            return False

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

        if not self.session:
            return False

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
        if not self.session:
            return False

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
            if self.session:
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

class AudioPlayer:
    """音声再生機能"""

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