#!/usr/bin/env python3
"""
音韻解析プログラム
AudioQuery から音韻情報を抽出し、口の形制御用に変換
"""

try:
    import pykakasi
except ImportError:
    pykakasi = None
    print("⚠️  pykakasiがインストールされていません。漢字の読み変換は利用できません。")

class PhonemeAnalyzer:
    """AudioQueryから音韻情報を抽出してリップシンク用に変換"""

    def __init__(self):
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

    def analyze_from_audio_query(self, audio_query, speed_scale: float = 1.0):
        """AudioQueryから音韻情報を抽出してリップシンク用に変換"""
        try:
            print(f"🔍 AudioQuery音韻解析開始 (速度: {speed_scale}x)")

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

                                mouth_shape = self.phoneme_to_mouth.get(consonant_phoneme, 'a')
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

    def get_mouth_shape_sequence(self, audio_query, speed_scale: float = 1.0):
        """AudioQueryから口形状シーケンスを生成"""
        phoneme_timeline = self.analyze_from_audio_query(audio_query, speed_scale)
        mouth_sequence = []

        for time_pos, mouth_shape, duration in phoneme_timeline:
            if mouth_shape:  # Noneでない場合のみ追加
                mouth_sequence.append((time_pos, mouth_shape, duration))
            else:
                print(f"⚠️  無音音韻をスキップ: {time_pos:.2f}s (duration: {duration:.2f}s)")

        return mouth_sequence

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