# シリウス音声対話システム

<!-- 実装すること -->
音声認識をして，認識したテキストをもとに，ローカルLLMで応答を生成し，
VOICEVOXで音声合成し，音声を再生しつつ，シリウスの口の動きを連動させるシステム

## 必要なソフトウェア
- 音声認識: faster-whisper
- ローカルLLM: LM Studio
- 音声合成: VOICEVOX
- 口パク: sirius_face_animのa,i,oの口パターン

## 環境構築
### 1. VOICEVOX COREのインストール
[参考ページ](https://github.com/VOICEVOX/voicevox_core/blob/main/docs/guide/user/usage.md)

#### 1．Downloaderをダウンロードし，downloadとリネーム
Mac, Linuxの場合は実行権限を付与する.
``` bash
chmod +x download
```

#### 2．依存ライブラリとモデルをダウンロード
``` bash
# CPU版を利用する場合
./download --exclude c-api # C APIを使う場合は`--exclude c-api`は無し

# DirectML版を利用する場合
./download --exclude c-api --devices directml

# CUDA版を利用する場合
./download --exclude c-api --devices cuda
```
利用規約を確認し，インストールする

#### 4. Pythonライブラリをインストール
Downloaderと同じページで，whlファイルをダウンロードし，pipでインストールする
``` bash
pip install voicevox_core-*.whl
```

例：voicevox_core-0.16.1-cp310-abi3-macosx_11_0_arm64.whl
注意：仮想環境を利用している場合は，その環境をアクティベートしてからインストールすること

インストール後はwhlファイルを削除して良い

#### 5. voicevox_test.pyで動作確認
``` bash
cd /path/to/oshaberi_sirius_control 
. bin/activate
python3 voicevox_test.py
```
音声がoutput/output.wavに保存される
### 2. LM Studioのインストール
なんやかんやでインストール
#### 1. LM Studioを起動し，ローカルLLMサーバーを起動
#### 2. 動作確認(mistralai/magistral-small-2509を利用)
これは，mistralai/magistral-small-2509を利用して
メッセージを投げて，応答が返ってくるか確認するもの．
curlコマンドとは，HTTPリクエストを送信するためのコマンドラインツールである．curl ~~のように，URLを指定してHTTPリクエストを送信し，サーバーからの応答を取得できる(GETリクエスト)
``` bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/magistral-small-2509",
    "messages": [
      { "role": "system", "content": "Always answer in rhymes. Today is Thursday" },
      { "role": "user", "content": "What day is it today?" }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
}'
```
これを実行すると，以下のような応答が返ってくる
``` json
{
  "id": "chatcmpl-mkyjjnwwwejc6n4hs4k39u",
  "object": "chat.completion",
  "created": 1758824902,
  "model": "mistralai/magistral-small-2509",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Today's the day, so clear and bright,\nIt's Thursday, the week's delight!",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 20,
    "total_tokens": 40
  },
  "stats": {},
  "system_fingerprint": "mistralai/magistral-small-2509"
}
```
#### 3. Pythonでの動作確認
``` bash
cd /path/to/oshaberi_sirius_control
. bin/activate
python3 tests/llm_chat_test.py
```
もしくは， 
``` bash
python3 tests/llm_chat_async_test.py
```

### 3. faster-whisperのインストール
``` bash 
pip install faster-whisper
```

### 4. Pythonファイルのコンパイル
``` bash
python3 -m py_compile tests/faster_whisper_test.py
```
これで多少は起動動作が速くなる（数％程度）