#!/usr/bin/env python3
"""
LM Studio チャットテストプログラム
コンソールから入力してAI応答を表示
"""

import requests
import json
import time

# 利用可能なモデルリスト
MODELS = {
    "1": {"name": "mistralai/magistral-small-2509", "description": "Mistral Smallモデル"},
    "2": {"name": "openai/gpt-oss-20b", "description": "OpenAI GPT OSS 20Bモデル"}
}

def select_model():
    """起動時にモデルを選択"""
    print("利用可能なモデル:")
    for key, model in MODELS.items():
        print(f"{key}: {model['description']} ({model['name']})")
    
    while True:
        choice = input("モデルを選択してください (1 または 2): ").strip()
        if choice in MODELS:
            selected_model = MODELS[choice]["name"]
            print(f"選択されたモデル: {MODELS[choice]['description']}")
            return selected_model
        else:
            print("無効な選択です。1 または 2 を入力してください。")

def chat_with_ai(user_message, model):
    """LM Studio APIでAIとチャット"""
    url = "http://localhost:1234/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            # システムメッセージで日本語応答を指定
            {
                "role": "system",
                "content": "あなたは親切で役立つAIアシスタントです。日本語で答えてください。"
            },
            # ユーザーメッセージで入力を渡す
            {
                "role": "user", 
                "content": user_message
            } 
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }

    # APIリクエストを送信
    try:
        start_time = time.time()  # 開始時間を記録
        response = requests.post(url, json=payload) # POSTリクエストを送信
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる

        result = response.json() # レスポンスをJSONとして解析
        end_time = time.time()  # 終了時間を記録
        elapsed_time = end_time - start_time  # 経過時間を計算

        ai_message = result["choices"][0]["message"]["content"] # AIの応答を取得

        return ai_message, elapsed_time  # 応答と時間を返す

    except requests.exceptions.RequestException as e:
        return f"エラー: {e}", 0.0

def main():
    """メイン関数"""
    print("🤖 LM Studio チャットテスト")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("-" * 50)
    
    # モデルを選択
    selected_model = select_model()
    
    try:
        while True:
            user_input = input("あなた: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("チャットを終了します。")
                break

            if not user_input:
                continue

            print("AI: ", end="", flush=True)
            ai_response, elapsed_time = chat_with_ai(user_input, selected_model)
            print(ai_response)
            print(f"応答時間: {elapsed_time:.2f}秒")
            print("-" * 50)
    except KeyboardInterrupt:
        print("\nチャットを終了します。")

if __name__ == "__main__":
    main()