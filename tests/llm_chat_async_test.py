#!/usr/bin/env python3
"""
LM Studio チャットテストプログラム（非同期版）
コンソールから入力してAI応答を表示

非同期のメリット:
- レスポンス待ち中に他のタスクを実行可能（例: 並行リクエストやUI更新）
- 複数のリクエストを同時に処理できるため、高負荷時に効率的
- GUIアプリケーションでUIのフリーズを防げる
- ネットワーク遅延が大きい場合に全体のパフォーマンスが向上
"""

import aiohttp
import asyncio
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

async def chat_with_ai_async(user_message, model):
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "あなたは親切で役立つAIアシスタントです。日本語で答えてください。"},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            end_time = time.time()
            elapsed_time = end_time - start_time
            ai_message = result["choices"][0]["message"]["content"]
            return ai_message, elapsed_time

async def main():
    print("🤖 LM Studio チャットテスト（非同期版）")
    print("終了するには 'quit' または 'exit' と入力してください")
    print("-" * 50)
    
    # モデルを選択
    selected_model = select_model()
    
    try:
        while True:
            user_input = input("あなた: ").strip()  # 入力は同期のまま
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("チャットを終了します。")
                break
            if not user_input:
                continue
            
            print("AI: ", end="", flush=True)
            ai_response, elapsed_time = await chat_with_ai_async(user_input, selected_model)
            print(ai_response)
            print(f"応答時間: {elapsed_time:.2f}秒")
            print("-" * 50)
    except KeyboardInterrupt:
        print("\nチャットを終了します。")

if __name__ == "__main__":
    asyncio.run(main())