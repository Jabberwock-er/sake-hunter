import os
import requests
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# 環境変数
LINE_ACCESS_TOKEN = os.environ["LINE_ACCESS_TOKEN"]
LINE_USER_ID = os.environ["LINE_USER_ID"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Geminiの設定
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

def send_line_message(text):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    data = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": text}]
    }
    requests.post(url, headers=headers, json=data)

search_tool = SerperDevTool()

researcher = Agent(
    role='コンビニお酒リサーチャー',
    goal='主要コンビニの今週のお酒新商品を見つける',
    backstory='新商品情報に詳しいリサーチャー。',
    tools=[search_tool],
    llm=llm,
    verbose=True,
    memory=False
)

writer = Agent(
    role='晩酌まとめ担当',
    goal='新商品情報をLINEで見やすい短文にまとめる',
    backstory='情報を箇条書きで整理するのが得意。',
    llm=llm,
    verbose=True,
    memory=False
)

task1 = Task(
    description='2025年11月（現在）または直近に日本のコンビニで発売されるお酒の新商品を検索してください。',
    expected_output='新商品リスト',
    agent=researcher
)

task2 = Task(
    description='得られた情報から注目すべき3つを選び、LINE用に「・商品名 (コンビニ名): 特徴」の形式でまとめてください。冒頭に【今週の注目酒🍺】とつけてください。',
    expected_output='LINE送信用のテキスト',
    agent=writer,
    context=[task1]
)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])

if __name__ == "__main__":
    try:
        result = crew.kickoff()
        send_line_message(str(result))
    except Exception as e:
        print(f"Error: {e}")
