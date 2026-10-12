import asyncio
import os
import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from tools import html2text_tool, fetch_usd_rub_rate, save_rate_to_db, forecast_usd_rub_next_week


load_dotenv()

today = datetime.date.today().strftime("%Y-%m-%d")

api_key = os.getenv("GROQ_API_KEY")

system_prompt = (
    f"Ты — финансовый ассистент. Сегодняшняя дата: {today}. "
    "Твоя задача — узнать актуальный курс доллара к рублю,"
    "а затем сохранить этот курс в базу данных на сегодняшнюю дату."
)
#user_query = "Найди текущую цену доллара к рублю и засунь результат в нашу базу данных Postgres," \
#"затем попробуй предсказать цену доллара на неделю вперед."

async def main():
    model = ChatGroq(
        api_key=api_key,
        model="qwen/qwen3.8-27b",
        temperature=0.6,
        max_tokens=1000, 
        streaming=True, 
        verbose=True
    )
    api_wrapper = DuckDuckGoSearchAPIWrapper(region="rus", max_results=5)
    duckduckgo_tool = DuckDuckGoSearchResults(api_wrapper=api_wrapper)
    
    agent = create_agent(model, tools=[duckduckgo_tool, html2text_tool, 
                        fetch_usd_rub_rate, save_rate_to_db, forecast_usd_rub_next_week], 
                        system_prompt=system_prompt)

    chat_history = []

    print("🤖 Ассистент готов к работе! Для выхода введите 'exit' или 'quit'.\n")

    while True:
        try:
            user_query = input("Вы: ")
            
            if user_query.strip().lower() in ["exit", "quit", "выход", "стоп"]:
                print("🤖 До свидания!")
                break
            if not user_query.strip():
                continue

            chat_history.append(("user", user_query))
            inputs = {"messages": chat_history}
            print("🤖 Ответ ассистента: ")
            
            async for chunk in agent.astream(inputs, stream_mode="values"):
                final_message = chunk["messages"][-1]
                final_message.pretty_print()
            
            chat_history.append(("assistant", final_message.content))
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n🤖 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
