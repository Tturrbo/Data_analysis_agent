import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
system_prompt = (
    "Ищи информацию строго на том языке, на котором обратился пользователь. "
    "Делай только один поисковый запрос за раз."
)

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
    
    async def html2text_tool(url: str) -> str:
        """Fetches a URL asynchronously and converts its HTML content into clean markdown text."""
        loader = AsyncHtmlLoader(url)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed[0].page_content if docs_transformed else "No content found."

    agent = create_agent(model, tools=[duckduckgo_tool, html2text_tool], system_prompt=system_prompt)

    inputs = {"messages": [("user", "Найди официальный вебсайт langchain, прочитай текст внтури и сделай саммари.")]}
        
    async for chunk in agent.astream(inputs, stream_mode="values"):
        final_message = chunk["messages"][-1]
        final_message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
