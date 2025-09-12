import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor


# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    return TavilySearchResults(k=6, name="web_search")


# --------------------------------------------------------------------
# 2. PDF Tool (고정 PDF 사용)
# --------------------------------------------------------------------
def load_fixed_pdf():
    pdf_path = ".data/SW 중심사회 5월호 전문.pdf"   
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="This tool gives you direct access to the reference PDF document."
    )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "당신은 KEPCO KDN 임직원을 위한 유용한 어시스턴트입니다."
        "항상 먼저 `pdf_search`를 사용해 답변하세요."
        "`pdf_search`에서 관련 결과가 없을 경우에만 `web_search`를 사용하세요."
        "두 도구를 절대 함께 사용하지 마세요."
        "항상 전문적이고 친근한 톤으로 한국어로 답변하며, 이모지도 포함하세요."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]

    # intermediate_steps에서 마지막만 가져오기
    if result.get("intermediate_steps"):
        last_action, _ = result["intermediate_steps"][-1]
        answer += f"\n\n출처:\n- Tool: {last_action.tool}, Query: {last_action.tool_input}"

    return f"답변:\n{answer}"
# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="한전KDN AI 비서", layout="wide", page_icon="🤖")
    st.image('data/kdn_image.jpg', width=800)
    st.markdown('---')
    st.title("안녕하세요! PDF + Web 기반 RAG 챗봇 '한전KDN AI 비서' 입니다")  

    with st.sidebar:
        openai_api = st.text_input("OPENAI API 키", type="password")
        tavily_api = st.text_input("TAVILY API 키", type="password")

    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        # 툴 구성 (고정 PDF + 웹검색)
        tools = [search_web(), load_fixed_pdf()]
        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("질문을 입력하세요")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키를 입력하세요.")


if __name__ == "__main__":
    main()
