# 02_ollama_test.py

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class ChatLLM:
    def __init__(self):
        # Model
        self._model = ChatOllama(model="gemma2:2b", temperature=0.7)

        # Prompt
        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.
        Question: {question} """
        self._prompt = ChatPromptTemplate.from_template(self._template)

        # Chain 연결
        self._chain = (
            {'question': RunnablePassthrough()} | self._prompt
            | self._model
            | StrOutputParser()
        )

    def invoke(self, user_input):
        # LLM 체인을 실행하여 응답 생성
        response = self._chain.invoke({"question": user_input})
        return response

    def format_docs(self, docs):
        # 문서 내용을 문자열로 포맷
        return '\n\n'.join([d.page_content for d in docs])


class ChatWeb:
    def __init__(self, llm, page_title="Gazzi Chatbot", page_icon=":books:"):
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

    def run(self):
        # 웹 페이지 기본 환경 설정
        st.set_page_config(page_title=self._page_title, page_icon=self._page_icon)
        st.title(self._page_title)

        # 대화 기록 목록 초기화
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # 이전 대화 기록 출력
        if st.session_state["messages"]:
            for message in st.session_state["messages"]:
                st.chat_message(message["role"]).write(message["content"])

        # 사용자 입력 처리
        if user_input := st.chat_input("질문을 입력해 주세요."):
            # 사용자가 입력한 내용 출력 및 저장
            st.chat_message("user").write(user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # LLM 호출 및 응답 생성
            response = self._llm.invoke(user_input)

            # AI 메시지 출력 및 저장
            st.chat_message("assistant").write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})


if __name__ == '__main__':
    llm = ChatLLM()  # LLM 초기화
    web = ChatWeb(llm=llm)  # 웹 챗봇 실행
    web.run()
