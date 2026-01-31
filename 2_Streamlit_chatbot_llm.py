"""
Chatbot LLM 모듈
================
OpenAI API를 활용한 대화형 챗봇의 핵심 로직
"""

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# === 프롬프트 설정 ===
BASIC_PROMPT = "You are a kind assistant."
REVISED_PROMPT = "You are a grumpy assistant."

# === LLM 초기화 ===
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# 대화 기록 저장 (최근 3개 대화만 유지)
memory = ConversationBufferWindowMemory(k=3, return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)


def get_basic_response(user_input: str) -> str:
    """기본 페르소나로 응답 생성"""
    full_prompt = f"{BASIC_PROMPT}\n\n{user_input}"
    return conversation.predict(input=full_prompt)


def get_revised_response(user_input: str) -> str:
    """수정된 페르소나로 응답 생성"""
    full_prompt = f"{REVISED_PROMPT}\n\n{user_input}"
    return conversation.predict(input=full_prompt)
