"""
Chatbot LLM 모듈
================
GPT-5 또는 Gemini API를 활용한 대화형 챗봇
API 키에 따라 자동 선택 (GOOGLE_API_KEY > OPENAI_API_KEY)
"""

import os
from typing import Optional
from abc import ABC, abstractmethod


# === API 키 관리 ===
def get_secret(key: str) -> Optional[str]:
    """API 키를 가져옵니다. (Streamlit secrets > 환경변수)"""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, Exception):
        pass
    return os.getenv(key)


# === 프롬프트 설정 ===
BASIC_PROMPT = "You are a kind assistant. Please respond in Korean."
REVISED_PROMPT = "You are a grumpy assistant. Please respond in Korean."


# === LLM 추상 클래스 ===
class BaseChatLLM(ABC):
    """챗봇용 LLM 추상 클래스"""

    def __init__(self):
        self.history: list[dict] = []

    @abstractmethod
    def generate(self, system_prompt: str, user_input: str) -> str:
        """응답 생성"""
        pass

    def clear_history(self):
        """대화 기록 초기화"""
        self.history = []


class GeminiChatLLM(BaseChatLLM):
    """Google Gemini API"""

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__()
        from google import genai

        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY를 설정하세요")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_input: str) -> str:
        # 대화 기록 포함한 프롬프트 구성
        history_text = ""
        for msg in self.history[-6:]:  # 최근 3턴 (6개 메시지)
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            history_text += f"{role}: {msg['content']}\n"

        full_prompt = f"""{system_prompt}

{history_text}사용자: {user_input}

어시스턴트:"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt
        )

        # 대화 기록 저장
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response.text})

        return response.text


class OpenAIChatLLM(BaseChatLLM):
    """OpenAI GPT API"""

    def __init__(self, model: str = "gpt-5"):
        super().__init__()
        from openai import OpenAI

        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY를 설정하세요")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_input: str) -> str:
        # 메시지 구성
        messages = [{"role": "system", "content": system_prompt}]

        # 대화 기록 추가 (최근 6개)
        for msg in self.history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        assistant_message = response.choices[0].message.content

        # 대화 기록 저장
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message


# === LLM 인스턴스 생성 ===
def create_chat_llm() -> BaseChatLLM:
    """
    챗봇용 LLM 생성 (자동 선택)
    우선순위: GOOGLE_API_KEY > OPENAI_API_KEY
    """
    google_key = get_secret("GOOGLE_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    if google_key:
        print("   Gemini API 사용 (gemini-2.5-flash)")
        return GeminiChatLLM(model="gemini-2.5-flash")

    if openai_key:
        print("   OpenAI API 사용 (gpt-5)")
        return OpenAIChatLLM(model="gpt-5")

    raise ValueError(
        "API 키를 설정하세요.\n"
        "- GOOGLE_API_KEY (Gemini)\n"
        "- OPENAI_API_KEY (GPT-5)"
    )


# === 전역 LLM 인스턴스 (lazy loading) ===
_llm_instance: Optional[BaseChatLLM] = None


def get_llm() -> BaseChatLLM:
    """LLM 인스턴스 반환 (싱글톤)"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_chat_llm()
    return _llm_instance


def reset_llm():
    """LLM 인스턴스 초기화"""
    global _llm_instance
    if _llm_instance:
        _llm_instance.clear_history()


# === 응답 생성 함수 ===
def get_basic_response(user_input: str) -> str:
    """기본 페르소나로 응답 생성"""
    llm = get_llm()
    return llm.generate(BASIC_PROMPT, user_input)


def get_revised_response(user_input: str) -> str:
    """수정된 페르소나로 응답 생성"""
    llm = get_llm()
    return llm.generate(REVISED_PROMPT, user_input)
