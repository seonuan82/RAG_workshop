"""
RAG 챗봇 - 심리 뉴스 검색 (정답 버전)
======================================
실행: streamlit run rag_chatbot.py

심리 관련 뉴스 데이터를 기반으로 질문에 답변하는 RAG 챗봇
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import os
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# === 페이지 설정 ===
st.set_page_config(
    page_title="RAG 챗봇 - 심리 뉴스",
    page_icon="📰",
    layout="wide"
)

# === 설정 ===
AVATAR_USER = "👤"
AVATAR_BOT = "🤖"

# 파일 경로 설정 (로컬 및 Streamlit Cloud 모두 지원)
def get_data_path():
    """데이터 파일 경로를 반환합니다."""
    # 현재 파일 기준 경로
    current_dir = Path(__file__).parent if "__file__" in dir() else Path(".")
    local_path = current_dir / "Practice_data_NewsResult.CSV"

    if local_path.exists():
        return str(local_path)

    # Streamlit Cloud에서는 현재 작업 디렉토리 기준
    cloud_path = Path("Practice_data_NewsResult.CSV")
    if cloud_path.exists():
        return str(cloud_path)

    # 상대 경로 시도
    return "Practice_data_NewsResult.CSV"

DATA_PATH = get_data_path()


# === 데이터 클래스 ===
@dataclass
class NewsItem:
    """뉴스 데이터 클래스"""
    news_id: str
    date: str
    publisher: str
    title: str
    content: str
    url: str
    embedding: Optional[list] = None


# === 세션 상태 초기화 ===
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "news_data" not in st.session_state:
        st.session_state.news_data = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "embeddings_ready" not in st.session_state:
        st.session_state.embeddings_ready = False


init_session()


def reset_chat():
    """대화 초기화"""
    st.session_state.messages = []


def reset_all():
    """전체 초기화"""
    st.session_state.messages = []
    st.session_state.news_data = []
    st.session_state.llm = None
    st.session_state.embeddings_ready = False


# ═══════════════════════════════════════════════════════════════════════════
# 뉴스 데이터 로드 함수
# ═══════════════════════════════════════════════════════════════════════════

def load_news_data(filepath: str, max_items: int = 100) -> list:
    """CSV 파일에서 뉴스 데이터를 로드합니다."""
    news_list = []

    # CSV 파일 읽기 (여러 인코딩 시도)
    for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        # 마지막 수단: 오류 무시
        df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore')

    # 최대 max_items개만 사용
    df = df.head(max_items)

    # 각 행을 NewsItem으로 변환
    for idx, row in df.iterrows():
        news = NewsItem(
            news_id=str(row['뉴스 식별자']),
            date=str(row['일자']),
            publisher=str(row['언론사']),
            title=str(row['제목']),
            content=str(row['본문'])[:500],  # 본문은 500자로 제한
            url=str(row['URL'])
        )
        news_list.append(news)

    return news_list


# ═══════════════════════════════════════════════════════════════════════════
# 관련 뉴스 가져오기
# ═══════════════════════════════════════════════════════════════════════════

def get_relevant_news(query: str, news_data: list, top_k: int = 5) -> list:
    """쿼리와 관련된 뉴스를 검색합니다."""
    # BM25 검색 사용
    results = bm25_search(query, news_data, top_k)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 뉴스 데이터 포맷팅
# ═══════════════════════════════════════════════════════════════════════════

def format_news_data(news_results: list) -> str:
    """검색된 뉴스를 문자열로 포맷팅합니다."""
    formatted_list = []

    for news, score in news_results:
        formatted = f"제목: {news.title}, 언론사: {news.publisher}, 날짜: {news.date}\n내용: {news.content[:200]}..."
        formatted_list.append(formatted)

    return "\n\n".join(formatted_list)


# ═══════════════════════════════════════════════════════════════════════════
# BM25 검색
# ═══════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    """간단한 토크나이저"""
    text = text.lower()
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    tokens = text.split()
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                 '은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로', '와', '과', '도', '한', '있다', '하다'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def bm25_score(query_tokens: list, doc_tokens: list,
               avg_doc_len: float, doc_count: int, doc_freqs: dict,
               k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 스코어 계산"""
    score = 0.0
    doc_len = len(doc_tokens)
    doc_token_counts = Counter(doc_tokens)

    for token in query_tokens:
        if token not in doc_token_counts:
            continue
        tf = doc_token_counts[token]
        df = doc_freqs.get(token, 0)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)

    return score


def bm25_search(query: str, news_data: list, top_k: int = 5) -> list:
    """BM25 알고리즘을 사용하여 관련 뉴스를 검색합니다."""
    if not news_data:
        return []

    # 1. 쿼리 토큰화
    query_tokens = tokenize(query)

    # 2. 모든 뉴스의 텍스트 토큰화 (제목 + 내용)
    news_tokens_list = [tokenize(news.title + " " + news.content) for news in news_data]

    # 3. 문서 빈도 계산 (IDF용)
    doc_freqs = Counter()
    for tokens in news_tokens_list:
        for token in set(tokens):
            doc_freqs[token] += 1

    # 4. 평균 문서 길이 계산
    avg_doc_len = sum(len(t) for t in news_tokens_list) / len(news_data)

    # 5. 각 뉴스에 대해 BM25 스코어 계산
    results = []
    for news, doc_tokens in zip(news_data, news_tokens_list):
        score = bm25_score(query_tokens, doc_tokens, avg_doc_len, len(news_data), doc_freqs)
        results.append((news, score))

    # 6. 점수순 정렬 및 상위 top_k개 반환
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Search
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list, b: list) -> float:
    """코사인 유사도 계산"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(query: str, news_data: list, llm, top_k: int = 5) -> list:
    """임베딩 기반 시맨틱 검색을 수행합니다."""
    if not news_data:
        return []

    # 1. 쿼리 임베딩 생성
    query_embedding = llm.get_embedding(query)

    # 2. 각 뉴스와 유사도 계산
    results = []
    for news in news_data:
        if news.embedding:
            sim = cosine_similarity(query_embedding, news.embedding)
            results.append((news, sim))

    # 3. 점수순 정렬 및 상위 top_k개 반환
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# RAG 파이프라인
# ═══════════════════════════════════════════════════════════════════════════

def generate_rag_answer(query: str, news_data: list, llm, use_semantic: bool = False) -> dict:
    """RAG 파이프라인: 검색 + 생성"""

    # 1. 관련 뉴스 검색
    if use_semantic:
        relevant_news = semantic_search(query, news_data, llm, top_k=3)
    else:
        relevant_news = get_relevant_news(query, news_data, top_k=3)

    if not relevant_news:
        return {
            "answer": "관련 뉴스를 찾을 수 없습니다.",
            "sources": []
        }

    # 2. 컨텍스트 포맷팅
    context = format_news_data(relevant_news)

    # 3. 프롬프트 생성 및 답변 생성
    prompt = f"""다음 뉴스 기사들을 참고하여 질문에 답변하세요. 한국어로 답변해주세요.

뉴스 기사:
{context}

질문: {query}

답변:"""

    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "sources": [(news.title, news.publisher, news.date) for news, _ in relevant_news]
    }


# === LLM 클래스 ===
def get_secret(key: str):
    """API 키를 가져옵니다. (환경변수 > Streamlit secrets)"""
    # 환경변수 먼저 확인
    value = os.getenv(key)
    if value:
        return value

    # Streamlit secrets 확인
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return None


class GeminiLLM:
    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai
        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY를 설정하세요")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = "text-embedding-004"

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def get_embedding(self, text: str) -> list:
        response = self.client.models.embed_content(model=self.embed_model, contents=text)
        return response.embeddings[0].values


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        from openai import OpenAI
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY를 설정하세요")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> list:
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding


def create_llm():
    """LLM 인스턴스를 생성합니다. (GOOGLE_API_KEY 우선)"""
    if get_secret("GOOGLE_API_KEY"):
        st.sidebar.success("✅ Gemini API 사용")
        return GeminiLLM()
    elif get_secret("OPENAI_API_KEY"):
        st.sidebar.success("✅ OpenAI API 사용")
        return OpenAILLM()
    else:
        raise ValueError("API 키를 설정하세요 (GOOGLE_API_KEY 또는 OPENAI_API_KEY)")


# === 사이드바 ===
with st.sidebar:
    st.header("📰 심리 뉴스 RAG")

    st.divider()

    # 데이터 로드 설정
    max_news = st.slider("로드할 뉴스 수", 10, 200, 50, 10)

    if st.button("🚀 데이터 로드", type="primary", use_container_width=True):
        with st.spinner("초기화 중..."):
            try:
                # 데이터 로드
                news_data = load_news_data(DATA_PATH, max_items=max_news)
                st.session_state.news_data = news_data

                # LLM 초기화
                llm = create_llm()
                st.session_state.llm = llm

                st.success(f"✅ {len(news_data)}개 뉴스 로드 완료!")

            except Exception as e:
                st.error(f"❌ 오류: {e}")

    st.divider()

    # 검색 방식 선택
    search_method = st.radio(
        "검색 방식",
        ["BM25 (키워드)", "Semantic (임베딩)"],
        index=0
    )

    # 임베딩 생성 (Semantic Search용)
    if st.session_state.news_data and st.session_state.llm:
        if search_method == "Semantic (임베딩)" and not st.session_state.embeddings_ready:
            if st.button("🧠 임베딩 생성", use_container_width=True):
                with st.spinner("임베딩 생성 중..."):
                    progress = st.progress(0)
                    for i, news in enumerate(st.session_state.news_data):
                        text = news.title + " " + news.content[:200]
                        news.embedding = st.session_state.llm.get_embedding(text)
                        progress.progress((i + 1) / len(st.session_state.news_data))
                    st.session_state.embeddings_ready = True
                    st.success("✅ 임베딩 생성 완료!")

    st.divider()

    # 버튼 영역
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔄 새 대화", on_click=reset_chat, use_container_width=True)
    with col2:
        st.button("🗑️ 전체 초기화", on_click=reset_all, use_container_width=True)

    # 상태 표시
    if st.session_state.news_data:
        st.success(f"📚 {len(st.session_state.news_data)}개 뉴스 준비됨")
        if st.session_state.embeddings_ready:
            st.success("🧠 임베딩 준비됨")


# === 메인 영역 ===
st.title("📰 RAG 챗봇 - 심리 뉴스")

st.markdown("""
심리 관련 뉴스 데이터를 기반으로 질문에 답변합니다.

**예시 질문:**
- "정신건강 관련 최신 뉴스는?"
- "심리상담 트렌드는?"
- "우울증 치료 관련 뉴스"
- "청소년 심리 문제"
- "직장인 스트레스 관련 기사"
""")

st.divider()

if not st.session_state.news_data:
    st.info("👈 사이드바에서 '데이터 로드'를 클릭하세요.")
else:
    # 데이터 미리보기
    with st.expander("📊 로드된 뉴스 미리보기"):
        for i, news in enumerate(st.session_state.news_data[:5]):
            st.markdown(f"**{i+1}. {news.title}**")
            st.caption(f"{news.publisher} | {news.date}")
            st.write(news.content[:150] + "...")
            st.divider()

    # 대화 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 참조 뉴스"):
                    for title, publisher, date in msg["sources"]:
                        st.markdown(f"**{title}** ({publisher}, {date})")

    # 사용자 입력
    if user_input := st.chat_input("심리 관련 뉴스에 대해 질문하세요"):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "avatar": AVATAR_USER
        })
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=AVATAR_BOT):
            with st.spinner("답변 생성 중..."):
                try:
                    use_semantic = (search_method == "Semantic (임베딩)" and st.session_state.embeddings_ready)

                    result = generate_rag_answer(
                        user_input,
                        st.session_state.news_data,
                        st.session_state.llm,
                        use_semantic=use_semantic
                    )
                    response = result["answer"]
                    sources = result["sources"]

                    st.markdown(response)

                    if sources:
                        with st.expander("📚 참조 뉴스"):
                            for title, publisher, date in sources:
                                st.markdown(f"**{title}** ({publisher}, {date})")

                except Exception as e:
                    response = f"⚠️ 오류: {str(e)}"
                    sources = []
                    st.error(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": AVATAR_BOT,
            "sources": sources
        })

        st.rerun()
