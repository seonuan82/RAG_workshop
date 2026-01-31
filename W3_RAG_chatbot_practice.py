"""
RAG 챗봇 실습 파일 - 심리 뉴스 검색
====================================
실행: streamlit run rag_chatbot_practice.py

📝 실습 목표:
1. 뉴스 데이터 로드 및 전처리
2. get_relevant_news() 함수 구현 (C1M1 스타일)
3. format_news_data() 함수 구현 (C1M1 스타일)
4. bm25_search() 함수 구현 (C1M2 스타일)
5. semantic_search() 함수 구현 (C1M2 스타일)

💡 데이터: Practice_data_NewsResult.CSV (심리 키워드 뉴스 3개월치)

🔍 예시 질문:
- "정신건강 관련 최신 뉴스는?"
- "심리상담 트렌드는?"
- "우울증 치료 관련 뉴스"
- "청소년 심리 문제"
- "직장인 스트레스 관련 기사"
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# === 페이지 설정 ===
st.set_page_config(
    page_title="RAG 챗봇 실습 - 심리 뉴스",
    page_icon="📰",
    layout="wide"
)

# === 설정 ===
AVATAR_USER = "👤"
AVATAR_BOT = "🤖"
DATA_PATH = os.path.join(os.path.dirname(__file__), "Practice_data_NewsResult.CSV")


# === 데이터 클래스 ===
@dataclass
class NewsItem:
    """뉴스 데이터 클래스"""
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


# ═══════════════════════════════════════════════════════════════════════════
# 실습 1: 뉴스 데이터 로드 함수
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
            date=str(row['일자']),
            publisher=str(row['언론사']),
            title=str(row['제목']),
            content=str(row['본문'])[:500],  # 본문은 500자로 제한
            url=str(row['URL'])
        )
        news_list.append(news)

    return news_list


# ═══════════════════════════════════════════════════════════════════════════
# 실습 2: 관련 뉴스 가져오기 (C1M1 Exercise 1 스타일)
# ═══════════════════════════════════════════════════════════════════════════

def get_relevant_news(query: str, news_data: list, top_k: int = 5) -> list:
    """
    쿼리와 관련된 뉴스를 검색합니다.

    Args:
        query: 검색 쿼리
        news_data: NewsItem 리스트
        top_k: 반환할 뉴스 수

    Returns:
        관련 뉴스 리스트 [(NewsItem, score), ...]

    💡 힌트:
    - bm25_search() 또는 semantic_search() 함수 사용
    - 결과를 점수 순으로 정렬하여 상위 top_k개 반환
    """
    # TODO: 관련 뉴스 검색 구현
    # ──────────────────────────────────────────
    # 1. 검색 함수 호출 (bm25_search 또는 semantic_search)
    #    results = bm25_search(query, news_data, top_k)
    #
    # 2. 결과 반환
    #    return results
    # ──────────────────────────────────────────

    return []


# ═══════════════════════════════════════════════════════════════════════════
# 실습 3: 뉴스 데이터 포맷팅 (C1M1 Exercise 2 스타일)
# ═══════════════════════════════════════════════════════════════════════════

def format_news_data(news_results: list) -> str:
    """
    검색된 뉴스를 문자열로 포맷팅합니다.

    Args:
        news_results: [(NewsItem, score), ...] 형태의 리스트

    Returns:
        포맷팅된 문자열

    💡 힌트:
    - 각 뉴스의 제목, 날짜, 언론사, 내용을 포함
    - 예시 형식:
      "제목: {title}, 언론사: {publisher}, 날짜: {date}
       내용: {content}..."
    """
    # TODO: 뉴스 포맷팅 구현
    # ──────────────────────────────────────────
    # 1. 빈 리스트 생성
    #    formatted_list = []
    #
    # 2. 각 뉴스 포맷팅
    #    for news, score in news_results:
    #        formatted = f"제목: {news.title}, 언론사: {news.publisher}, 날짜: {news.date}\n내용: {news.content[:200]}..."
    #        formatted_list.append(formatted)
    #
    # 3. 줄바꿈으로 연결하여 반환
    #    return "\n\n".join(formatted_list)
    # ──────────────────────────────────────────

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 실습 4: BM25 검색 (C1M2 Exercise 1 스타일)
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
    """BM25 스코어 계산 (이 함수는 제공됨)"""
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
    """
    BM25 알고리즘을 사용하여 관련 뉴스를 검색합니다.

    Args:
        query: 검색 쿼리
        news_data: NewsItem 리스트
        top_k: 반환할 결과 수

    Returns:
        [(NewsItem, score), ...] 리스트

    💡 힌트 (C1M2 Exercise 1 참고):
    1. 쿼리 토큰화: tokenize(query)
    2. 모든 뉴스 토큰화: [tokenize(news.title + " " + news.content) for news in news_data]
    3. 문서 빈도(doc_freqs) 계산
    4. 평균 문서 길이 계산
    5. 각 뉴스에 대해 bm25_score() 계산
    6. 점수순 정렬 후 상위 top_k개 반환
    """
    if not news_data:
        return []

    # TODO: BM25 검색 구현
    # ──────────────────────────────────────────
    # 1. 쿼리 토큰화
    #    query_tokens = tokenize(query)
    #
    # 2. 모든 뉴스의 텍스트 토큰화 (제목 + 내용)
    #    news_tokens_list = [tokenize(news.title + " " + news.content) for news in news_data]
    #
    # 3. 문서 빈도 계산 (IDF용)
    #    doc_freqs = Counter()
    #    for tokens in news_tokens_list:
    #        for token in set(tokens):
    #            doc_freqs[token] += 1
    #
    # 4. 평균 문서 길이 계산
    #    avg_doc_len = sum(len(t) for t in news_tokens_list) / len(news_data)
    #
    # 5. 각 뉴스에 대해 BM25 스코어 계산
    #    results = []
    #    for news, doc_tokens in zip(news_data, news_tokens_list):
    #        score = bm25_score(query_tokens, doc_tokens, avg_doc_len, len(news_data), doc_freqs)
    #        results.append((news, score))
    #
    # 6. 점수순 정렬 및 상위 top_k개 반환
    #    results.sort(key=lambda x: x[1], reverse=True)
    #    return results[:top_k]
    # ──────────────────────────────────────────

    return []


# ═══════════════════════════════════════════════════════════════════════════
# 실습 5: Semantic Search (C1M2 Exercise 2 스타일)
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list, b: list) -> float:
    """코사인 유사도 계산 (이 함수는 제공됨)"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(query: str, news_data: list, llm, top_k: int = 5) -> list:
    """
    임베딩 기반 시맨틱 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        news_data: NewsItem 리스트 (embedding 필드 필요)
        llm: LLM 인스턴스 (get_embedding 메서드 필요)
        top_k: 반환할 결과 수

    Returns:
        [(NewsItem, score), ...] 리스트

    💡 힌트 (C1M2 Exercise 2 참고):
    1. 쿼리 임베딩 생성: llm.get_embedding(query)
    2. 각 뉴스의 임베딩과 코사인 유사도 계산
    3. 점수순 정렬 후 상위 top_k개 반환
    """
    if not news_data:
        return []

    # TODO: Semantic Search 구현
    # ──────────────────────────────────────────
    # 1. 쿼리 임베딩 생성
    #    query_embedding = llm.get_embedding(query)
    #
    # 2. 각 뉴스와 유사도 계산
    #    results = []
    #    for news in news_data:
    #        if news.embedding:
    #            sim = cosine_similarity(query_embedding, news.embedding)
    #            results.append((news, sim))
    #
    # 3. 점수순 정렬 및 상위 top_k개 반환
    #    results.sort(key=lambda x: x[1], reverse=True)
    #    return results[:top_k]
    # ──────────────────────────────────────────

    return []


# ═══════════════════════════════════════════════════════════════════════════
# RAG 파이프라인 (제공됨)
# ═══════════════════════════════════════════════════════════════════════════

def generate_rag_answer(query: str, news_data: list, llm) -> dict:
    """RAG 파이프라인: 검색 + 생성"""

    # 1. 관련 뉴스 검색
    relevant_news = get_relevant_news(query, news_data, top_k=3)

    if not relevant_news:
        return {
            "answer": "⚠️ 관련 뉴스를 찾을 수 없습니다. 검색 함수를 구현해주세요.",
            "sources": []
        }

    # 2. 컨텍스트 포맷팅
    context = format_news_data(relevant_news)

    if not context:
        return {
            "answer": "⚠️ 뉴스 포맷팅이 되지 않았습니다. format_news_data() 함수를 구현해주세요.",
            "sources": []
        }

    # 3. 프롬프트 생성 및 답변 생성
    prompt = f"""다음 뉴스 기사들을 참고하여 질문에 답변하세요.

뉴스 기사:
{context}

질문: {query}

답변:"""

    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "sources": [(news.title, news.publisher, news.date) for news, _ in relevant_news]
    }


# === LLM 클래스 (rag_workshop.py에서 가져옴) ===
def get_secret(key: str):
    try:
        return st.secrets.get(key) or os.getenv(key)
    except:
        return os.getenv(key)


class GeminiLLM:
    def __init__(self, model: str = "gemini-2.5-flash"):
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
    def __init__(self, model: str = "gpt-5", embedding_model: str = "text-embedding-3-small"):
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
    if get_secret("GOOGLE_API_KEY"):
        return GeminiLLM()
    elif get_secret("OPENAI_API_KEY"):
        return OpenAILLM()
    else:
        raise ValueError("API 키를 설정하세요 (GOOGLE_API_KEY 또는 OPENAI_API_KEY)")


# === 사이드바 ===
with st.sidebar:
    st.header("📰 심리 뉴스 RAG 실습")

    st.markdown("""
    ### 실습 순서
    1. **load_news_data()** - 뉴스 로드
    2. **get_relevant_news()** - 검색
    3. **format_news_data()** - 포맷팅
    4. **bm25_search()** - BM25 검색
    5. **semantic_search()** - 시맨틱 검색
    """)

    st.divider()

    # 데이터 로드
    max_news = st.slider("로드할 뉴스 수", 10, 200, 50, 10)

    if st.button("🚀 데이터 로드 및 초기화", type="primary", use_container_width=True):
        with st.spinner("초기화 중..."):
            try:
                # 데이터 로드
                news_data = load_news_data(DATA_PATH, max_items=max_news)

                if not news_data:
                    st.warning("⚠️ load_news_data() 함수를 구현하세요!")
                else:
                    st.session_state.news_data = news_data

                    # LLM 초기화
                    llm = create_llm()
                    st.session_state.llm = llm

                    st.success(f"✅ {len(news_data)}개 뉴스 로드 완료!")

            except Exception as e:
                st.error(f"❌ 오류: {e}")

    st.divider()

    # 임베딩 생성 (선택사항)
    if st.session_state.news_data and st.session_state.llm:
        if st.button("🧠 임베딩 생성 (Semantic Search용)", use_container_width=True):
            with st.spinner("임베딩 생성 중..."):
                progress = st.progress(0)
                for i, news in enumerate(st.session_state.news_data):
                    text = news.title + " " + news.content[:200]
                    news.embedding = st.session_state.llm.get_embedding(text)
                    progress.progress((i + 1) / len(st.session_state.news_data))
                st.session_state.embeddings_ready = True
                st.success("✅ 임베딩 생성 완료!")

    st.divider()
    st.button("🔄 대화 초기화", on_click=reset_chat, use_container_width=True)

    # 상태 표시
    if st.session_state.news_data:
        st.success(f"📚 {len(st.session_state.news_data)}개 뉴스 준비됨")
        if st.session_state.embeddings_ready:
            st.success("🧠 임베딩 준비됨")


# === 메인 영역 ===
st.title("📰 RAG 챗봇 실습 - 심리 뉴스")

st.markdown("""
### 실습 안내

이 실습에서는 **심리 관련 뉴스 데이터**를 사용하여 RAG 시스템을 구현합니다.

#### 📋 구현해야 할 함수들:
1. `load_news_data()` - CSV에서 뉴스 데이터 로드
2. `get_relevant_news()` - 관련 뉴스 검색 (C1M1 Exercise 1)
3. `format_news_data()` - 뉴스 포맷팅 (C1M1 Exercise 2)
4. `bm25_search()` - BM25 키워드 검색 (C1M2 Exercise 1)
5. `semantic_search()` - 시맨틱 검색 (C1M2 Exercise 2)

#### 🔍 예시 질문:
- "정신건강 관련 최신 뉴스는?"
- "심리상담 트렌드는?"
- "우울증 치료 관련 뉴스"
- "청소년 심리 문제"
- "직장인 스트레스 관련 기사"
- "심리치료사 관련 정책"

💡 완성된 코드는 `rag_chatbot.py`와 `rag_workshop.py`를 참고하세요.
""")

st.divider()

if not st.session_state.news_data:
    st.info("👈 사이드바에서 '데이터 로드 및 초기화'를 클릭하세요.")
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
                    result = generate_rag_answer(
                        user_input,
                        st.session_state.news_data,
                        st.session_state.llm
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
