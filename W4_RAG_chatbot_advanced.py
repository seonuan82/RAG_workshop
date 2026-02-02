"""
RAG 챗봇 - 심리 뉴스 검색 (Hybrid Search 포함)
========================================================

심리/뇌과학 관련 뉴스 데이터를 기반으로 질문에 답변하는 RAG 챗봇
BM25, Semantic, Hybrid 검색 방식 지원
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
    page_title="RAG 챗봇 - Hybrid Search",
    page_icon="📰",
    layout="wide"
)

# === 설정 ===
AVATAR_USER = "👤"
AVATAR_BOT = "🤖"
CSV_FILENAME = "Practice_data_NewsResult.CSV"
GITHUB_CSV_URL = "https://raw.githubusercontent.com/seonuan82/RAG_Workshop/main/Practice_data_NewsResult.CSV"


def get_data_path():
    """데이터 파일 경로를 반환합니다. (로컬 > 현재 디렉토리 > None)"""
    try:
        current_dir = Path(__file__).parent
        local_path = current_dir / CSV_FILENAME
        if local_path.exists():
            return str(local_path)
    except:
        pass

    cloud_path = Path(CSV_FILENAME)
    if cloud_path.exists():
        return str(cloud_path)

    return None


def load_news_from_github(max_items: int = 100) -> list:
    """GitHub에서 뉴스 CSV 데이터를 다운로드하여 로드합니다."""
    import urllib.request
    import io

    st.info(f"📥 GitHub에서 뉴스 데이터 다운로드 중...")

    try:
        with urllib.request.urlopen(GITHUB_CSV_URL) as response:
            content = response.read()

        st.info(f"📦 다운로드 완료: {len(content):,} bytes")

        decoded = content.decode('cp949', errors='replace')
        df = pd.read_csv(io.StringIO(decoded))

        st.success(f"✅ 데이터 로드 성공")
        st.info(f"📊 총 {len(df)}개 행")

        return _parse_news_dataframe(df, max_items)

    except Exception as e:
        st.error(f"❌ GitHub 다운로드 실패: {e}")
        return []


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
    if "pending_response" not in st.session_state:
        st.session_state.pending_response = False
    if "search_method" not in st.session_state:
        st.session_state.search_method = "BM25"
    if "hybrid_alpha" not in st.session_state:
        st.session_state.hybrid_alpha = 0.5


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

def _parse_news_dataframe(df: pd.DataFrame, max_items: int = 100) -> list:
    """DataFrame을 NewsItem 리스트로 변환합니다."""
    news_list = []
    df = df.head(max_items)

    col_mapping = {
        'news_id': ['뉴스 식별자', '기사 고유번호', 'news_id', 'id'],
        'date': ['일자', 'date', '날짜'],
        'publisher': ['언론사', 'publisher', '매체'],
        'title': ['제목', 'title'],
        'content': ['본문', 'content', '내용'],
        'url': ['URL', 'url', '링크']
    }

    def find_column(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    id_col = find_column(col_mapping['news_id'])
    date_col = find_column(col_mapping['date'])
    publisher_col = find_column(col_mapping['publisher'])
    title_col = find_column(col_mapping['title'])
    content_col = find_column(col_mapping['content'])
    url_col = find_column(col_mapping['url'])

    if not title_col or not content_col:
        st.error(f"❌ 필수 컬럼을 찾을 수 없습니다.")
        return []

    for _, row in df.iterrows():
        try:
            news = NewsItem(
                news_id=str(row.get(id_col, '')) if id_col else '',
                date=str(row.get(date_col, '')) if date_col else '',
                publisher=str(row.get(publisher_col, '')) if publisher_col else '',
                title=str(row.get(title_col, '')),
                content=str(row.get(content_col, '')),
                url=str(row.get(url_col, '')) if url_col else ''
            )
            news_list.append(news)
        except Exception:
            continue

    st.success(f"✅ {len(news_list)}개 뉴스 파싱 완료")
    return news_list


def load_news_data(filepath: Optional[str] = None, max_items: int = 100) -> list:
    """GitHub에서 뉴스 데이터를 로드합니다."""
    return load_news_from_github(max_items)


# ═══════════════════════════════════════════════════════════════════════════
# 뉴스 데이터 포맷팅
# ═══════════════════════════════════════════════════════════════════════════

def format_news_data(news_results: list) -> str:
    """검색된 뉴스를 문자열로 포맷팅합니다."""
    formatted_list = []

    for news, score in news_results:
        formatted = f"제목: {news.title}\n언론사: {news.publisher}\n날짜: {news.date}\n본문: {news.content}"
        formatted_list.append(formatted)

    return "\n\n---\n\n".join(formatted_list)


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


def bm25_search(query: str, news_data: list, top_k: int = 5, k1: float = None, b: float = None) -> list:
    """BM25 알고리즘을 사용하여 관련 뉴스를 검색합니다."""
    if not news_data:
        return []

    if k1 is None:
        k1 = BM25_K1 if 'BM25_K1' in globals() else 1.5
    if b is None:
        b = BM25_B if 'BM25_B' in globals() else 0.75

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
        score = bm25_score(query_tokens, doc_tokens, avg_doc_len, len(news_data), doc_freqs, k1=k1, b=b)
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
# Hybrid Search (NEW!)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_scores(results: list) -> list:
    """
    점수를 0~1 범위로 정규화합니다. (Min-Max Normalization)

    Args:
        results: [(NewsItem, score), ...] 리스트

    Returns:
        [(NewsItem, normalized_score), ...] 리스트
    """
    if not results:
        return []

    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)

    # 모든 점수가 같으면 1로 설정
    if max_score == min_score:
        return [(news, 1.0) for news, _ in results]

    normalized = []
    for news, score in results:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((news, norm_score))

    return normalized


def hybrid_search(query: str, news_data: list, llm, top_k: int = 5, alpha: float = 0.5) -> list:
    """
    BM25와 Semantic Search를 결합한 하이브리드 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        news_data: NewsItem 리스트
        llm: LLM 인스턴스
        top_k: 반환할 결과 수
        alpha: BM25 가중치 (0~1). 1에 가까울수록 BM25 중심, 0에 가까울수록 Semantic 중심

    Returns:
        [(NewsItem, score), ...] 리스트

    📊 Alpha 값에 따른 특성:
    - alpha = 1.0: BM25만 사용 (키워드 매칭 중심)
    - alpha = 0.5: 균형 있는 하이브리드
    - alpha = 0.0: Semantic만 사용 (의미 유사도 중심)
    """
    if not news_data:
        return []

    # 1. BM25 검색 수행 (전체 문서에 대해)
    bm25_results = bm25_search(query, news_data, top_k=len(news_data))
    bm25_normalized = normalize_scores(bm25_results)

    # 2. Semantic 검색 수행 (전체 문서에 대해)
    semantic_results = semantic_search(query, news_data, llm, top_k=len(news_data))
    semantic_normalized = normalize_scores(semantic_results)

    # 3. 점수 딕셔너리 생성
    bm25_scores = {news.news_id: score for news, score in bm25_normalized}
    semantic_scores = {news.news_id: score for news, score in semantic_normalized}

    # 4. 하이브리드 점수 계산
    hybrid_results = []
    for news in news_data:
        bm25_s = bm25_scores.get(news.news_id, 0)
        sem_s = semantic_scores.get(news.news_id, 0)
        final_score = alpha * bm25_s + (1 - alpha) * sem_s
        hybrid_results.append((news, final_score))

    # 5. 점수순 정렬 및 상위 top_k개 반환
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    return hybrid_results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# 관련 뉴스 가져오기 (검색 방식 통합)
# ═══════════════════════════════════════════════════════════════════════════

def get_relevant_news(query: str, news_data: list, llm=None, top_k: int = 5,
                      search_method: str = "BM25", alpha: float = 0.5) -> list:
    """
    쿼리와 관련된 뉴스를 검색합니다.

    Args:
        query: 검색 쿼리
        news_data: NewsItem 리스트
        llm: LLM 인스턴스 (Semantic/Hybrid 검색 시 필요)
        top_k: 반환할 뉴스 수
        search_method: 검색 방식 ("BM25", "Semantic", "Hybrid")
        alpha: Hybrid 검색 시 BM25 가중치

    Returns:
        관련 뉴스 리스트 [(NewsItem, score), ...]
    """
    if search_method == "BM25":
        return bm25_search(query, news_data, top_k)
    elif search_method == "Semantic":
        return semantic_search(query, news_data, llm, top_k)
    elif search_method == "Hybrid":
        return hybrid_search(query, news_data, llm, top_k, alpha)
    else:
        return bm25_search(query, news_data, top_k)


# ═══════════════════════════════════════════════════════════════════════════
# RAG 파이프라인
# ═══════════════════════════════════════════════════════════════════════════

def generate_rag_answer(query: str, news_data: list, llm,
                        search_method: str = "BM25", alpha: float = 0.5, top_k: int = 3) -> dict:
    """RAG 파이프라인: 검색 + 생성"""

    # 1. 관련 뉴스 검색
    relevant_news = get_relevant_news(
        query, news_data, llm,
        top_k=top_k,
        search_method=search_method,
        alpha=alpha
    )

    if not relevant_news:
        return {
            "answer": "관련 뉴스를 찾을 수 없습니다.",
            "sources": [],
            "search_method": search_method
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
        "sources": [(news.title, news.publisher, news.date) for news, _ in relevant_news],
        "search_method": search_method
    }


# === LLM 클래스 ===
def get_secret(key: str):
    """API 키를 가져옵니다. (환경변수 > Streamlit secrets)"""
    value = os.getenv(key)
    if value:
        return value

    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return None


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


# === RAG 설정 (고정값) ===
MAX_NEWS_ITEMS = 1000  # 로드할 뉴스 수
TOP_K_RESULTS = 3      # 검색 결과 수
BM25_K1 = 1.5          # BM25 파라미터
BM25_B = 0.75          # BM25 파라미터

# === 사이드바 ===
with st.sidebar:
    st.header("📰 심리 뉴스 RAG")
    st.caption("Hybrid Search 지원")

    st.divider()

    # RAG 파라미터 표시
    with st.expander("⚙️ RAG 파라미터 (참조)", expanded=False):
        st.markdown(f"""
        | 파라미터 | 값 |
        |---------|-----|
        | 로드 뉴스 수 | **{MAX_NEWS_ITEMS}** |
        | 검색 결과 수 (top_k) | **{TOP_K_RESULTS}** |
        | BM25 k1 | **{BM25_K1}** |
        | BM25 b | **{BM25_B}** |
        """)

    if st.button("🚀 데이터 로드", type="primary", use_container_width=True):
        with st.spinner("초기화 중..."):
            try:
                news_data = load_news_data(max_items=MAX_NEWS_ITEMS)
                st.session_state.news_data = news_data

                llm = create_llm()
                st.session_state.llm = llm

                st.success(f"✅ {len(news_data)}개 뉴스 로드 완료!")

            except Exception as e:
                st.error(f"❌ 오류: {e}")

    st.divider()

    # 검색 방식 선택
    st.subheader("🔍 검색 방식")
    search_method = st.radio(
        "검색 알고리즘",
        ["BM25", "Semantic", "Hybrid"],
        index=0,
        help="BM25: 키워드 기반, Semantic: 의미 기반, Hybrid: 둘의 조합"
    )
    st.session_state.search_method = search_method

    # Hybrid 검색 시 alpha 값 조절
    if search_method == "Hybrid":
        alpha = st.slider(
            "Alpha (BM25 가중치)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="1.0: BM25만, 0.0: Semantic만, 0.5: 균형"
        )
        st.session_state.hybrid_alpha = alpha
        st.caption(f"📊 BM25: {alpha:.0%} / Semantic: {1-alpha:.0%}")

    st.divider()

    # 임베딩 생성 (Semantic/Hybrid용)
    if st.session_state.news_data and st.session_state.llm:
        if search_method in ["Semantic", "Hybrid"] and not st.session_state.embeddings_ready:
            st.warning("⚠️ 임베딩이 필요합니다.")
            if st.button("🧠 임베딩 생성", use_container_width=True):
                with st.spinner("임베딩 생성 중..."):
                    progress = st.progress(0)
                    for i, news in enumerate(st.session_state.news_data):
                        text = news.title + " " + news.content[:500]
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
st.title("📰 RAG 챗봇 - Hybrid Search")

st.markdown("심리 관련 뉴스 데이터를 기반으로 질문에 답변합니다."}
md1, md2 = st.columns(2)
with md1:
    st.markdown("""
#### 🔍 검색 방식 비교:
| 방식 | 장점 | 단점 |
|------|------|------|
| **BM25** | 정확한 키워드 매칭, 빠름 | 동의어/유사어 인식 못함 |
| **Semantic** | 의미적 유사성 파악 | 키워드 정확도 낮음, 느림 |
| **Hybrid** | 두 장점 결합 | 파라미터 튜닝 필요 |
""")
with md2:
    st.markdown("""
**예시 질문:**
- "정신건강 관련 최신 뉴스는?"
- "경제심리 관련 뉴스는?"
- "공직자 대상 정신건강"
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
            st.write(news.content[:200] + "...")
            st.divider()

    # 현재 검색 방식 표시
    method_emoji = {"BM25": "🔤", "Semantic": "🧠", "Hybrid": "⚡"}
    alpha_info = f" (Alpha: {st.session_state.hybrid_alpha})" if st.session_state.search_method == "Hybrid" else ""
    st.info(f"{method_emoji.get(st.session_state.search_method, '')} 현재 검색 방식: **{st.session_state.search_method}**{alpha_info}")

    # 대화 기록 표시 (스크롤 가능한 컨테이너)
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📚 참조 뉴스 ({msg.get('search_method', 'N/A')})"):
                        for title, publisher, date in msg["sources"]:
                            st.markdown(f"**{title}** ({publisher}, {date})")

        # 응답 생성 중인 경우 (컨테이너 안에서 처리)
        if st.session_state.pending_response:
            last_user_input = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant", avatar=AVATAR_BOT):
                with st.spinner(f"답변 생성 중... ({st.session_state.search_method})"):
                    try:
                        result = generate_rag_answer(
                            last_user_input,
                            st.session_state.news_data,
                            st.session_state.llm,
                            search_method=st.session_state.search_method,
                            alpha=st.session_state.hybrid_alpha,
                            top_k=TOP_K_RESULTS
                        )
                        response = result["answer"]
                        sources = result["sources"]
                        used_method = result.get("search_method", st.session_state.search_method)

                    except Exception as e:
                        response = f"⚠️ 오류: {str(e)}"
                        sources = []
                        used_method = st.session_state.search_method

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "avatar": AVATAR_BOT,
                "sources": sources,
                "search_method": used_method
            })
            st.session_state.pending_response = False
            st.rerun()

    # 자동 스크롤
    if st.session_state.messages:
        st.components.v1.html(
            """
            <script>
                const chatContainers = window.parent.document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
                chatContainers.forEach(container => {
                    const scrollable = container.querySelector('[data-testid="stVerticalBlock"]');
                    if (scrollable && scrollable.scrollHeight > scrollable.clientHeight) {
                        scrollable.scrollTop = scrollable.scrollHeight;
                    }
                });
            </script>
            """,
            height=0
        )

    # 사용자 입력
    if user_input := st.chat_input("질문하세요"):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "avatar": AVATAR_USER
        })
        st.session_state.pending_response = True
        st.rerun()

