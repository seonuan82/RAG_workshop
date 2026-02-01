"""
RAG 챗봇 실습 파일 - 심리 뉴스 검색
====================================
실행: streamlit run rag_chatbot_practice.py

📝 실습 목표:
1. get_relevant_news() 함수 구현
2. bm25_search() 함수 구현
3. semantic_search() 함수 구현

💡 데이터: Practice_data_NewsResult.CSV (심리 키워드 뉴스 1개월치)

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
from pathlib import Path
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
CSV_FILENAME = "Practice_data_NewsResult.CSV"
# GitHub Raw URL (저장소에 CSV 파일 업로드 후 이 URL 수정 필요)
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
    return None  # GitHub에서 다운로드 필요


def load_news_from_github(max_items: int = 100) -> list:
    """GitHub에서 뉴스 CSV 데이터를 다운로드하여 로드합니다."""
    import urllib.request
    import io

    st.info(f"📥 GitHub에서 뉴스 데이터 다운로드 중...")

    try:
        with urllib.request.urlopen(GITHUB_CSV_URL) as response:
            content = response.read()

        st.info(f"📦 다운로드 완료: {len(content):,} bytes")

        # cp949 인코딩으로 디코딩 (일부 잘못된 바이트는 대체)
        decoded = content.decode('cp949', errors='replace')
        df = pd.read_csv(io.StringIO(decoded))

        st.success(f"✅ 데이터 로드 성공")

        return _parse_news_dataframe(df, max_items)

    except Exception as e:
        st.error(f"❌ GitHub 다운로드 실패: {e}")
        return []


def _parse_news_dataframe(df: pd.DataFrame, max_items: int = 100) -> list:
    """DataFrame을 NewsItem 리스트로 변환합니다. (제공됨)"""
    news_list = []
    df = df.head(max_items)

    # 컬럼명 매핑 (다양한 CSV 포맷 지원)
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
        return candidates[0]

    id_col = find_column(col_mapping['news_id'])
    date_col = find_column(col_mapping['date'])
    publisher_col = find_column(col_mapping['publisher'])
    title_col = find_column(col_mapping['title'])
    content_col = find_column(col_mapping['content'])
    url_col = find_column(col_mapping['url'])

    for _, row in df.iterrows():
        try:
            news = NewsItem(
                news_id=str(row.get(id_col, '')),
                date=str(row.get(date_col, '')),
                publisher=str(row.get(publisher_col, '')),
                title=str(row.get(title_col, '')),
                content=str(row.get(content_col, ''))[:500],
                url=str(row.get(url_col, ''))
            )
            news_list.append(news)
        except:
            continue
    return news_list


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


init_session()


def reset_chat():
    """대화 초기화"""
    st.session_state.messages = []


def _parse_news_dataframe(df: pd.DataFrame, max_items: int = 100) -> list:
    """DataFrame을 NewsItem 리스트로 변환합니다."""
    news_list = []

    # 최대 max_items개만 사용
    df = df.head(max_items)

    # 컬럼명 확인 및 매핑 (다양한 CSV 포맷 지원) -------------------------------------------------- BM25 함수 뉴스 토큰화 때 참고!
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
        return None  # 찾지 못함

    id_col = find_column(col_mapping['news_id'])
    date_col = find_column(col_mapping['date'])
    publisher_col = find_column(col_mapping['publisher'])
    title_col = find_column(col_mapping['title'])
    content_col = find_column(col_mapping['content'])
    url_col = find_column(col_mapping['url'])

    # 컬럼 매핑 결과 표시
    st.info(f"🔍 컬럼 매핑: news_id={id_col}, date={date_col}, publisher={publisher_col}, title={title_col}, content={content_col}, url={url_col}")

    # 필수 컬럼 확인
    if not title_col or not content_col:
        st.error(f"❌ 필수 컬럼을 찾을 수 없습니다. 제목={title_col}, 본문={content_col}")
        st.error(f"📋 사용 가능한 컬럼: {list(df.columns)}")
        return []

    # 각 행을 NewsItem으로 변환
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
        except Exception as e:
            continue

    st.success(f"✅ {len(news_list)}개 뉴스 파싱 완료")
    return news_list


def load_news_data(filepath: Optional[str] = None, max_items: int = 100) -> list:
    """
    GitHub에서 뉴스 데이터를 로드합니다.

    Args:
        filepath: 사용하지 않음 (호환성 유지용)
        max_items: 로드할 최대 뉴스 수

    Returns:
        NewsItem 리스트
    """
    # GitHub에서 다운로드
    return load_news_from_github(max_items)

def format_news_data(news_results: list) -> str:
    """검색된 뉴스를 문자열로 포맷팅합니다."""
    formatted_list = []

    for news, score in news_results:
        formatted = f"제목: {news.title}\n언론사: {news.publisher}\n날짜: {news.date}\n본문: {news.content}"
        formatted_list.append(formatted)

    return "\n\n---\n\n".join(formatted_list)


# ═══════════════════════════════════════════════════════════════════════════
# 실습 1: BM25 검색 함수 구현
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

    💡 힌트:
    1. 쿼리 토큰화: tokenize(query)
    2. 모든 뉴스 토큰화: [tokenize() for 00 in 000]
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
    #    query_tokens = 00000000
    #
    # 2. 모든 뉴스의 텍스트 토큰화 (제목 + 내용)
    #    news_tokens_list = 00000000
    #
    # 3. 문서 빈도 계산 (IDF용)
    #    0000
    #
    # 4. 평균 문서 길이 계산
    #    avg_doc_len = 000000000
    #
    # 5. 각 뉴스에 대해 BM25 스코어 계산
    #    results = []
    #    00000000
    #
    # 6. 점수순 정렬 및 상위 top_k개 반환
    #    0000000000
    #    return 00000000
    # ──────────────────────────────────────────

    return []


# ═══════════════════════════════════════════════════════════════════════════
# 실습 2: Semantic Search 함수 구현
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

    💡 힌트:
    1. 쿼리 임베딩 생성: llm.get_embedding(query)
    2. 각 뉴스의 임베딩과 코사인 유사도 계산
    3. 점수순 정렬 후 상위 top_k개 반환
    """
    if not news_data:
        return []

    # TODO: Semantic Search 구현
    # ──────────────────────────────────────────
    # 1. 쿼리 임베딩 생성
    #    query_embedding = 0000000
    #
    # 2. 각 뉴스와 유사도 계산
    #    results = []
    #    000000000
    #
    # 3. 점수순 정렬 및 상위 top_k개 반환
    #    000000000
    #    return 00000000
    # ──────────────────────────────────────────

    return []


# ═══════════════════════════════════════════════════════════════════════════
# 실습 3: 관련 뉴스 가져오기
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
    
    추가 실습
    - top_k 수정해 보기!
    """
    # TODO: 관련 뉴스 검색 구현
    # ──────────────────────────────────────────
    # 1. 검색 함수 호출 (bm25_search 또는 semantic_search)
    #    results = 00000000
    #
    # 2. 결과 반환
    #    return 00000000
    # ──────────────────────────────────────────

    return []




# ═══════════════════════════════════════════════════════════════════════════
# 선택 실습: top_k 등 RAG 파라미터 수정해 보기
# ═══════════════════════════════════════════════════════════════════════════

def generate_rag_answer(query: str, news_data: list, llm) -> dict:
    """RAG 파이프라인: 검색 + 생성"""

    # 1. 관련 뉴스 검색
    relevant_news = get_relevant_news(query, news_data, top_k=3)  # ---------------------- top_k 수정 가능

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

    # 3. 프롬프트 생성 및 답변 생성 # --------------------------------------------------------- 챗봇 프롬프트 수정 가능
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
    def __init__(self, model: str = "gpt-5-mini", embedding_model: str = "text-embedding-3-small"):
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
    st.header("📰 심리 뉴스 RAG 실습")

    st.markdown("""
    ### 실습 순서
    1. **get_relevant_news()** - 검색
    2. **bm25_search()** - BM25 검색
    3. **semantic_search()** - 시맨틱 검색
    """)

    st.divider()

    # 데이터 로드
    max_news = st.slider("로드할 뉴스 수", 10, 200, 50, 10)

    if st.button("🚀 데이터 로드 및 초기화", type="primary", use_container_width=True):
        with st.spinner("초기화 중..."):
            try:
                # 데이터 로드
                news_data = load_news_data(max_items=max_news)

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


# === 메인 영역 (앱 UI) ===
st.title("📰 RAG 챗봇 실습 - 심리 뉴스")

st.markdown("""
### 실습: **심리 관련 뉴스 데이터**를 사용하여 RAG 시스템을 구현하는 챗봇 만들어 보기.

#### 📋 구현해야 할 함수들:
1. `get_relevant_news()` - 관련 뉴스 검색
2. `bm25_search()` - BM25 키워드 검색
3. `semantic_search()` - 시맨틱 검색

💡 구글 코랩을 활용해서 실습해 보세요!

#### 🔍 챗봇에게 물어볼 수 있는 질문 예시:
- "정신건강 관련 최신 뉴스는?"
- "심리상담 트렌드는?"
- "우울증 치료 관련 뉴스"
- "청소년 심리 문제"
- "직장인 스트레스 관련 기사"
- "심리치료사 관련 정책"
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

    # 대화 기록 표시 (스크롤 가능한 컨테이너)
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 참조 뉴스"):
                        for title, publisher, date in msg["sources"]:
                            st.markdown(f"**{title}** ({publisher}, {date})")

        # 응답 생성 중인 경우 (컨테이너 안에서 처리)
        if st.session_state.pending_response:
            last_user_input = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant", avatar=AVATAR_BOT):
                with st.spinner("답변 생성 중..."):
                    try:
                        result = generate_rag_answer(
                            last_user_input,
                            st.session_state.news_data,
                            st.session_state.llm
                        )
                        response = result["answer"]
                        sources = result["sources"]

                    except Exception as e:
                        response = f"⚠️ 오류: {str(e)}"
                        sources = []

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "avatar": AVATAR_BOT,
                "sources": sources
            })
            st.session_state.pending_response = False
            st.rerun()

    # 자동 스크롤 (새 메시지 입력 시)
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
    if user_input := st.chat_input("심리 관련 뉴스에 대해 질문하세요"):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "avatar": AVATAR_USER
        })
        st.session_state.pending_response = True
        st.rerun()
