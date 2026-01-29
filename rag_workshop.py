"""
RAG Workshop - OECD Digital Government Review of Korea (2025)
==============================================================

이 모듈은 OECD 한국 디지털 정부 리뷰 (2025) 문서를 활용한
RAG(Retrieval-Augmented Generation) 실습을 위한 코드입니다.

이 데이터셋을 사용하는 이유:
- 2025년 1월 발표된 최신 문서로, LLM이 사전 학습하지 않은 내용
- 한국 디지털 정부에 대한 구체적인 정책 권고와 통계 포함
- RAG의 가치를 명확히 보여줄 수 있음 (순수 LLM은 답할 수 없는 질문에 답변 가능)

사용법:
    [로컬 실행]
    1. .env 파일에 API 키 설정
    2. config에서 원하는 LLM 선택 (gemini 또는 openai)
    3. main() 실행

    [Streamlit Cloud 배포]
    1. Streamlit Cloud의 Secrets에 API 키 설정
    2. streamlit run app.py 실행
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional


# === API 키 관리 ===
def get_secret(key: str) -> Optional[str]:
    """
    API 키를 가져옵니다.
    우선순위: Streamlit secrets > 환경변수 > None
    """
    # 1. Streamlit secrets 시도
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, Exception):
        pass

    # 2. 환경변수 fallback
    return os.getenv(key)

# === 설정 ===
@dataclass
class Config:
    """RAG 설정"""
    # LLM 선택: "gemini" 또는 "openai"
    llm_provider: str = "gemini"

    # 모델명 (각 provider별)
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-5"

    # 임베딩 설정
    embedding_model: str = "text-embedding-3-small"  # OpenAI 임베딩

    # 청크 설정
    chunk_size: int = 500  # 문자 수 기준
    chunk_overlap: int = 100

    # 검색 설정
    top_k: int = 3  # 검색할 문서 수

    # 데이터 설정
    max_documents: int = 100  # 로드할 최대 문서 수 (None이면 전체)


# === 데이터 로더 ===
class HTMLStripper(HTMLParser):
    """HTML 태그 제거용 파서"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_html_tags(html: str) -> str:
    """HTML 태그를 제거하고 순수 텍스트만 반환"""
    s = HTMLStripper()
    s.feed(html)
    text = s.get_data()
    # 연속된 공백/줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@dataclass
class Document:
    """문서 데이터 클래스"""
    doc_id: str
    title: str
    url: str
    content: str  # 클린 텍스트
    questions: list[dict]  # [{"question": "...", "answer": "..."}]


def load_korquad_from_github(max_docs: Optional[int] = None) -> list[Document]:
    """GitHub에서 KorQuAD 데이터 직접 다운로드 (Streamlit Cloud용)"""
    import urllib.request
    import tempfile

    # MY_GITHUB에서 korquad_half.json 다운로드
    url = "https://raw.githubusercontent.com/seonuan82/RAG_Workshop/main/korquad_div9.json"

    documents = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON 다운로드
        json_path = os.path.join(tmpdir, "korquad_div9.json")
        print("   GitHub에서 다운로드 중...")
        urllib.request.urlretrieve(url, json_path)

        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data['data'][:max_docs] if max_docs else data['data']

        for idx, item in enumerate(items):
            clean_content = strip_html_tags(item['context'])

            questions = []
            for qa in item.get('qas', []):
                questions.append({
                    "question": qa['question'],
                    "answer": qa['answer']['text']
                })

            doc = Document(
                doc_id=f"doc_{idx}",
                title=item['title'],
                url=item.get('url', ''),
                content=clean_content,
                questions=questions
            )
            documents.append(doc)

    return documents


def load_korquad_from_json(filepath: str, max_docs: Optional[int] = None) -> list[Document]:
    """로컬 JSON 파일에서 KorQuAD 2.1 데이터 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    items = data['data'][:max_docs] if max_docs else data['data']

    for idx, item in enumerate(items):
        # HTML에서 텍스트 추출
        clean_content = strip_html_tags(item['context'])

        # QA 쌍 추출
        questions = []
        for qa in item.get('qas', []):
            questions.append({
                "question": qa['question'],
                "answer": qa['answer']['text']
            })

        doc = Document(
            doc_id=f"doc_{idx}",
            title=item['title'],
            url=item['url'],
            content=clean_content,
            questions=questions
        )
        documents.append(doc)

    return documents


def load_korquad_data(filepath: Optional[str] = None, max_docs: Optional[int] = None) -> list[Document]:
    """
    KorQuAD 데이터 로드 (자동 선택)
    - 로컬 파일이 있으면 JSON에서 로드
    - 없으면 GitHub에서 직접 다운로드
    """
    # 로컬 파일 확인
    if filepath and os.path.exists(filepath):
        print(f"   로컬 파일에서 로드: {filepath}")
        return load_korquad_from_json(filepath, max_docs)

    # GitHub에서 직접 다운로드
    return load_korquad_from_github(max_docs)


# === OECD 데이터 로더 ===
# 샘플 질문과 답변 (RAG 테스트용)
OECD_SAMPLE_QA = [
    {
        "question": "OECD 디지털 정부 리뷰에서 한국 디지털 정부의 4가지 주요 평가 영역은 무엇인가요?",
        "answer": "거버넌스, 데이터, AI, 인간 중심 서비스"
    },
    {
        "question": "한국의 디지털 정부 전환을 주도하는 핵심 정부 부처는 어디인가요?",
        "answer": "행정안전부 (MOIS)"
    },
    {
        "question": "OECD 디지털정부지수에서 한국의 데이터 관련 성과는 어떤가요?",
        "answer": "OECD 국가 중 정부 데이터 및 공개 데이터 성숙도에서 1위"
    },
    {
        "question": "한국의 디지털 정부 플랫폼인 Government24에서 제공하는 서비스 수는 몇 개인가요?",
        "answer": "1,500개 이상"
    },
    {
        "question": "한국의 전자정부법은 언제 제정되었나요?",
        "answer": "2001년"
    },
    {
        "question": "한국의 기본 AI법(Basic AI Act)은 언제 시행될 예정인가요?",
        "answer": "2026년"
    },
    {
        "question": "OECD가 권고하는 디지털 정부 투자 관리 개선 방안은 무엇인가요?",
        "answer": "다년도 예산 옵션, 혁신팀 자금 지원, 조건부 초과지출 허용, AI 등 디지털 기술 전용 펀드"
    },
    {
        "question": "한국 디지털 정부의 데이터 관련 주요 과제는 무엇인가요?",
        "answer": "데이터 발견 가능성, 데이터 접근 및 공유에 대한 법적 준수 지원, 구식 법률로 인한 데이터 공유 장애"
    },
    {
        "question": "한국 정부의 AI 사용 분야 예시를 들어주세요.",
        "answer": "노동 검사 도구, 특허 심사 지원, 홍수 예측 시스템"
    },
    {
        "question": "OECD가 한국에 권고하는 AI 관련 투명성 강화 방안은 무엇인가요?",
        "answer": "정부 AI 시스템의 공개 등록부(public inventory) 구축"
    },
]


def load_oecd_data(filepath: Optional[str] = None, max_docs: Optional[int] = None) -> list[Document]:
    """
    OECD 디지털 정부 리뷰 데이터 로드

    Args:
        filepath: oecd.txt 파일 경로 (None이면 기본 경로 사용)
        max_docs: 최대 문서 수 (챕터 기준)

    Returns:
        Document 리스트 (챕터별로 분할)
    """
    # 기본 경로 설정
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "oecd.txt")

    # GitHub에서 다운로드 시도 (로컬 파일이 없는 경우)
    if not os.path.exists(filepath):
        return load_oecd_from_github(max_docs)

    print(f"   로컬 파일에서 로드: {filepath}")

    # 파일 읽기
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 챕터별로 분할
    chapters = [
        ("Executive Summary", "Executive summary", "Assessment and recommendations"),
        ("Assessment and Recommendations", "Assessment and recommendations", "Korea's journey to becoming"),
        ("Korea's Digital Government Journey", "Korea's journey to becoming a global leader", "Strengthening governance"),
        ("Governance, Investment, and Skills", "Strengthening governance, investment, and skills for digital government", "Improving data governance"),
        ("Data Governance", "Improving data governance, sharing, and use", "Leveraging AI for government"),
        ("AI for Government", "Leveraging AI for government transformation", "Delivering human-centred"),
        ("Human-Centred Services", "Delivering human-centred and proactive public services", None),
    ]

    documents = []
    qa_idx = 0

    for idx, (title, start_marker, end_marker) in enumerate(chapters):
        # 챕터 내용 추출
        start_pos = content.find(start_marker)
        if start_pos == -1:
            continue

        if end_marker:
            end_pos = content.find(end_marker, start_pos + len(start_marker))
            if end_pos == -1:
                end_pos = len(content)
        else:
            end_pos = len(content)

        chapter_content = content[start_pos:end_pos].strip()

        # 연속된 공백/줄바꿈 정리
        chapter_content = re.sub(r'\n{3,}', '\n\n', chapter_content)
        chapter_content = re.sub(r'[ \t]+', ' ', chapter_content)

        # 챕터에 해당하는 Q&A 할당 (순환)
        questions = []
        if qa_idx < len(OECD_SAMPLE_QA):
            # 각 챕터에 1-2개의 Q&A 할당
            questions.append(OECD_SAMPLE_QA[qa_idx])
            qa_idx += 1
            if qa_idx < len(OECD_SAMPLE_QA):
                questions.append(OECD_SAMPLE_QA[qa_idx])
                qa_idx += 1

        doc = Document(
            doc_id=f"oecd_chapter_{idx}",
            title=title,
            url="https://www.oecd.org/en/publications/digital-government-review-of-korea_d5ac8a8f-en.html",
            content=chapter_content,
            questions=questions
        )
        documents.append(doc)

        if max_docs and len(documents) >= max_docs:
            break

    return documents


def load_oecd_from_github(max_docs: Optional[int] = None) -> list[Document]:
    """GitHub에서 OECD 데이터 직접 다운로드"""
    import urllib.request
    import tempfile

    url = "https://raw.githubusercontent.com/seonuan82/RAG_Workshop/main/oecd.txt"

    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = os.path.join(tmpdir, "oecd.txt")
        print("   GitHub에서 OECD 데이터 다운로드 중...")
        urllib.request.urlretrieve(url, txt_path)

        return load_oecd_data(txt_path, max_docs)


# === 텍스트 청킹 ===
@dataclass
class Chunk:
    """청크 데이터 클래스"""
    chunk_id: str
    doc_id: str
    title: str
    content: str
    embedding: Optional[list[float]] = None


def create_chunks(documents: list[Document], chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """문서를 청크로 분할"""
    chunks = []

    for doc in documents:
        text = doc.content
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # 문장 경계에서 자르기 (가능하면)
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > chunk_size // 2:
                    chunk_text = chunk_text[:last_period + 1]
                    end = start + last_period + 1

            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx}",
                doc_id=doc.doc_id,
                title=doc.title,
                content=chunk_text.strip()
            )
            chunks.append(chunk)

            start = end - overlap
            chunk_idx += 1

    return chunks


# === LLM 추상화 레이어 ===
class BaseLLM(ABC):
    """LLM 추상 기본 클래스"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """텍스트 생성"""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """텍스트 임베딩 반환"""
        pass


class GeminiLLM(BaseLLM):
    """Google Gemini API (new SDK)"""

    def __init__(self, model: str = "gemini-2.5-flash"):
        from google import genai

        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY를 설정하세요 (Streamlit secrets 또는 환경변수)")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = "text-embedding-004"

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=text
        )
        return response.embeddings[0].values


class OpenAILLM(BaseLLM):
    """OpenAI API (new SDK)"""

    def __init__(self, model: str = "gpt-5-mini", embedding_model: str = "text-embedding-3-small"):
        from openai import OpenAI

        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY를 설정하세요 (Streamlit secrets 또는 환경변수)")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )
        return response.output_text

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding


def create_llm(config: Config) -> BaseLLM:
    """
    LLM 인스턴스 생성 (자동 선택)
    우선순위: GOOGLE_API_KEY > OPENAI_API_KEY
    """
    google_key = get_secret("GOOGLE_API_KEY")
    openai_key = get_secret("OPENAI_API_KEY")

    # 1. Google API 키가 있으면 Gemini 사용
    if google_key:
        print(f"   Gemini API 사용 ({config.gemini_model})")
        return GeminiLLM(model=config.gemini_model)

    # 2. OpenAI API 키가 있으면 GPT 사용
    if openai_key:
        print(f"   OpenAI API 사용 ({config.openai_model})")
        return OpenAILLM(model=config.openai_model, embedding_model=config.embedding_model)

    # 3. 둘 다 없으면 에러
    raise ValueError(
        "API 키를 설정하세요.\n"
        "- GOOGLE_API_KEY (Gemini)\n"
        "- OPENAI_API_KEY (GPT)\n"
        "Streamlit secrets 또는 환경변수로 설정 가능합니다."
    )


# === 벡터 검색 (간단한 구현) ===
import numpy as np
import math
from collections import Counter

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """코사인 유사도 계산"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# === 키워드 검색 (BM25) ===
def tokenize(text: str) -> list[str]:
    """간단한 토크나이저 (한국어/영어 모두 지원)"""
    # 소문자 변환 및 특수문자 제거
    text = text.lower()
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # 공백 기준 분리
    tokens = text.split()
    # 불용어 제거 (간단한 버전)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 '은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로', '와', '과', '도'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               avg_doc_len: float, doc_count: int, doc_freqs: dict,
               k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 스코어 계산"""
    score = 0.0
    doc_len = len(doc_tokens)
    doc_token_counts = Counter(doc_tokens)

    for token in query_tokens:
        if token not in doc_token_counts:
            continue

        # TF (Term Frequency)
        tf = doc_token_counts[token]

        # IDF (Inverse Document Frequency)
        df = doc_freqs.get(token, 0)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

        # BM25 공식
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)

    return score


def keyword_search(query: str, chunks: list, top_k: int = 3) -> list[tuple]:
    """
    BM25 기반 키워드 검색

    Args:
        query: 검색 쿼리
        chunks: Chunk 리스트
        top_k: 반환할 결과 수

    Returns:
        [(Chunk, score), ...] 리스트
    """
    # 쿼리 토큰화
    query_tokens = tokenize(query)

    # 모든 청크 토큰화
    chunk_tokens_list = [tokenize(chunk.content) for chunk in chunks]

    # 문서 빈도 계산 (IDF용)
    doc_freqs = Counter()
    for tokens in chunk_tokens_list:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freqs[token] += 1

    # 평균 문서 길이
    avg_doc_len = sum(len(tokens) for tokens in chunk_tokens_list) / len(chunks) if chunks else 1

    # BM25 스코어 계산
    results = []
    for chunk, doc_tokens in zip(chunks, chunk_tokens_list):
        score = bm25_score(
            query_tokens, doc_tokens,
            avg_doc_len, len(chunks), doc_freqs
        )
        results.append((chunk, score))

    # 스코어 기준 내림차순 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


class SimpleVectorStore:
    """간단한 인메모리 벡터 저장소"""

    def __init__(self):
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]):
        """임베딩이 포함된 청크 추가"""
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[tuple[Chunk, float]]:
        """쿼리와 가장 유사한 청크 검색"""
        results = []

        for chunk in self.chunks:
            if chunk.embedding:
                similarity = cosine_similarity(query_embedding, chunk.embedding)
                results.append((chunk, similarity))

        # 유사도 기준 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# === RAG 파이프라인 ===
class RAGPipeline:
    """RAG 파이프라인"""

    def __init__(self, llm: BaseLLM, vector_store: SimpleVectorStore, config: Config):
        self.llm = llm
        self.vector_store = vector_store
        self.config = config

    def index_documents(self, chunks: list[Chunk], show_progress: bool = True):
        """문서 인덱싱 (임베딩 생성)"""
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  임베딩 생성 중... {i + 1}/{total}")

            chunk.embedding = self.llm.get_embedding(chunk.content)

        self.vector_store.add_chunks(chunks)
        print(f"  총 {total}개 청크 인덱싱 완료")

    def retrieve(self, query: str) -> list[Chunk]:
        """관련 문서 검색"""
        query_embedding = self.llm.get_embedding(query)
        results = self.vector_store.search(query_embedding, top_k=self.config.top_k)
        return [chunk for chunk, _ in results]

    def generate_answer(self, query: str, contexts: list[Chunk]) -> str:
        """컨텍스트 기반 답변 생성"""
        context_text = "\n\n---\n\n".join([
            f"[{c.title}]\n{c.content}" for c in contexts
        ])

        prompt = f"""다음 컨텍스트를 참고하여 질문에 답변하세요.

컨텍스트:
{context_text}

질문: {query}

답변:"""

        return self.llm.generate(prompt)

    def query(self, question: str) -> dict:
        """전체 RAG 파이프라인 실행"""
        # 1. 검색
        relevant_chunks = self.retrieve(question)

        # 2. 생성
        answer = self.generate_answer(question, relevant_chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": [{"title": c.title, "content": c.content[:200] + "..."} for c in relevant_chunks]
        }


# === 평가 ===
def evaluate_rag(rag: RAGPipeline, documents: list[Document], num_samples: int = 10) -> dict:
    """RAG 성능 평가 (Ground Truth 비교)"""
    correct = 0
    total = 0
    results = []

    # QA가 있는 문서에서 샘플 추출
    qa_docs = [d for d in documents if d.questions][:num_samples]

    for doc in qa_docs:
        for qa in doc.questions:
            question = qa['question']
            ground_truth = qa['answer']

            # RAG 답변 생성
            result = rag.query(question)
            predicted = result['answer']

            # 간단한 포함 여부 체크 (실제로는 더 정교한 메트릭 사용)
            is_correct = ground_truth.lower() in predicted.lower()
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted[:200] + "..." if len(predicted) > 200 else predicted,
                "correct": is_correct,
                "sources": [s['title'] for s in result['sources']]
            })

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }


def compare_chunk_sizes(documents: list[Document], llm: BaseLLM,
                        chunk_sizes: list[int], test_question: str) -> None:
    """청크 크기별 결과 비교 실험"""
    print(f"\n=== 청크 크기 비교 실험 ===")
    print(f"질문: {test_question}\n")

    for size in chunk_sizes:
        chunks = create_chunks(documents, chunk_size=size, overlap=size//5)
        vector_store = SimpleVectorStore()

        config = Config(chunk_size=size, top_k=3)
        rag = RAGPipeline(llm, vector_store, config)

        # 인덱싱 (조용히)
        for chunk in chunks:
            chunk.embedding = llm.get_embedding(chunk.content)
        vector_store.add_chunks(chunks)

        result = rag.query(test_question)

        print(f"[Chunk Size: {size}]")
        print(f"  총 청크 수: {len(chunks)}")
        print(f"  답변: {result['answer'][:150]}...")
        print(f"  참조: {[s['title'] for s in result['sources']]}")
        print()


# === 메인 실행 ===
def main():
    # 환경변수 로드 (dotenv 사용 시)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv가 없습니다. 환경변수를 직접 설정하세요.")

    # 설정
    config = Config(
        llm_provider="gemini",  # "gemini" 또는 "openai"로 변경
        max_documents=50,       # 실습용으로 50개만 로드
        top_k=3
    )

    print(f"=== RAG Workshop ===")
    print(f"OECD Digital Government Review of Korea (2025)")
    print(f"LLM Provider: {config.llm_provider}")
    print()

    # 1. 데이터 로드 (OECD 문서)
    print("1. 데이터 로드 중...")
    documents = load_oecd_data(max_docs=config.max_documents)
    print(f"   {len(documents)}개 챕터 로드 완료")

    # 2. 청킹
    print("\n2. 문서 청킹 중...")
    chunks = create_chunks(documents, config.chunk_size, config.chunk_overlap)
    print(f"   {len(chunks)}개 청크 생성 완료")

    # 3. LLM 및 벡터 저장소 초기화
    print("\n3. LLM 초기화 중...")
    llm = create_llm(config)
    vector_store = SimpleVectorStore()

    # 4. RAG 파이프라인 생성
    rag = RAGPipeline(llm, vector_store, config)

    # 5. 인덱싱
    print("\n4. 문서 인덱싱 중...")
    rag.index_documents(chunks)

    # 6. 인터랙티브 실습 메뉴
    print("\n" + "="*50)
    print("RAG 실습 준비 완료!")
    print("="*50)

    while True:
        print("\n[실습 메뉴]")
        print("1. 자유 질문하기")
        print("2. 샘플 질문 테스트 (데이터셋 기반)")
        print("3. 청크 크기 비교 실험")
        print("4. RAG 정확도 평가")
        print("5. 종료")

        choice = input("\n선택 (1-5): ").strip()

        if choice == "1":
            question = input("질문을 입력하세요: ").strip()
            if question:
                result = rag.query(question)
                print(f"\n[답변]\n{result['answer']}")
                print(f"\n[참조 문서]")
                for s in result['sources']:
                    print(f"  - {s['title']}: {s['content'][:100]}...")

        elif choice == "2":
            sample_questions = [doc.questions[0]['question'] for doc in documents[:5] if doc.questions]
            print("\n[샘플 질문 목록]")
            for i, q in enumerate(sample_questions):
                print(f"  {i+1}. {q}")

            idx = input("번호 선택: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(sample_questions):
                q = sample_questions[int(idx)-1]
                result = rag.query(q)
                print(f"\n질문: {q}")
                print(f"답변: {result['answer']}")
                print(f"참조: {[s['title'] for s in result['sources']]}")

        elif choice == "3":
            test_q = input("비교할 질문 입력 (Enter시 샘플 사용): ").strip()
            if not test_q:
                test_q = documents[0].questions[0]['question'] if documents[0].questions else "한국의 수도는?"

            compare_chunk_sizes(
                documents[:20],  # 실험용 20개만
                llm,
                chunk_sizes=[200, 500, 1000],
                test_question=test_q
            )

        elif choice == "4":
            print("\n평가 중... (시간이 걸릴 수 있습니다)")
            eval_result = evaluate_rag(rag, documents, num_samples=5)
            print(f"\n[평가 결과]")
            print(f"정확도: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})")
            print("\n[상세 결과]")
            for r in eval_result['results']:
                status = "✓" if r['correct'] else "✗"
                print(f"{status} Q: {r['question']}")
                print(f"   정답: {r['ground_truth']}")
                print(f"   예측: {r['predicted'][:100]}...")
                print()

        elif choice == "5":
            print("종료합니다.")
            break

        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
