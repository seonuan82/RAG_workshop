"""
RAG Workshop - KorQuAD 2.1 기반 실습
=====================================

이 모듈은 KorQuAD 2.1 데이터셋을 활용한 RAG(Retrieval-Augmented Generation) 실습을 위한 코드입니다.

사용법:
    1. .env 파일에 API 키 설정
    2. config에서 원하는 LLM 선택 (gemini 또는 openai)
    3. main() 실행
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional

# === 설정 ===
@dataclass
class Config:
    """RAG 설정"""
    # LLM 선택: "gemini" 또는 "openai"
    llm_provider: str = "gemini"

    # 모델명 (각 provider별)
    gemini_model: str = "gemini-1.5-flash"
    openai_model: str = "gpt-4o-mini"

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


def load_korquad_data(filepath: str, max_docs: Optional[int] = None) -> list[Document]:
    """KorQuAD 2.1 JSON 파일에서 문서 로드"""
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
    """Google Gemini API"""

    def __init__(self, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY 환경변수를 설정하세요")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.embed_model = "models/text-embedding-004"

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    def get_embedding(self, text: str) -> list[float]:
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.embed_model,
            content=text
        )
        return result['embedding']


class OpenAILLM(BaseLLM):
    """OpenAI API (GPT)"""

    def __init__(self, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수를 설정하세요")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding


def create_llm(config: Config) -> BaseLLM:
    """설정에 따라 LLM 인스턴스 생성"""
    if config.llm_provider == "gemini":
        return GeminiLLM(model=config.gemini_model)
    elif config.llm_provider == "openai":
        return OpenAILLM(model=config.openai_model, embedding_model=config.embedding_model)
    else:
        raise ValueError(f"지원하지 않는 LLM provider: {config.llm_provider}")


# === 벡터 검색 (간단한 구현) ===
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """코사인 유사도 계산"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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
    print(f"LLM Provider: {config.llm_provider}")
    print()

    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    data_path = os.path.join(os.path.dirname(__file__), "korquad2.1_train_00.json")
    documents = load_korquad_data(data_path, max_docs=config.max_documents)
    print(f"   {len(documents)}개 문서 로드 완료")

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

    # 6. 테스트 쿼리
    print("\n5. 테스트 쿼리 실행...")

    # 데이터에서 샘플 질문 사용
    sample_questions = [doc.questions[0]['question'] for doc in documents[:3] if doc.questions]

    for q in sample_questions:
        print(f"\n질문: {q}")
        result = rag.query(q)
        print(f"답변: {result['answer']}")
        print(f"참조 문서: {[s['title'] for s in result['sources']]}")


if __name__ == "__main__":
    main()
