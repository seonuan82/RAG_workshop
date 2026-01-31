"""
RAG 챗봇 실습 파일
==================
실행: streamlit run rag_chatbot_practice.py

📝 실습 목표:
1. 텍스트 청킹 함수 구현
2. RAG 초기화 로직 구현
3. 챗봇 응답 생성 로직 구현

💡 힌트: rag_workshop.py의 클래스들을 활용하세요
- Config: 설정
- Chunk: 청크 데이터
- create_llm(): LLM 생성
- SimpleVectorStore: 벡터 저장소
- RAGPipeline: RAG 파이프라인
"""

import streamlit as st
from 3_RAG_utilities import (
    Config, Chunk, create_llm, SimpleVectorStore, RAGPipeline
)

# === 페이지 설정 ===
st.set_page_config(
    page_title="RAG 챗봇 실습",
    page_icon="📝",
    layout="wide"
)

# === 설정 ===
AVATAR_USER = "👤"
AVATAR_BOT = "🤖"


# === 세션 상태 초기화 ===
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "llm" not in st.session_state:
        st.session_state.llm = None


init_session()


def reset_chat():
    """대화 초기화"""
    st.session_state.messages = []


def reset_all():
    """전체 초기화"""
    st.session_state.messages = []
    st.session_state.rag = None
    st.session_state.chunks = []
    st.session_state.llm = None


# === 실습 1: 텍스트 청킹 함수 구현 ===
def create_chunks_from_text(text: str, filename: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """
    텍스트를 청크로 분할하는 함수

    Args:
        text: 분할할 텍스트
        filename: 파일명 (청크 ID에 사용)
        chunk_size: 청크 크기 (문자 수)
        overlap: 청크 간 겹침 (문자 수)

    Returns:
        Chunk 객체 리스트

    💡 힌트:
    - Chunk 클래스 필드: chunk_id, doc_id, title, content
    - 슬라이딩 윈도우 방식으로 텍스트를 분할
    - start 위치에서 chunk_size만큼 추출
    - 다음 start = 현재 end - overlap
    """
    chunks = []

    # TODO: 청킹 로직 구현
    # ──────────────────────────────────────────
    # 1. start = 0, chunk_idx = 0으로 시작
    # 2. while start < len(text):
    #    - end = start + chunk_size
    #    - chunk_text = text[start:end]
    #    - Chunk 객체 생성 (chunk_id, doc_id, title, content 설정)
    #    - chunks 리스트에 추가
    #    - start = end - overlap
    #    - chunk_idx += 1
    # 3. return chunks
    # ──────────────────────────────────────────

    return chunks


# === 사이드바 ===
with st.sidebar:
    st.header("📁 문서 업로드")

    # 파일 업로드
    uploaded_files = st.file_uploader(
        "텍스트 파일 업로드",
        type=["txt"],
        accept_multiple_files=True,
        help="txt 파일을 업로드하세요"
    )

    # 청킹 설정
    st.subheader("⚙️ 설정")
    chunk_size = st.slider("청크 크기", 200, 1000, 500, 100)
    chunk_overlap = st.slider("청크 오버랩", 0, 200, 100, 50)
    top_k = st.slider("검색 문서 수 (Top-K)", 1, 10, 3)

    st.divider()

    # === 실습 2: RAG 초기화 버튼 ===
    if st.button("🚀 RAG 초기화", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("파일을 먼저 업로드하세요.")
        else:
            with st.spinner("초기화 중..."):
                try:
                    # TODO: RAG 초기화 로직 구현
                    # ──────────────────────────────────────────
                    # 1. Config 생성 (chunk_size, chunk_overlap, top_k 설정)
                    #    config = Config(...)
                    #
                    # 2. 파일에서 텍스트 추출 및 청킹
                    #    all_chunks = []
                    #    for file in uploaded_files:
                    #        content = file.read().decode('utf-8')
                    #        file_chunks = create_chunks_from_text(...)
                    #        all_chunks.extend(file_chunks)
                    #    st.session_state.chunks = all_chunks
                    #
                    # 3. LLM 초기화
                    #    llm = create_llm(config)
                    #    st.session_state.llm = llm
                    #
                    # 4. 벡터 저장소 생성 및 임베딩 생성
                    #    vector_store = SimpleVectorStore()
                    #    for chunk in all_chunks:
                    #        chunk.embedding = llm.get_embedding(chunk.content)
                    #    vector_store.add_chunks(all_chunks)
                    #
                    # 5. RAG 파이프라인 생성
                    #    rag = RAGPipeline(llm, vector_store, config)
                    #    st.session_state.rag = rag
                    #
                    # 6. 성공 메시지
                    #    st.success("✅ 초기화 완료!")
                    # ──────────────────────────────────────────

                    st.warning("⚠️ RAG 초기화 로직을 구현하세요!")

                except Exception as e:
                    st.error(f"❌ 오류: {e}")

    st.divider()

    # 버튼 영역
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔄 새 대화", on_click=reset_chat, use_container_width=True)
    with col2:
        st.button("🗑️ 전체 초기화", on_click=reset_all, use_container_width=True)

    # 상태 표시
    if st.session_state.rag:
        st.success(f"📚 {len(st.session_state.chunks)}개 청크 준비됨")


# === 메인 영역 ===
st.title("📝 RAG 챗봇 실습")

st.markdown("""
### 실습 안내
1. **실습 1**: `create_chunks_from_text()` 함수 구현
2. **실습 2**: RAG 초기화 로직 구현
3. **실습 3**: 챗봇 응답 생성 로직 구현

💡 완성된 코드는 `rag_chatbot.py`를 참고하세요.
""")

st.divider()

if not st.session_state.rag:
    st.info("👈 사이드바에서 문서를 업로드하고 'RAG 초기화'를 클릭하세요.")
else:
    # 대화 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])

            # 참조 문서 표시
            if msg.get("sources"):
                with st.expander("📚 참조 문서"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(f"**{i+1}. {source['title']}**")
                        st.caption(source["content"])

    # === 실습 3: 사용자 입력 및 응답 생성 ===
    if user_input := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "avatar": AVATAR_USER
        })
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(user_input)

        # RAG 응답 생성
        with st.chat_message("assistant", avatar=AVATAR_BOT):
            with st.spinner("답변 생성 중..."):
                try:
                    # TODO: RAG 응답 생성 로직 구현
                    # ──────────────────────────────────────────
                    # 1. RAG 쿼리 실행
                    #    result = st.session_state.rag.query(user_input)
                    #
                    # 2. 결과에서 답변과 참조 문서 추출
                    #    response = result["answer"]
                    #    sources = result["sources"]
                    #
                    # 3. 화면에 표시
                    #    st.markdown(response)
                    #    with st.expander("📚 참조 문서"):
                    #        for i, source in enumerate(sources):
                    #            st.markdown(f"**{i+1}. {source['title']}**")
                    #            st.caption(source["content"])
                    # ──────────────────────────────────────────

                    response = "⚠️ 응답 생성 로직을 구현하세요!"
                    sources = []
                    st.warning(response)

                except Exception as e:
                    response = f"⚠️ 오류가 발생했습니다: {str(e)}"
                    sources = []
                    st.error(response)

        # 어시스턴트 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": AVATAR_BOT,
            "sources": sources
        })

        st.rerun()
