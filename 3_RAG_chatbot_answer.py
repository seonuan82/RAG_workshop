"""
RAG 챗봇 (파일 업로드 기능 포함)
================================
실행: streamlit run rag_chatbot.py

업로드한 문서를 기반으로 질문에 답변하는 RAG 챗봇
"""

import streamlit as st
from 3_RAG_utilities import (
    Config, Chunk, create_llm, SimpleVectorStore, RAGPipeline
)

# === 페이지 설정 ===
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="🤖",
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


# === 텍스트 청킹 함수 ===
def create_chunks_from_text(text: str, filename: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """텍스트를 청크로 분할"""
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # 문장 경계에서 자르기
        if end < len(text):
            last_period = chunk_text.rfind('.')
            if last_period > chunk_size // 2:
                chunk_text = chunk_text[:last_period + 1]
                end = start + last_period + 1

        chunk = Chunk(
            chunk_id=f"{filename}_chunk_{chunk_idx}",
            doc_id=filename,
            title=filename,
            content=chunk_text.strip()
        )
        chunks.append(chunk)

        start = end - overlap
        chunk_idx += 1

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

    # 초기화 버튼
    if st.button("🚀 RAG 초기화", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("파일을 먼저 업로드하세요.")
        else:
            with st.spinner("초기화 중..."):
                try:
                    # Config 설정
                    config = Config(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k
                    )

                    # 파일에서 텍스트 추출 및 청킹
                    all_chunks = []
                    for file in uploaded_files:
                        content = file.read().decode('utf-8')
                        file_chunks = create_chunks_from_text(
                            content, file.name, chunk_size, chunk_overlap
                        )
                        all_chunks.extend(file_chunks)

                    st.session_state.chunks = all_chunks

                    # LLM 초기화
                    llm = create_llm(config)
                    st.session_state.llm = llm

                    # 벡터 저장소 및 RAG 파이프라인
                    vector_store = SimpleVectorStore()

                    # 임베딩 생성
                    progress = st.progress(0)
                    for i, chunk in enumerate(all_chunks):
                        chunk.embedding = llm.get_embedding(chunk.content)
                        progress.progress((i + 1) / len(all_chunks))

                    vector_store.add_chunks(all_chunks)

                    # RAG 파이프라인 생성
                    rag = RAGPipeline(llm, vector_store, config)
                    st.session_state.rag = rag

                    st.success(f"✅ {len(uploaded_files)}개 파일, {len(all_chunks)}개 청크 처리 완료!")

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
st.title("🤖 RAG 챗봇")

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

    # 사용자 입력
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
                    result = st.session_state.rag.query(user_input)
                    response = result["answer"]
                    sources = result["sources"]

                    st.markdown(response)

                    # 참조 문서 표시
                    with st.expander("📚 참조 문서"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**{i+1}. {source['title']}**")
                            st.caption(source["content"])

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
