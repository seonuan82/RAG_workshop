"""
RAG Workshop - Streamlit UI
============================
실행: streamlit run app.py
"""

import streamlit as st
from rag_workshop import (
    Config, Document, Chunk,
    load_korquad_data, create_chunks, create_llm,
    SimpleVectorStore, RAGPipeline, cosine_similarity
)
import os

# 페이지 설정
st.set_page_config(
    page_title="RAG Workshop",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Workshop")
st.markdown("KorQuAD 2.1 기반 RAG 실습")

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # 청크 설정
    chunk_size = st.slider("청크 크기", 200, 1000, 500, 100)
    chunk_overlap = st.slider("청크 오버랩", 0, 200, 100, 50)
    top_k = st.slider("검색할 문서 수 (Top-K)", 1, 10, 3)
    max_documents = st.slider("로드할 문서 수", 10, 100, 50, 10)

    st.divider()

    # API 상태 표시
    st.header("🔑 API 상태")
    google_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if google_key:
        st.success("✅ Gemini API 사용 가능")
    if openai_key:
        st.success("✅ OpenAI API 사용 가능")
    if not google_key and not openai_key:
        st.error("❌ API 키를 설정하세요")


# 세션 상태 초기화
if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.llm = None  # LLM 객체 저장 (순수 API 비교용)
    st.session_state.documents = None
    st.session_state.chunks = None
    st.session_state.initialized = False


# 초기화 버튼
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🚀 RAG 초기화", type="primary"):
        with st.spinner("초기화 중..."):
            try:
                # Config 설정
                config = Config(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    max_documents=max_documents
                )

                # 데이터 로드
                st.info("📥 데이터 로드 중...")
                documents = load_korquad_data(max_docs=config.max_documents)
                st.session_state.documents = documents

                # 청킹
                st.info("✂️ 청킹 중...")
                chunks = create_chunks(documents, config.chunk_size, config.chunk_overlap)
                st.session_state.chunks = chunks

                # LLM 초기화
                st.info("🤖 LLM 초기화 중...")
                llm = create_llm(config)

                # 벡터 저장소
                vector_store = SimpleVectorStore()

                # RAG 파이프라인
                rag = RAGPipeline(llm, vector_store, config)

                # 인덱싱
                progress_bar = st.progress(0)
                st.info("🔢 임베딩 생성 중...")

                total = len(chunks)
                for i, chunk in enumerate(chunks):
                    chunk.embedding = llm.get_embedding(chunk.content)
                    progress_bar.progress((i + 1) / total)

                vector_store.add_chunks(chunks)

                st.session_state.rag = rag
                st.session_state.llm = llm  # LLM 저장
                st.session_state.initialized = True
                st.success(f"✅ 초기화 완료! ({len(documents)}개 문서, {len(chunks)}개 청크)")

            except Exception as e:
                st.error(f"❌ 오류: {e}")

with col2:
    if st.session_state.initialized:
        st.success(f"✅ RAG 준비 완료 | {len(st.session_state.documents)}개 문서 | {len(st.session_state.chunks)}개 청크")


# 탭 구성
if st.session_state.initialized:
    tab1, tab2, tab3 = st.tabs(["💬 질문하기", "📊 샘플 테스트", "🔬 비교 실험"])

    # 탭 1: 자유 질문
    with tab1:
        st.subheader("💬 자유 질문하기")

        question = st.text_input("질문을 입력하세요:", placeholder="예: 김연아의 출생지는?")

        if st.button("답변 생성", key="free_query"):
            if question:
                with st.spinner("답변 생성 중..."):
                    result = st.session_state.rag.query(question)

                    st.markdown("### 📝 답변")
                    st.write(result["answer"])

                    st.markdown("### 📚 참조 문서")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(f"{i+1}. {source['title']}"):
                            st.write(source["content"])
            else:
                st.warning("질문을 입력하세요.")

    # 탭 2: 샘플 테스트 (RAG vs 순수 API 비교)
    with tab2:
        st.subheader("📊 RAG vs 순수 API 비교")
        st.markdown("**RAG 사용 시와 사용하지 않을 때의 답변을 비교합니다.**")

        # 샘플 질문 선택
        sample_questions = []
        for doc in st.session_state.documents[:10]:
            if doc.questions:
                sample_questions.append({
                    "question": doc.questions[0]["question"],
                    "answer": doc.questions[0]["answer"],
                    "title": doc.title
                })

        if sample_questions:
            selected = st.selectbox(
                "샘플 질문 선택:",
                range(len(sample_questions)),
                format_func=lambda i: f"{sample_questions[i]['title']}: {sample_questions[i]['question'][:50]}..."
            )

            st.markdown(f"**정답 (Ground Truth):** `{sample_questions[selected]['answer']}`")

            if st.button("🔄 비교 실행", key="compare_rag"):
                question = sample_questions[selected]["question"]
                ground_truth = sample_questions[selected]["answer"]

                col1, col2, col3 = st.columns(3)

                # 1. 정답
                with col1:
                    st.markdown("### 🎯 정답")
                    st.info(ground_truth)

                # 2. 순수 API (RAG 없이)
                with col2:
                    st.markdown("### 💬 순수 API")
                    with st.spinner("순수 API 호출 중..."):
                        pure_prompt = f"다음 질문에 답변하세요.\n\n질문: {question}\n\n답변:"
                        pure_answer = st.session_state.llm.generate(pure_prompt)
                        st.warning(pure_answer)

                        pure_correct = ground_truth.lower() in pure_answer.lower()
                        if pure_correct:
                            st.success("✅ 정답 포함")
                        else:
                            st.error("❌ 정답 미포함")

                # 3. RAG 사용
                with col3:
                    st.markdown("### 🔍 RAG 사용")
                    with st.spinner("RAG 답변 생성 중..."):
                        rag_result = st.session_state.rag.query(question)
                        st.success(rag_result["answer"])

                        rag_correct = ground_truth.lower() in rag_result["answer"].lower()
                        if rag_correct:
                            st.success("✅ 정답 포함")
                        else:
                            st.error("❌ 정답 미포함")

                # 참조 문서 표시
                st.markdown("---")
                st.markdown("### 📚 RAG가 참조한 문서")
                for i, source in enumerate(rag_result["sources"]):
                    with st.expander(f"{i+1}. {source['title']}"):
                        st.write(source["content"])

    # 탭 3: 비교 실험
    with tab3:
        st.subheader("🔬 Top-K 비교 실험")

        test_question = st.text_input("비교할 질문:", value=sample_questions[0]["question"] if sample_questions else "")

        k_values = st.multiselect("비교할 Top-K 값:", [1, 2, 3, 5, 7, 10], default=[1, 3, 5])

        if st.button("비교 실행", key="compare"):
            if test_question and k_values:
                results = {}

                progress = st.progress(0)
                for i, k in enumerate(k_values):
                    with st.spinner(f"Top-{k} 테스트 중..."):
                        # 임시로 top_k 변경
                        original_k = st.session_state.rag.config.top_k
                        st.session_state.rag.config.top_k = k

                        result = st.session_state.rag.query(test_question)
                        results[k] = result

                        st.session_state.rag.config.top_k = original_k
                        progress.progress((i + 1) / len(k_values))

                # 결과 표시
                st.markdown("### 📊 결과 비교")

                for k, result in results.items():
                    with st.expander(f"Top-{k} 결과"):
                        st.write(f"**답변:** {result['answer']}")
                        st.write(f"**참조 문서:** {[s['title'] for s in result['sources']]}")

else:
    st.info("👆 사이드바에서 설정 후 '🚀 RAG 초기화' 버튼을 클릭하세요.")


# 푸터
st.divider()
st.caption("RAG Workshop | KorQuAD 2.1 | Gemini / OpenAI")
