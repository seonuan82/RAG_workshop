"""
RAG Workshop - Streamlit UI
============================
실행: streamlit run app.py

OECD 한국 디지털 정부 리뷰 (2025) 문서를 활용한 RAG 실습
- 2025년 최신 문서로 LLM이 사전 학습하지 않은 내용
- RAG의 가치를 명확히 보여줄 수 있음
"""

import streamlit as st
import pandas as pd
from 3_RAG_playground_utilities import (
    Config, Document, Chunk,
    load_oecd_data, create_chunks, create_llm,
    SimpleVectorStore, RAGPipeline, cosine_similarity,
    OECD_SAMPLE_QA, keyword_search
)
import os

# 페이지 설정
st.set_page_config(
    page_title="RAG Workshop - OECD Korea Review",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Workshop")
st.markdown("""
**OECD 한국 디지털 정부 리뷰 (2025)** 기반 RAG 실습

이 워크샵에서는 2025년 1월 발표된 OECD 문서를 활용합니다.
LLM은 이 문서를 사전 학습하지 않았기 때문에, RAG 없이는 정확한 답변이 어렵습니다.
""")

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # 청크 설정
    chunk_size = st.slider("청크 크기", 700, 1500, 1000, 200)
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

                # 데이터 로드 (OECD 문서)
                st.info("📥 OECD 데이터 로드 중...")
                documents = load_oecd_data(max_docs=config.max_documents)
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
                st.success(f"✅ 초기화 완료! ({len(documents)}개 챕터, {len(chunks)}개 청크)")

            except Exception as e:
                st.error(f"❌ 오류: {e}")

with col2:
    if st.session_state.initialized:
        st.success(f"✅ RAG 준비 완료 | {len(st.session_state.documents)}개 챕터 | {len(st.session_state.chunks)}개 청크")


# 탭 구성
if st.session_state.initialized:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 질문하기", "📊 RAG vs API", "🔤 검색 방식", "🔬 Top-K 실험", "✂️ 청킹 실험"])

    # 탭 1: 자유 질문
    with tab1:
        st.subheader("💬 자유 질문하기")

        question = st.text_input(
            "질문을 입력하세요:",
            placeholder="예: 한국 디지털 정부의 주요 과제는 무엇인가요?"
        )

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
        st.markdown("""
        **RAG 사용 시와 사용하지 않을 때의 답변을 비교합니다.**

        💡 *OECD 한국 디지털 정부 리뷰는 2025년 1월 발표된 문서입니다.*
        *LLM은 이 문서를 학습하지 않았기 때문에, RAG 없이는 정확한 답변이 어렵습니다.*
        """)

        # OECD 샘플 질문 사용
        sample_questions = OECD_SAMPLE_QA

        if sample_questions:
            selected = st.selectbox(
                "샘플 질문 선택:",
                range(len(sample_questions)),
                format_func=lambda i: f"{sample_questions[i]['question'][:60]}..."
            )

            st.markdown(f"**정답 (Ground Truth):** `{sample_questions[selected]['answer']}`")

            if st.button("🔄 비교 실행", key="compare_rag"):
                question = sample_questions[selected]["question"]
                ground_truth = sample_questions[selected]["answer"]

                col1, col2 = st.columns(2)

                # 1. 순수 API (RAG 없이)
                with col1:
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

                # 2. RAG 사용
                with col2:
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

    # 탭 3: 검색 방식 비교 (Semantic vs Keyword)
    with tab3:
        st.subheader("🔤 Semantic Search vs Keyword Search")
        st.markdown("""
        **두 가지 검색 방식의 결과를 비교합니다.**

        | 방식 | 원리 | 특징 |
        |------|------|------|
        | **Keyword (BM25)** | 단어 빈도 + 역문서 빈도 | 빠름, 정확한 용어 매칭 |
        | **Semantic** | 임베딩 벡터 유사도 | 의미 이해, 동의어 처리 |

        💡 *예: "AI법"을 검색하면 Keyword는 정확히 "AI법"이 있는 문서만, Semantic은 "인공지능 법률"도 찾습니다.*
        """)

        st.divider()

        # 검색 질문 입력
        search_query = st.text_input(
            "검색할 질문:",
            value="한국의 인공지능 관련 법률은 언제 시행되나요?",
            key="search_compare_query"
        )

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            search_top_k = st.slider("검색 결과 수 (Top-K)", 1, 10, 5, key="search_topk")

        if st.button("🔍 검색 비교 실행", key="run_search_compare", type="primary"):
            if search_query:
                col_semantic, col_keyword = st.columns(2)

                # Semantic Search
                with col_semantic:
                    st.markdown("### 🧠 Semantic Search")
                    st.caption("임베딩 기반 의미 유사도 검색")

                    with st.spinner("Semantic 검색 중..."):
                        query_embedding = st.session_state.llm.get_embedding(search_query)
                        semantic_results = st.session_state.rag.vector_store.search(
                            query_embedding, top_k=search_top_k
                        )

                    for i, (chunk, score) in enumerate(semantic_results):
                        with st.expander(f"{i+1}. [{chunk.title}] (유사도: {score:.4f})"):
                            st.write(chunk.content[:300] + "...")

                # Keyword Search
                with col_keyword:
                    st.markdown("### 📝 Keyword Search (BM25)")
                    st.caption("단어 빈도 기반 검색")

                    with st.spinner("Keyword 검색 중..."):
                        keyword_results = keyword_search(
                            search_query,
                            st.session_state.chunks,
                            top_k=search_top_k
                        )

                    for i, (chunk, score) in enumerate(keyword_results):
                        with st.expander(f"{i+1}. [{chunk.title}] (BM25: {score:.4f})"):
                            st.write(chunk.content[:300] + "...")

                # 결과 비교 분석
                st.divider()
                st.markdown("### 📊 결과 비교 분석")

                semantic_titles = [c.title for c, _ in semantic_results]
                keyword_titles = [c.title for c, _ in keyword_results]

                overlap = set(semantic_titles) & set(keyword_titles)
                only_semantic = set(semantic_titles) - set(keyword_titles)
                only_keyword = set(keyword_titles) - set(semantic_titles)

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("공통 결과", f"{len(overlap)}개")
                col_stat2.metric("Semantic만", f"{len(only_semantic)}개")
                col_stat3.metric("Keyword만", f"{len(only_keyword)}개")

                if only_semantic:
                    st.info(f"🧠 Semantic만 찾은 챕터: {', '.join(only_semantic)}")
                if only_keyword:
                    st.info(f"📝 Keyword만 찾은 챕터: {', '.join(only_keyword)}")

    # 탭 4: Top-K 비교 실험
    with tab4:
        st.subheader("🔬 Top-K 비교 실험")

        test_question = st.text_input("비교할 질문:", value=OECD_SAMPLE_QA[0]["question"] if OECD_SAMPLE_QA else "")

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

    # 탭 5: 청킹 실험 (토이 프로젝트)
    with tab5:
        st.subheader("✂️ 청킹 실험 (Toy Project)")
        st.markdown("""
        **청크 크기와 오버랩이 RAG 성능에 미치는 영향을 실험합니다.**
        - 소규모 데이터 (3개 챕터)로 빠르게 실험
        - 3가지 질문으로 결과 비교
        """)

        st.divider()

        # 실험용 데이터 준비 (3개 챕터만)
        toy_docs = st.session_state.documents[:3]

        # OECD 샘플 질문 3개 사용
        toy_questions = OECD_SAMPLE_QA[:3]

        if len(toy_questions) < 1:
            st.warning("질문이 있는 문서가 없습니다.")
        else:
            # 실험 설정
            col_settings1, col_settings2 = st.columns(2)

            with col_settings1:
                st.markdown("### 📐 청킹 설정")
                exp_chunk_sizes = st.multiselect(
                    "청크 크기 선택:",
                    [500, 700, 1000, 1200, 1500],
                    default=[500, 1000],
                    key="exp_chunk_size"
                )
                exp_overlap_ratio = st.slider(
                    "오버랩 비율 (%):",
                    0, 50, 20, 5,
                    key="exp_overlap",
                    help="청크 크기의 몇 %를 오버랩할지"
                )

            with col_settings2:
                st.markdown("### ❓ 테스트 질문")
                for i, q in enumerate(toy_questions):
                    st.markdown(f"**Q{i+1}.** {q['question'][:50]}...")
                    st.caption(f"정답: {q['answer']}")

            st.divider()

            # 실험 실행
            if st.button("🧪 청킹 실험 실행", key="run_chunk_exp", type="primary"):
                if not exp_chunk_sizes:
                    st.warning("청크 크기를 1개 이상 선택하세요.")
                else:
                    results_table = []

                    progress = st.progress(0)
                    total_steps = len(exp_chunk_sizes) * len(toy_questions)
                    current_step = 0

                    for chunk_size in exp_chunk_sizes:
                        overlap = int(chunk_size * exp_overlap_ratio / 100)

                        # 청킹
                        exp_chunks = create_chunks(toy_docs, chunk_size=chunk_size, overlap=overlap)

                        # 임베딩 생성 (소규모라 빠름)
                        with st.spinner(f"청크 크기 {chunk_size} 임베딩 중..."):
                            for chunk in exp_chunks:
                                chunk.embedding = st.session_state.llm.get_embedding(chunk.content)

                        # 벡터 저장소 및 RAG
                        exp_vector_store = SimpleVectorStore()
                        exp_vector_store.add_chunks(exp_chunks)

                        exp_config = Config(chunk_size=chunk_size, chunk_overlap=overlap, top_k=3)
                        exp_rag = RAGPipeline(st.session_state.llm, exp_vector_store, exp_config)

                        # 각 질문에 대해 테스트
                        for q_idx, q in enumerate(toy_questions):
                            result = exp_rag.query(q["question"])
                            is_correct = q["answer"].lower() in result["answer"].lower()

                            results_table.append({
                                "청크크기": chunk_size,
                                "오버랩": overlap,
                                "청크수": len(exp_chunks),
                                "질문": f"Q{q_idx+1}",
                                "정답포함": "✅" if is_correct else "❌",
                                "참조문서": ", ".join([s["title"][:10] for s in result["sources"]])
                            })

                            current_step += 1
                            progress.progress(current_step / total_steps)

                    # 결과 표시
                    st.markdown("### 📊 실험 결과")

                    df = pd.DataFrame(results_table)
                    st.dataframe(df, use_container_width=True)

                    # 요약
                    st.markdown("### 📈 요약")
                    for chunk_size in exp_chunk_sizes:
                        subset = [r for r in results_table if r["청크크기"] == chunk_size]
                        correct_count = sum(1 for r in subset if r["정답포함"] == "✅")
                        total_count = len(subset)
                        accuracy = correct_count / total_count * 100 if total_count > 0 else 0

                        chunk_count = subset[0]["청크수"] if subset else 0

                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"청크 {chunk_size}", f"{chunk_count}개 청크")
                        col2.metric("정확도", f"{accuracy:.0f}%")
                        col3.metric("정답", f"{correct_count}/{total_count}")

                    # 상세 결과
                    st.markdown("### 🔍 상세 결과")
                    for chunk_size in exp_chunk_sizes:
                        with st.expander(f"청크 크기: {chunk_size}"):
                            subset = [r for r in results_table if r["청크크기"] == chunk_size]
                            for r in subset:
                                status = r["정답포함"]
                                st.markdown(f"{status} **{r['질문']}** - 참조: {r['참조문서']}")

else:
    st.info("👆 사이드바에서 설정 후 '🚀 RAG 초기화' 버튼을 클릭하세요.")


# 푸터
st.divider()
st.caption("RAG Workshop | OECD Digital Government Review of Korea (2025) | Gemini / OpenAI")
