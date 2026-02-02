# TODO 1: BM25 검색 구현
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



# TODO 2: Semantic Search 구현
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
    # 3. 점수 순 정렬 및 상위 top_k개 반환
    #    results.sort(key=lambda x: x[1], reverse=True)
    #    return results[:top_k]
    # ──────────────────────────────────────────


# TODO 3: 관련 뉴스 검색 구현
    # ──────────────────────────────────────────
    # 1. 검색 함수 호출 (bm25_search 또는 semantic_search)
    #    results = bm25_search(query, news_data, top_k)
    #
    # 2. 결과 반환
    #    return results
    # ──────────────────────────────────────────

    
# TODO 4: generate_rag_answer 호출 및 결과 추출
    # ──────────────────────────────────────────
    # 1. generate_rag_answer() 에 들어가야 하는 정보 확인하기
    #   result = generate_rag_answer(
    #       last_user_input, 
    #       st.session_state.news_data, 
    #       st.session_state.llm
    #   )
    #
    # 2. generate_rag_answer() 에서 반환하는 결과값 확인하기
    #   response = result["answer"]
    #   sources = result["sources"]
