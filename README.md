# An Introduction to RAG Through Chatbot Experiment Design with Streamlit

**Build your own AI Co-Scientist Workshop - February 2026**

---

## 준비물

| 준비물 | 링크 | 필수 |
|--------|------|:----:|
| **OpenAI API Key** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | ✅ |
| **Google API Key** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | 선택 |
| **GitHub Account** | [https://github.com/signup](https://github.com/signup) | ✅ |
| **Streamlit Account** | [https://authkit.streamlit.io/sign-up](https://authkit.streamlit.io/sign-up) | ✅ |

---

## 워크샵 개요

| 순서 | 내용 | 실습 파일 | Colab |
|------|------|----------|-------|
| **1** | RAG 기본 개념 및 구현 | [발표 슬라이드 링크](https://www.canva.com/design/DAG_kVIUpn0/WsItAe2XgXWxYOL70i9WEw/view?utm_content=DAG_kVIUpn0&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hf5d16a1ffb) | [Google Colab](https://colab.research.google.com/drive/1YYwC7SEqd0FaRr0qdUe4upY-D6OvTvq6?usp=sharing) |
| **2** | Streamlit 기본 챗봇 만들기 | `W2_Streamlit_chatbot.py` |  |
| **3** | RAG 챗봇 (BM25 & Semantic Search) | `W3_RAG_chatbot_practice.py` |  |
| **선택** | RAG 챗봇 심화 (Hybrid Search) | `W4_RAG_chatbot_advanced.py` |  |


---

## 파일 구조

```
RAG_Workshop/
├── W2_Streamlit_chatbot.py           # 2: 기본 Streamlit 챗봇
├── W2_Streamlit_chatbot_utilities.py # 2: LLM 유틸리티 모듈
├── W3_RAG_chatbot_practice.py        # 3: RAG 챗봇 실습 (빈칸)
├── W3_RAG_chatbot_answer.py          # 3: RAG 챗봇 정답
├── W4_RAG_chatbot_advanced.py        # 4: Advanced RAG 챗봇
├── Colab_data_OECD.txt               # 1 실습용 데이터
├── Practice_data_NewsResult.CSV      # 3 실습용 뉴스 데이터
├── Streamlit Guide.pdf               # Streamlit 배포 가이드
└── requirements.txt                  # 패키지 의존성 (필수)
```

---

## W2 Streamlit 챗봇
- 실행: https://w2-build-chatbot.streamlit.app/
- Streamlit을 활용한 기본 챗봇

---

## W3 RAG 챗봇
- 실행: https://w3-build-rag-chatbot.streamlit.app/
- 기본 RAG 기법이 적용된 챗봇

**실습 목표**:
1. `bm25_search()` - BM25 검색 함수 구현
2. `semantic_search()` - 시맨틱 검색 함수 구현
3. `get_relevant_news()` - 관련 뉴스 검색 함수 구현
4. `generate_rag_answer()` - RAG 파이프라인 적용

**데이터**: `Practice_data_NewsResult.CSV` (심리/뇌과학 키워드 뉴스)

---

## W4 Advanced RAG 챗봇
- 실행: https://w4-rag-chatbot-advanced.streamlit.app/
- W3 RAG 챗봇에 Hybrid Search 기법을 추가한 챗봇


---

## 설치 및 실행

**Streamlit Cloud App Deploy 시**:
- Streamlit Guide.pdf 참고
1. requirements.txt 반드시 GitHub 레포지토리에 넣어두기
2. Streamlit cloud App Secrets 탭에 API Key 넣기
```
GOOGLE_API_KEY = "your-api-key"
# 또는
OPENAI_API_KEY = "your-api-key"
```

---
*링크를 걸어둔 Streamlit app은 2월 5일까지만 접속 가능합니다.
*Last updated: 2026-02-03 by Seonu An*
