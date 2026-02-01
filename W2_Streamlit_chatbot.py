"""
간단한 Streamlit 챗봇
=====================
실행: streamlit run W2_Streamlit_chatbot.py
"""

import streamlit as st
from W2_Streamlit_chatbot_utilities import get_basic_response

# === 설정 ===
AVATAR_USER = "🎃"
AVATAR_BOT = "🤖"

# === 세션 상태 초기화 ===
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_chat():
    """대화 초기화"""
    st.session_state.messages = []


# === UI 구성 ===
st.title("💬 Chatbot")

# 상단 버튼
_, col_button = st.columns([3, 1])
with col_button:
    st.button("새 대화", on_click=reset_chat, use_container_width=True)

# 대화 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

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

    # 응답 생성
    try:
        response = get_basic_response(user_input)
    except Exception as e:
        response = f"⚠️ 오류가 발생했습니다: {str(e)}"

    # 어시스턴트 메시지 저장 및 표시
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "avatar": AVATAR_BOT
    })

    st.rerun()
