"""
간단한 Streamlit 챗봇
=====================
실행: streamlit run chatbot_app.py
"""

import streamlit as st
from 2_Streamlit_chatbot_utilities import get_basic_response, get_revised_response

# === 설정 ===
AVATAR_USER = "🎃"
AVATAR_DEFAULT = "🤖"
AVATAR_REVISED = "🦾"

# === 세션 상태 초기화 ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "version" not in st.session_state:
    st.session_state.version = "Default"


def reset_chat():
    """대화 초기화"""
    st.session_state.messages = []


# === UI 구성 ===
st.title("💬 Chatbot")

# 상단 버튼
_, col_version, col_button = st.columns([2, 1, 1])
with col_version:
    st.markdown(f"**{st.session_state.version}**")
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
        if user_input == "password":
            st.session_state.version = "Revised"
            response = "🔓 새로운 버전으로 전환합니다.\n\n" + get_revised_response(user_input)
        elif user_input == "return":
            st.session_state.version = "Default"
            response = "🔙 기본 버전으로 돌아갑니다.\n\n" + get_basic_response(user_input)
        elif st.session_state.version == "Revised":
            response = get_revised_response(user_input)
        else:
            response = get_basic_response(user_input)
    except Exception as e:
        response = f"⚠️ 오류가 발생했습니다: {str(e)}"

    # 어시스턴트 아바타 선택
    avatar = AVATAR_REVISED if st.session_state.version == "Revised" else AVATAR_DEFAULT

    # 어시스턴트 메시지 저장 및 표시
    with st.chat_message("assistant", avatar=avatar):
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "avatar": avatar
    })

    st.rerun()
