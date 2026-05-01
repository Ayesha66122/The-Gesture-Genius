import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
from chatbot import chatbot_response

st.set_page_config(
    page_title="The Gesture Genius",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 The Gesture Genius")
st.subheader("Pakistani Sign Language (PSL) Detection + Chatbot")
st.markdown("### Hackathon Project - Team: The Gesture Genius")
st.write("---")

# Model load karo
@st.cache_resource
def load_model():
    try:
        with open('gesture_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['signs']
    except:
        return None, None

model, SIGNS = load_model()

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if model is None:
    st.error("Model load nahi hua! gesture_model.pkl check karo.")
else:
    st.success("✅ Model ready hai!")

    st.write("### 📷 Sign Karein")
    st.write("Camera mein sign karein — AI pehchaan kar jawab dega!")

    img_file = st.camera_input("Sign karein yahan")

    if img_file is not None:
        img = Image.open(img_file)
        img_array = np.array(img)

        IMG_SIZE = 64
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        features = normalized.flatten().reshape(1, -1)

        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction] * 100
        sign_name = SIGNS[prediction]

        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Sign", sign_name.upper())
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        if confidence > 30:
            # Chatbot se jawab lo
            bot_reply = chatbot_response(sign_name)

            # History mein add karo
            st.session_state.chat_history.append({
                "sign": sign_name.upper(),
                "reply": bot_reply
            })

            st.success(f"🤖 Chatbot: {bot_reply}")
        else:
            st.warning("Confidence kam hai - dobara try karein")

    # Chat history dikhao
    if st.session_state.chat_history:
        st.write("---")
        st.write("### 💬 Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.write(f"🤟 **Sign:** {chat['sign']}")
            st.write(f"🤖 **Bot:** {chat['reply']}")
            st.write("---")

st.write("### 👥 Our Team")
st.write("**The Gesture Genius**")
st.write("Second Semester Students")
st.caption("Hackathon Submission - Built with ❤️ in Pakistan")
