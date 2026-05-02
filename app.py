import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
from chatbot import chatbot_response
from gtts import gTTS
import base64
import tempfile
import os

st.set_page_config(
    page_title="The Gesture Genius",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 The Gesture Genius")
st.subheader("Pakistani Sign Language (PSL) Detection")
st.markdown("### Hackathon Project - Team: The Gesture Genius")
st.write("---")

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            with open(fp.name, 'rb') as f:
                audio_data = f.read()
            os.unlink(fp.name)
        audio_b64 = base64.b64encode(audio_data).decode()
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                    unsafe_allow_html=True)
    except:
        pass

@st.cache_resource
def load_model():
    try:
        with open('gesture_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['signs']
    except:
        return None, None

model, SIGNS = load_model()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if model is None:
    st.error("Model load nahi hua!")
else:
    st.success("✅ Model ready hai!")
    st.write("### 📷 Sign Karein")
    st.info("👆 Haath bilkul camera ke saamne rakho — seedha aur clear!")
    img_file = st.camera_input("Sign karein yahan")

    if img_file is not None:
        img = Image.open(img_file)
        img_array = np.array(img)

        IMG_SIZE = 64
        h, w = img_array.shape[:2]
        cx, cy = w//2, h//2
        size = min(h, w) // 3
        x1 = max(0, cx - size)
        x2 = min(w, cx + size)
        y1 = max(0, cy - size)
        y2 = min(h, cy + size)

        hand_roi = img_array[y1:y2, x1:x2]
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_RGB2GRAY)
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
            bot_reply = chatbot_response(sign_name)
            st.session_state.chat_history.append({
                "sign": sign_name.upper(),
                "reply": bot_reply
            })
            st.success(f"🤖 Chatbot: {bot_reply}")
            text_to_speech(bot_reply)
        else:
            st.warning("Dobara try karein — haath bilkul beech mein rakho!")

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
