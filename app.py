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
import mediapipe as mp

st.set_page_config(
    page_title="The Gesture Genius",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 The Gesture Genius")
st.subheader("Pakistani Sign Language (PSL) Detection")
st.markdown("### Hackathon Project - Team: The Gesture Genius")
st.write("---")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            with open(fp.name, 'rb') as f:
                audio_data = f.read()
            os.unlink(fp.name)
        audio_b64 = base64.b64encode(audio_data).decode()
        st.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )
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
    st.info("👆 Haath bilkul camera ke saamne rakho!")

    img_file = st.camera_input("Sign karein yahan")

    if img_file is not None:
        img = Image.open(img_file)
        img_array = np.array(img)

        rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            features = np.array(landmarks).reshape(1, -1)
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
                st.warning("Dobara try karein!")
        else:
            st.warning("⚠️ Haath nahi dikh raha! Haath camera ke saamne rakho!")

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
