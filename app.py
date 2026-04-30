import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image

st.set_page_config(
    page_title="The Gesture Genius",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 The Gesture Genius")
st.subheader("Pakistani Sign Language (PSL) Detection")
st.markdown("### Hackathon Project - Team: The Gesture Genius")
st.write("---")

st.info("""
**Kaise Kaam Karega yeh Project?**

1. Webcam se PSL sign karo
2. AI sign ko text mein convert karega
3. Result screen pe dikhega
""")

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

st.write("---")
st.write("### 📷 Sign Detection")

if model is None:
    st.error("Model load nahi hua! gesture_model.pkl check karo.")
else:
    st.success("Model ready hai!")

    # Camera input - Streamlit ka built-in
    img_file = st.camera_input("Camera kholein aur sign karein")

    if img_file is not None:
        # Image process karo
        img = Image.open(img_file)
        img_array = np.array(img)

        IMG_SIZE = 64
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        features = normalized.flatten().reshape(1, -1)

        # Prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction] * 100
        sign_name = SIGNS[prediction]

        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Sign", sign_name.upper())
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        if confidence > 60:
            st.success(f"Sign pehchana gaya: {sign_name.upper()}")
        else:
            st.warning("Confidence kam hai - dobara try karo")

st.write("---")
st.write("### 👥 Our Team")
st.write("**The Gesture Genius**")
st.write("Second Semester Students")
st.write("---")
st.caption("Hackathon Submission - Built with ❤️ in Pakistan")
