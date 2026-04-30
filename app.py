import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="The Gesture Genius",
    page_icon="👐",
    layout="centered"
)

# Header
st.title("👐 The Gesture Genius")
st.subheader("Pakistani Sign Language (PSL) Chatbot")

st.markdown("### Hackathon Project - Team: The Gesture Genius")

st.write("---")

# Main Description
st.info("""
**Kaise Kaam Karega yeh Project?**

1. Webcam se PSL sign karo  
2. AI sign ko text mein convert karega  
3. Chatbot jawab dega  
4. Jawab text + voice (TTS) mein milega
""")

st.success("✅ Python + Streamlit setup successfully complete ho gaya hai!")

# Team Section
st.write("### 👥 Our Team")
st.write("**The Gesture Genius**")
st.write("Second Semester Students")

if st.button("🚀 Start Camera Test"):
    st.balloons()
    st.warning("Abhi camera feature add nahi hua. Yeh sirf frontend hai.")

st.write("---")
st.caption("Hackathon Submission - Built with ❤️ in Pakistan")

# Footer
st.write("**Status:** Basic App Ready | Next: Webcam + PSL Detection")
from streamlit_webrtc import webrtc_streamer

st.write("### 🎥 Live Camera")

webrtc_streamer(key="camera")