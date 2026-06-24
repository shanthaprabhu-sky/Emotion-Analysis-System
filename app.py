import streamlit as st
import os
from PIL import Image
from emotion_detector import detect_emotion

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
from deepface import DeepFace

os.makedirs("assets", exist_ok=True)

st.set_page_config(
    page_title="Emotion Analysis System",
    page_icon="😊",
    layout="wide"
)

st.sidebar.title("😊 Emotion Analysis System")

st.sidebar.markdown("### Tech Stack")

st.sidebar.success("Python")
st.sidebar.success("Streamlit")
st.sidebar.success("DeepFace")
st.sidebar.success("OpenCV")
st.sidebar.success("TensorFlow")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Detect emotions from:
    - Uploaded Images
    - Live Webcam Feed
    """
)

st.title("😊 AI-Powered Emotion Analysis System")

st.markdown(
    """
    Detect human emotions from images and live webcam feeds
    using Deep Learning and Computer Vision.
    """
)

class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        try:

            result = DeepFace.analyze(
                img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )

            emotion = result[0]["dominant_emotion"]

            cv2.putText(
                img,
                f"Emotion: {emotion}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except:
            pass

        return img
    
tab1, tab2 = st.tabs([
    "📷 Image Upload",
    "🎥 Webcam Detection"
])

# -------------------------
# IMAGE UPLOAD TAB
# -------------------------

with tab1:

    st.header("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        file_path = os.path.join(
            "assets",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Analyze Emotion"):

            emotion, confidence, emotions = detect_emotion(file_path)

            if emotion is None:

                st.error(
                    "❌ No face detected in the uploaded image."
                )

            else:

                st.success(
                    f"Detected Emotion: {emotion.capitalize()}"
                )

                st.info(
                    f"Confidence Score: {confidence:.2f}%"
                )

                st.subheader("📊 Emotion Breakdown")

                st.bar_chart(emotions)

                st.subheader("📋 Detailed Scores")

                for emotion_name, score in emotions.items():

                         st.write(
                            f"{emotion_name.capitalize()} : {score:.2f}%"
                                  )
# -------------------------
# WEBCAM TAB
# -------------------------

with tab2:

    st.header("🎥 Live Webcam Emotion Detection")

    st.write(
        "Allow camera access and show different facial expressions."
    )

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionDetector
    )