from deepface import DeepFace

def detect_emotion_frame(frame):

    result = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv"
    )

    emotion = result[0]["dominant_emotion"]
    confidence = result[0]["emotion"][emotion]

    return emotion, confidence