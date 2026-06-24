from deepface import DeepFace

def detect_emotion(image_path):

    try:

        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="opencv"
        )

        emotions = result[0]["emotion"]

        dominant_emotion = result[0]["dominant_emotion"]

        confidence = emotions[dominant_emotion]

        return dominant_emotion, confidence, emotions

    except Exception:

        return None, None, None