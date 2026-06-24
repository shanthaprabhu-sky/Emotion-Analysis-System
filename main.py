from emotion_detector import detect_emotion

emotion, confidence = detect_emotion(
    "sample_images/happy_image.jpg"
)

print(f"Emotion: {emotion}")
print(f"Confidence: {confidence:.2f}%")