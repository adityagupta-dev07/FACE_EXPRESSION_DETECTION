# ai_judge.py

def judge_expression(detected_emotion, current_expression):
    """
    Simple rule-based caption generator.
    Safe: no AI APIs, no extra libraries.
    """

    captions = {
        "happy": "You look happy 😄 Keep smiling!",
        "sad": "You seem a bit sad 😢 Everything will be okay.",
        "angry": "You look angry 😠 Take a deep breath.",
        "surprise": "That expression says surprise 😲",
        "fear": "You look scared 😨 Stay calm.",
        "disgust": "Hmm… that expression shows disgust 😖",
        "neutral": "You look calm and neutral 🙂"
    }

    return captions.get(detected_emotion.lower(), "Reading your expression...")

