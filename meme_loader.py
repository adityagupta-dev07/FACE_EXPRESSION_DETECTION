def get_meme_by_expression(expression):
    memes = {
        "happy": {
            "meme_name": "Happy Meme",
            "image": "happy.gif",
            "default_caption": "Life is good 😄"
        },
        "sad": {
            "meme_name": "Sad Meme",
            "image": "sad.gif",
            "default_caption": "Pain is real 😔"
        },
        "angry": {
            "meme_name": "Angry Meme",
            "image": "angry.gif",
            "default_caption": "Control your rage 😡"
        },
        "fear": {
            "meme_name": "Fear Meme",
            "image": "fear.gif",
            "default_caption": "Something’s not right 😨"
        },
        "surprise": {
            "meme_name": "Surprise Meme",
            "image": "surprise.gif",
            "default_caption": "Unexpected!"
        },
        "neutral": {
            "meme_name": "Neutral Meme",
            "image": "neutral.gif",
            "default_caption": "Just vibing 😐"
        }
    }

    return memes.get(expression)
