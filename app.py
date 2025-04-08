from flask import Flask, render_template, request
import instaloader
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stopwords.words("english"))
    return text

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        return result[0]['label']
    except:
        return "unknown"

def extract_shortcode(url):
    match = re.search(r"/p/([^/]+)/", url)
    return match.group(1) if match else None

def fetch_post_data(shortcode):
    try:
        L = instaloader.Instaloader()

        # Load your browser sessionid here
        sessionid = "PASTE-YOUR-SESSIONID-HERE"  # 🔁 Replace this with your actual session ID
        L.context._session_cookie = {"sessionid": sessionid}

        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return {
            'caption': post.caption if post.caption else "",
            'comments': [comment.text for comment in post.get_comments()]
        }
    except Exception as e:
        return {"error": f"Session login failed: {str(e)}"}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        url = request.form['url']
        shortcode = extract_shortcode(url)

        if not shortcode:
            error = "Invalid Instagram URL!"
        else:
            data = fetch_post_data(shortcode)

            if "error" in data:
                error = data["error"]
            else:
                captions = [clean_text(data["caption"])]
                comments = [clean_text(c) for c in data["comments"][:10]]
                caption_sentiment = get_sentiment(captions[0])
                caption_emotion = detect_emotion(captions[0])

                analyzed_comments = []
                for original, text in zip(data["comments"][:10], comments):
                    analyzed_comments.append({
                        "text": original,
                        "sentiment": get_sentiment(text),
                        "emotion": detect_emotion(text)
                    })

                # WordCloud
                text_blob = " ".join(captions + comments)
                wc = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
                wordcloud_path = "static/wordcloud.png"
                wc.to_file(wordcloud_path)

                # Emotion Pie Chart
                emotion_counts = {}
                for text in captions + comments:
                    emo = detect_emotion(text)
                    emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

                plt.figure(figsize=(5, 5))
                plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
                plt.axis("equal")
                piechart_path = "static/emotion_pie.png"
                plt.savefig(piechart_path)
                plt.close()

                result = {
                    "caption": data["caption"],
                    "caption_sentiment": caption_sentiment,
                    "caption_emotion": caption_emotion,
                    "comments": analyzed_comments,
                    "wordcloud": wordcloud_path,
                    "piechart": piechart_path
                }

    return render_template("index.html", result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
