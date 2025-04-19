from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import sqlite3
import os
import gdown

app = Flask(__name__)
app.secret_key = "secret-key-change-this"

DB_PATH = "forum.db"

# 🗕 Initialiser la base de données
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS tweets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT,
                        content TEXT,
                        label TEXT,
                        proba_misog REAL,
                        proba_nonmisog REAL,
                        likes INTEGER DEFAULT 0,
                        comments INTEGER DEFAULT 0
                    )''')
        c.execute('''CREATE TABLE IF NOT EXISTS comments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id INTEGER,
                        author TEXT,
                        text TEXT,
                        FOREIGN KEY(tweet_id) REFERENCES tweets(id)
                    )''')
        conn.commit()

init_db()

# 📅 Téléchargement du modèle depuis Google Drive
MODEL_DIR = "model_camembert_final"
if not os.path.exists(MODEL_DIR):
    folder_id = "1B2Nriruvr4lpWs7OA7Gyekk80JPwkTcK"
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{folder_id}",
        output=MODEL_DIR,
        quiet=False,
        use_cookies=False
    )

# 🔄 Charger le modèle et le tokenizer
model = CamembertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model.eval()

# 🩰 Prédiction
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    label = logits.argmax(dim=1).item()
    return label, probs

# 📜 Page principale
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        content = request.form.get("texte", "").strip()
        if user and content:
            label, probs = predict_text(content)
            label_str = "Misogyne" if label == 1 else "Non Misogyne"
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO tweets (user, content, label, proba_misog, proba_nonmisog) VALUES (?, ?, ?, ?, ?)",
                          (user, content, label_str, float(probs[1]), float(probs[0])))
                conn.commit()
        return redirect(url_for("index"))

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, user, content, label, proba_misog, proba_nonmisog, likes, comments FROM tweets ORDER BY id DESC")
        tweets = c.fetchall()

        c.execute("SELECT tweet_id, author, text FROM comments")
        all_comments = c.fetchall()
        comments_by_tweet = {}
        for tweet_id, author, text in all_comments:
            comments_by_tweet.setdefault(tweet_id, []).append((author, text))

    return render_template("index.html", tweets=tweets, comments_by_tweet=comments_by_tweet)

# ❤️ Like
@app.route("/like/<int:tweet_id>")
def like(tweet_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT label FROM tweets WHERE id = ?", (tweet_id,))
        label = c.fetchone()
        if label and label[0] == "Non Misogyne":
            c.execute("UPDATE tweets SET likes = likes + 1 WHERE id = ?", (tweet_id,))
            conn.commit()
    return redirect(url_for("index"))

# 💬 Commentaires
@app.route("/comment/<int:tweet_id>", methods=["GET", "POST"])
def comment(tweet_id):
    if request.method == "POST":
        author = request.form.get("author", "").strip()
        text = request.form.get("text", "").strip()
        if author and text:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO comments (tweet_id, author, text) VALUES (?, ?, ?)",
                          (tweet_id, author, text))
                c.execute("UPDATE tweets SET comments = comments + 1 WHERE id = ?", (tweet_id,))
                conn.commit()
        return redirect(url_for("index"))

    return render_template("comment_form.html", tweet_id=tweet_id)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
