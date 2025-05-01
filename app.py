from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import psycopg2
import os
import gdown

app = Flask(__name__)
app.secret_key = "secret-key-change-this"

# 🔌 Connexion PostgreSQL
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

# 🗕 Initialiser la base PostgreSQL
def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            id SERIAL PRIMARY KEY,
            user TEXT,
            content TEXT,
            label TEXT,
            proba_misog REAL,
            proba_nonmisog REAL,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id SERIAL PRIMARY KEY,
            tweet_id INTEGER,
            author TEXT,
            text TEXT,
            FOREIGN KEY(tweet_id) REFERENCES tweets(id)
        );
    ''')
    conn.commit()
    cur.close()
    conn.close()

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

# 🔄 Charger modèle
model = CamembertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model.eval()

# 🧠 Prédiction
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    label = logits.argmax(dim=1).item()
    return label, probs

# 🏠 Page principale
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        content = request.form.get("texte", "").strip()
        if user and content:
            label, probs = predict_text(content)
            label_str = "Misogyne" if label == 1 else "Non Misogyne"
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO tweets (user, content, label, proba_misog, proba_nonmisog)
                VALUES (%s, %s, %s, %s, %s)
            """, (user, content, label_str, float(probs[1]), float(probs[0])))
            conn.commit()
            cur.close()
            conn.close()
        return redirect(url_for("index"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, user, content, label, proba_misog, proba_nonmisog, likes, comments FROM tweets ORDER BY id DESC")
    tweets = cur.fetchall()

    cur.execute("SELECT tweet_id, author, text FROM comments")
    all_comments = cur.fetchall()
    cur.close()
    conn.close()

    comments_by_tweet = {}
    for tweet_id, author, text in all_comments:
        comments_by_tweet.setdefault(tweet_id, []).append((author, text))

    return render_template("index.html", tweets=tweets, comments_by_tweet=comments_by_tweet)

# ❤️ Like
@app.route("/like/<int:tweet_id>")
def like(tweet_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT label FROM tweets WHERE id = %s", (tweet_id,))
    label = cur.fetchone()
    if label and label[0] == "Non Misogyne":
        cur.execute("UPDATE tweets SET likes = likes + 1 WHERE id = %s", (tweet_id,))
        conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for("index"))

# 💬 Commentaires
@app.route("/comment/<int:tweet_id>", methods=["GET", "POST"])
def comment(tweet_id):
    if request.method == "POST":
        author = request.form.get("author", "").strip()
        text = request.form.get("text", "").strip()
        if author and text:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("INSERT INTO comments (tweet_id, author, text) VALUES (%s, %s, %s)", (tweet_id, author, text))
            cur.execute("UPDATE tweets SET comments = comments + 1 WHERE id = %s", (tweet_id,))
            conn.commit()
            cur.close()
            conn.close()
        return redirect(url_for("index"))
    return render_template("comment_form.html", tweet_id=tweet_id)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
