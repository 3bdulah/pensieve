import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
from chatbot import chat
import secrets

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a message."})

    history = session.get("history", [])
    reply, history = chat(user_message, history)
    session["history"] = history
    return jsonify({"reply": reply})

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
