from flask import Flask, render_template, request, jsonify, send_file
from backend import F1KnowledgeBase
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Initialize the backend
user_login = "example_user"
api_key = os.getenv("API_KEY")
kb = F1KnowledgeBase(user_login=user_login, api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        if not user_input:
            raise ValueError("No message provided")
        response = kb.generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/text-to-image", methods=["POST"])
def text_to_image():
    try:
        text = request.json.get("text")
        if not text:
            raise ValueError("No text provided")
        image_path = "output_image.png"
        kb.text_to_image(text, image_path)
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)