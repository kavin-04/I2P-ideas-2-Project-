from flask import Flask, request, jsonify, send_from_directory
from backend import generate_response

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static"
)

@app.route("/")
def index():
    return send_from_directory(".", "i2p.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "")
    cont = data.get("continue", False)

    if not user_input and not cont:
        return jsonify({"error": "Message is required"}), 400

    result = generate_response(user_input, continue_generation=cont)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
