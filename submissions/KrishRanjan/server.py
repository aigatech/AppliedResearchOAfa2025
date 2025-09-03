from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import requests

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = None

# model trained to classify sentiment of twitter/reddit posts
sentiment_pipeline = pipeline(
    "text-classification", model="StephanAkkerman/FinTwitBERT-sentiment"
)


# fetch posts from reddit's public api
# return json response
@app.route("/api/reddit", methods=["GET"])
def wsb():
    sort = request.args.get("sort", "hot")
    limit = request.args.get("limit", 10)
    time_filter = request.args.get("t")

    url = f"https://www.reddit.com/r/wallstreetbets/{sort}.json"
    params = {"limit": limit}
    if time_filter:
        params["t"] = time_filter

    headers = {"User-Agent": "local-proxy"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": "Failed to fetch Reddit data", "details": str(e)}), 500


# get sentiment of reddit posts using finbert model
# classifies them as bearish, bullish, or neutral and gives a score rating
@app.route("/api/sentiment", methods=["POST"])
def sentiment():
    body = request.get_json(force=True)

    if "inputs" not in body:
        return jsonify({"error": "Missing 'inputs' in request body"}), 400

    try:
        texts = body["inputs"]
        results = [sentiment_pipeline(text)[0] for text in texts]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3500)
