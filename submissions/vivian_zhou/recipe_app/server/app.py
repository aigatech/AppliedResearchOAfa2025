from flask import Flask, request, jsonify
from flask_cors import CORS
from model import RecipeGenerator

app = Flask(__name__)
CORS(app)  # Dev only; lock down origins for prod.

# Initialize the recipe generator once when the app starts
recipe_generator = RecipeGenerator()

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/generate")
def generate():
    data = request.get_json(force=True) or {}
    pantry = (data.get("pantry") or "").strip()
    if not pantry:
        return jsonify({"error": "pantry is required"}), 400
    diet = data.get("diet", "none")
    time_limit = data.get("time", 30)
    
    try:
        recipe = recipe_generator.generate_recipe(pantry, diet, time_limit)
        return jsonify(recipe)
    except Exception as e:
        return jsonify({"error": f"Failed to generate recipe: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)

