# Recipe Generator App Setup Instructions

This project contains a recipe generator application with a React frontend and Flask backend.

## Project Structure
```
submissions/vivian_zhou/recipe_app/
├── client/          # React frontend
├── server/          # Flask backend
└── README.md
```

## Setup Instructions

### Backend Setup

1. Navigate to the server directory:
```bash
cd submissions/vivian_zhou/recipe_app/server
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the Flask server:
```bash
python app.py
```

The backend will be available at http://localhost:8000

### Frontend Setup

1. Navigate to the client directory:
```bash
cd submissions/vivian_zhou/recipe_app/client
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will be available at http://localhost:3000

## Usage

1. Start both the backend and frontend servers
2. Open http://localhost:3000 in your browser
3. Enter ingredients in the pantry field (comma-separated)
4. Select dietary restrictions (optional)
5. Set cooking time
6. Click "Generate Recipe" to get an AI-generated recipe
