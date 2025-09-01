# 🤖 🌍 Object Recognition & Intelligent Operational Navigator (ORION)

## 🤔 What it does

**ORION** is a Python-based robotics simulation that transforms a real-world image into a navigable 2D map. This project leverages multiple Hugging Face AI models to achieve a sophisticated level of scene understanding and human-robot interaction.

- **Vision-Based World Generation**: Load any top-down image of a room or area. The application uses a zero-shot object detection model (`google/owlvit-base-patch32`) to identify and map real-world objects like chairs, tables, and couches as obstacles.
- **Generative AI Color Assignment**: A text-generation model (`distilgpt2`) creatively assigns a common color to each detected object type, making every generated world visually unique.
- **Natural Language Command & Planning**: Command the robot using plain English. The AI parser understands complex, multi-step commands (e.g., "go to chair_1 then the tv") and can distinguish between specific targets (`chair_2`) and general targets ("the closest chair").
- **Optimal Pathfinding**: The robot uses a Breadth-First Search (BFS) algorithm to calculate and follow the mathematically shortest path to its targets, navigating around all detected obstacles.

---

## 🏃 How to Run

### 1. 🔧 Installation

**Prerequisites**: Python 3.x

1. Create a python virtual environment:

2. Install dependencies:

```bash
pip install numpy Pillow transformers torch
```

_Note: First run will download pre-trained models_

### 2. Execution

```bash
python simulation.py
```

A Tkinter GUI window will appear, ready for you to load an environment.

---

## 📃 Usage Guide

1. **Load an Environment**: Click **"Load World from Image..."** and select a top-down JPG or PNG image of a room or area.

![Main GUI](https://github.com/Hiptostee/image-hosting/blob/main/environment.jpeg?raw=true)

2. **AI Analysis**: Wait while the AI models process the image. The status bar will update as it detects objects and assigns colors. Once complete, the navigable map will be drawn on the canvas.

3. **Command the Robot**: Use the text entry box to give commands. Click **"Execute"** to run the plan.

### 🚀 Example Commands:

**⚠️ SPELL CORRECTLY AND INCLUDE UNDERSCORES WHERE NEEDED**

- `go to the tv` (finds closest 'tv' object)
- `chair_1` (targets specific 'chair_1' object)
- `go to couch_1, then the tv, then chair_2` (multi-step plan)
- `find the potted plant and then the bed` (multi-word objects)

### 📡 Future Plans:

- Upgrade NLP for nuanced commands
- Dynamic environments with moving obstacles
- Cost-based pathfinding using A\*
- Physics engine for realistic movement
