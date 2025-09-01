# simulation.py
# AI@GT Applied Research Submission
# Project: Object Recognition & Intelligent Operational Navigator (ORION)

import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import random
import time
from collections import deque
from transformers import pipeline

CELL_SIZE = 25
GRID_WIDTH = 32
GRID_HEIGHT = 24

VALID_COLORS = [
    'saddlebrown', 'sienna', 'peru', 'burlywood', 'sandybrown', 'tan', 'chocolate',
    'maroon', 'darkred', 'firebrick', 'brown', 'indianred', 'rosybrown',
    'darkslateblue', 'midnightblue', 'navy', 'darkblue', 'mediumblue',
    'deeppink', 'hotpink', 'palevioletred', 'mediumvioletred',
    'darkorange', 'orange', 'coral', 'tomato', 'orangered',
    'gold', 'goldenrod', 'darkgoldenrod', 'yellow',
    'darkolivegreen', 'olivedrab', 'forestgreen', 'seagreen', 'teal', 'darkcyan',
    'purple', 'darkorchid', 'blueviolet', 'indigo', 'slategray'
]
OBJECT_CANDIDATES = ["chair", "couch", "table", "tv", "person", "cat", "dog", "bed", "potted plant",  "bench"]

EMPTY = 0
OBSTACLE = 1

class RobotEnvironment:
    # Handles the grid, robot, and obstacle data
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.robot_pos = (0, 0)
        self.labeled_obstacles = {}
        self.obstacle_colors = {}

    def generateGridFromWorld(self, detections, image_size):
        # Clears the old world and builds a new one from object detection results
        self.grid.fill(EMPTY)
        self.labeled_obstacles.clear()
        
        img_w, img_h = image_size
        scale_x = self.width / img_w
        scale_y = self.height / img_h

        label_counts = {}

        for detection in detections:
            base_label = detection['label']
            
            # Name objects uniquely
            count = label_counts.get(base_label, 0) + 1
            label_counts[base_label] = count
            unique_label = f"{base_label}_{count}"

            box = detection['box']
            xmin = int(box['xmin'] * scale_x)
            xmax = int(box['xmax'] * scale_x)
            ymin = int(box['ymin'] * scale_y)
            ymax = int(box['ymax'] * scale_y)
            
            self.labeled_obstacles[unique_label] = []

            # Mark the grid cells covered by the bounding box as obstacles
            for r in range(ymin, ymax):
                for c in range(xmin, xmax):
                    if 0 <= r < self.height and 0 <= c < self.width:
                        self.grid[r, c] = OBSTACLE
                        self.labeled_obstacles[unique_label].append((c, r))

        empty_cells = self.getAllEmptyCells()
        if empty_cells:
            self.robot_pos = random.choice(empty_cells)
        else:
            self.robot_pos = (0, 0) 

    def getAllEmptyCells(self):
        # Gets a list of all walkable coordinates
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == EMPTY:
                    cells.append((x, y))
        return cells

def findShortestPath(grid, start, goals):
    # Standard BFS
    width, height = grid.shape[1], grid.shape[0]
    queue = deque([[start]])
    visited = {start}

    goal_set = set(goals)
    while queue:
        path = queue.popleft()
        x, y = path[-1]

        if (x, y) in goal_set:
            return path 

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = x + dx, y + dy

            if (0 <= next_x < width and 0 <= next_y < height and
                    grid[next_y, next_x] == EMPTY and
                    (next_x, next_y) not in visited):
                
                visited.add((next_x, next_y))
                new_path = list(path)
                new_path.append((next_x, next_y))
                queue.append(new_path)
    return None

class RobotSimulatorGUI:
    # Manages the Tkinter GUI, user input, and AI model interactions
    def __init__(self, root, env):
        self.root = root
        self.root.title("Object Recognition & Intelligent Operational Navigator (ORION)")
        self.env = env
        
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(main_frame, width=GRID_WIDTH * CELL_SIZE, height=GRID_HEIGHT * CELL_SIZE, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.legend_frame = tk.Frame(main_frame, padx=10, pady=10)
        self.legend_frame.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(self.legend_frame, text="Legend", font=("Arial", 14, "bold")).pack(anchor="w")

        controls_frame = tk.Frame(root)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(controls_frame, text="Load an image to begin the simulation.", wraplength=700)
        self.status_label.pack(pady=5)

        command_frame = tk.Frame(controls_frame)
        command_frame.pack(pady=5, fill=tk.X, expand=True)
        tk.Label(command_frame, text="Command:").pack(side=tk.LEFT)
        self.command_entry = tk.Entry(command_frame)
        self.command_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.execute_button = tk.Button(command_frame, text="Execute Plan", command=self.executeNLPCommand)
        self.execute_button.pack(side=tk.LEFT)
        
        tk.Button(controls_frame, text="Load Environment from Image...", command=self.loadImageAndBuildWorld).pack(pady=5)
        
        self.object_detector = None
        self.color_generator = None
        self.updateLegend()

    def loadImageAndBuildWorld(self):
        
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not filepath: return
            
        self.status_label.config(text="Loading AI models... This may take a moment on first run.")
        self.root.update()

        # Initialize Hugging Face models on first use.
        if self.object_detector is None:
            print("Loading zero-shot object detection model...")
            self.object_detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
        if self.color_generator is None:
            print("Loading text generation model for color assignment...")
            self.color_generator = pipeline(model="distilgpt2", task="text-generation")
        
        self.status_label.config(text="Analyzing image for objects...")
        self.root.update()
        image = Image.open(filepath).convert("RGB")
        detections = self.object_detector(image, candidate_labels=OBJECT_CANDIDATES)
        
        self.env.generateGridFromWorld(detections, image.size)
        
        self.env.obstacle_colors.clear()
        detected_base_labels = sorted(list(set(lbl.split('_')[0] for lbl in self.env.labeled_obstacles.keys())))
        
        for label in detected_base_labels:
            self.status_label.config(text=f"AI is choosing a color for '{label}'...")
            self.root.update()
            
            prompt = f"A common color for a {label} is"
            result = self.color_generator(prompt, max_new_tokens=5, num_return_sequences=1)
            generated_text = result[0]['generated_text'].replace(prompt, "").strip().lower()
            
            
            chosen_color = "gray" 
            for word in generated_text.split():
                clean_word = word.strip(".,'\"")
                if clean_word in VALID_COLORS:
                    chosen_color = clean_word
                    break
            
            self.env.obstacle_colors[label] = chosen_color

        self.updateLegend()
        self.drawGrid()
        
        if detected_base_labels:
            self.status_label.config(text=f"World built! Detected: {', '.join(detected_base_labels)}. Ready for commands.")
        else:
            self.status_label.config(text="Could not detect any known objects. Try a clearer image.")

    def updateLegend(self):
        # Clears and redraws the legend based on the current world state
        for widget in self.legend_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.destroy()

        self.addLegendItem("blue", "Robot")
        
        detected_base_labels = sorted(list(set(lbl.split('_')[0] for lbl in self.env.labeled_obstacles.keys())))
        for label in detected_base_labels:
            color = self.env.obstacle_colors.get(label, "gray")
            self.addLegendItem(color, label.capitalize())

    def addLegendItem(self, color, text):
        # Helper to create a single row in the legend
        frame = tk.Frame(self.legend_frame)
        frame.pack(anchor="w", pady=2)
        tk.Label(frame, text="", bg=color, width=2, relief="solid", borderwidth=1).pack(side=tk.LEFT)
        tk.Label(frame, text=f" - {text}", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

    def drawGrid(self):
        # Renders the entire world state to the canvas
        self.canvas.delete("all")
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.grid[y, x] == EMPTY:
                    self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE, 
                                                 fill="white", outline="#f0f0f0")

        for unique_label, cells in self.env.labeled_obstacles.items():
            base_label = unique_label.split('_')[0]
            color = self.env.obstacle_colors.get(base_label, "gray")
            
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

            for x, y in cells:
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)
                self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                             fill=color, outline="black")
            
            center_x = (min_x + max_x + 1) / 2 * CELL_SIZE
            center_y = (min_y + max_y + 1) / 2 * CELL_SIZE
            self.canvas.create_text(center_x, center_y, text=unique_label, fill="white", font=("Arial", 8, "bold"))

        if self.env.robot_pos:
            rx, ry = self.env.robot_pos
            self.canvas.create_oval(rx * CELL_SIZE + 4, ry * CELL_SIZE + 4, (rx + 1) * CELL_SIZE - 4, (ry + 1) * CELL_SIZE - 4, 
                                    fill="blue", outline="white", width=2)
        
        self.root.update()
        
    def executeNLPCommand(self):
        # Parses the user's natural language command into an executable plan
        command_text = self.command_entry.get().strip().lower()
        if not command_text: return

        all_detected_labels = list(self.env.labeled_obstacles.keys())
        all_base_labels = list(set(lbl.split('_')[0] for lbl in all_detected_labels))
        
        all_possible_targets = sorted(all_detected_labels + all_base_labels, key=len, reverse=True)
        
        found_tasks_with_indices = []
        temp_command_text = command_text
        
        for target in all_possible_targets:
            index = temp_command_text.find(target)
            if index != -1:
                found_tasks_with_indices.append((index, target))
                temp_command_text = temp_command_text.replace(target, '_' * len(target), 1)

        found_tasks_with_indices.sort()
        
        task_sequence = []
        for _, task in found_tasks_with_indices:
            if task not in task_sequence:
                base_task = task.split('_')[0]
                is_more_specific_version_present = any(t.startswith(base_task + '_') for t in task_sequence)
                
                if not ('_' not in task and is_more_specific_version_present):
                    task_sequence.append(task)
        
        if not task_sequence:
            self.status_label.config(text=f"No known targets in command. Try: {', '.join(all_base_labels)}")
            return

        self.status_label.config(text=f"Plan confirmed: {' -> '.join(task_sequence)}. Executing...")
        self.execute_button.config(state=tk.DISABLED)
        
        current_pos = self.env.robot_pos
        for i, task_item in enumerate(task_sequence):
            target_label = task_item 

            if '_' not in target_label:
                # Filter for possible targets that actually exist on the grid
                possible_targets = [
                    lbl for lbl in all_detected_labels 
                    if lbl.startswith(target_label) and self.env.labeled_obstacles[lbl]
                ]
                if not possible_targets: continue
                
                target_label = min(possible_targets, 
                                   key=lambda lbl: abs(self.env.labeled_obstacles[lbl][0][0] - current_pos[0]) + abs(self.env.labeled_obstacles[lbl][0][1] - current_pos[1]))

            self.status_label.config(text=f"Step {i+1}/{len(task_sequence)}: Pathfinding to '{target_label}'...")
            self.root.update()

            obstacle_cells = self.env.labeled_obstacles[target_label]
            adjacent_cells = set()
            for ox, oy in obstacle_cells:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_x, adj_y = ox + dx, oy + dy
                    if (0 <= adj_x < self.env.width and 0 <= adj_y < self.env.height and self.env.grid[adj_y, adj_x] == EMPTY):
                        adjacent_cells.add((adj_x, adj_y))
            
            if not adjacent_cells:
                self.status_label.config(text=f"Execution failed: '{target_label}' is completely blocked.")
                self.execute_button.config(state=tk.NORMAL)
                return
            
            path = findShortestPath(self.env.grid, current_pos, adjacent_cells)
            
            if not path:
                self.status_label.config(text=f"Execution failed: Cannot find a path to '{target_label}'.")
                self.execute_button.config(state=tk.NORMAL)
                return
            
            self.animate(path)
            current_pos = self.env.robot_pos 

        self.status_label.config(text="Plan successfully completed!")
        self.execute_button.config(state=tk.NORMAL)
        self.command_entry.delete(0, tk.END)

    def animate(self, path):
        # Animates the robot's movement along the calculated path
        for pos in path[1:]: 
            self.env.robot_pos = pos
            self.drawGrid()
            time.sleep(0.075)

if __name__ == "__main__":
    root = tk.Tk()
    env = RobotEnvironment(GRID_WIDTH, GRID_HEIGHT)
    app = RobotSimulatorGUI(root, env)
    root.mainloop()