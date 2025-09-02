#!/usr/bin/env python3
import tkinter as tk

from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageDraw, ImageFont

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection


class ToiletPaperDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Toilet Paper Fullness Detector")
        self.root.geometry("800x600")
        
        self.model = None
        self.processor = None
        self.current_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="Toilet Paper Fullness Detector", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        instructions = ttk.Label(main_frame, 
                               text="Upload an image with toilet paper rolls facing the camera.\nThe app will detect them and estimate fullness based on paper thickness.",
                               font=("Arial", 10))
        instructions.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        self.upload_btn = ttk.Button(button_frame, text="Upload Image", 
                                   command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.detect_btn = ttk.Button(button_frame, text="Detect & Analyze", 
                                   command=self.detect_toilet_paper, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT)
        

        
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(results_frame, text="No image loaded")
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        self.results_text = tk.Text(results_frame, height=8, width=40, wrap=tk.WORD)
        self.results_text.grid(row=0, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.debug_text = tk.Text(results_frame, height=8, width=30, wrap=tk.WORD, bg="#f0f0f0")
        self.debug_text.grid(row=0, column=2, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar1 = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar1.grid(row=0, column=3, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar1.set)
        
        scrollbar2 = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.debug_text.yview)
        scrollbar2.grid(row=0, column=4, sticky=(tk.N, tk.S))
        self.debug_text.configure(yscrollcommand=scrollbar2.set)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
    def load_model(self):
        if self.model is None:
            try:
                self.debug_text.delete(1.0, tk.END)
                self.debug_text.insert(tk.END, "Loading DETR model...\n")
                self.root.update()
                self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                self.debug_text.insert(tk.END, "Model loaded successfully!\n")
                self.debug_text.insert(tk.END, f"Model: {self.model.config.model_type}\n")
                self.debug_text.insert(tk.END, f"Classes: {len(self.model.config.id2label)}\n")
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                return False
        return True
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                self.display_image(self.current_image)
                self.detect_btn.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded successfully!\nClick 'Detect & Analyze' to find toilet paper rolls.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image, max_size=(400, 300)):
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def detect_toilet_paper(self):
        if not self.current_image:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        if not self.load_model():
            return
        
        try:
            self.detect_btn.config(state=tk.DISABLED)
            self.debug_text.delete(1.0, tk.END)
            self.debug_text.insert(tk.END, "Starting detection...\n")
            self.root.update()
            
            image_rgb = self.current_image.convert("RGB")
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image_rgb.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.3
            )[0]
            
            self.debug_text.insert(tk.END, f"Image size: {image_rgb.size}\n")
            self.debug_text.insert(tk.END, f"Detected objects: {len(results['scores'])}\n")
            
            # Show all detected objects
            for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
                if score.item() > 0.3:
                    object_name = self.model.config.id2label[label.item()]
                    self.debug_text.insert(tk.END, f"  {i+1}. {object_name} ({score.item():.2f})\n")
            
            self.root.update()
            
            potential_rolls = self.find_potential_toilet_paper_rolls(image_rgb, results)
            self.debug_text.insert(tk.END, f"Potential rolls: {len(potential_rolls)}\n")
            self.root.update()
            
            analyzed_rolls = []
            for i, roll in enumerate(potential_rolls):
                self.debug_text.insert(tk.END, f"\nAnalyzing roll {i+1}:\n")
                fullness = self.estimate_fullness(image_rgb, roll)
                self.debug_text.insert(tk.END, f"  Fullness: {fullness}%\n")
                analyzed_rolls.append((roll, fullness))
                self.root.update()
            
            self.display_results(image_rgb, analyzed_rolls, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
        finally:
            self.detect_btn.config(state=tk.NORMAL)
    
    def find_potential_toilet_paper_rolls(self, image, detection_results):
        potential_rolls = []
        
        for score, label, box in zip(detection_results["scores"], 
                                   detection_results["labels"], 
                                   detection_results["boxes"]):
            x_min, y_min, x_max, y_max = box.tolist()
            
            if score.item() > 0.3:
                width = x_max - x_min
                height = y_max - y_min
                aspect_ratio = width / height
                
                if 0.7 <= aspect_ratio <= 1.4 and width > 50 and height > 50:
                    potential_rolls.append({
                        'box': (int(x_min), int(y_min), int(x_max), int(y_max)),
                        'score': score.item(),
                        'label': self.model.config.id2label[label.item()]
                    })
        
        return potential_rolls
    
    def estimate_fullness(self, image, roll_info):
        x_min, y_min, x_max, y_max = roll_info['box']
        
        roi = image.crop((x_min, y_min, x_max, y_max))
        gray_roi = roi.convert('L')
        
        # conversion to list for the whole pixel processing part
        pixels = list(gray_roi.getdata())
        total_pixels = len(pixels)
        
        paper_pixels = sum(1 for p in pixels if p > 160)
        hole_pixels = sum(1 for p in pixels if p < 60)
        # pixel calculation
        if total_pixels > 0:
            hole_ratio = hole_pixels / total_pixels
            paper_ratio = paper_pixels / total_pixels
            self.debug_text.insert(tk.END, f"  Paper: {paper_ratio:.1%}, Hole: {hole_ratio:.1%}\n")
            if hole_ratio > 0.03:
                fullness = max(0, min(100, (1 - hole_ratio) * 100))
            else:
                fullness = paper_ratio * 100
        else:
            fullness = 0
        
        return round(fullness, 1)
    
    def display_results(self, image, analyzed_rolls, all_results):
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw all detected objects
        for score, label, box in zip(all_results["scores"], all_results["labels"], all_results["boxes"]):
            if score.item() > 0.3:
                x_min, y_min, x_max, y_max = box.tolist()
                object_name = self.model.config.id2label[label.item()]
                
                # Draw bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
                
                # Draw label
                label_text = f"{object_name} ({score.item():.2f})"
                draw.text((x_min, y_min - 20), label_text, fill="blue", font=font)
        
        # Draw toilet paper rolls with red boxes
        for i, (roll_info, fullness) in enumerate(analyzed_rolls, 1):
            x_min, y_min, x_max, y_max = roll_info['box']
            
            # Draw red box over the blue one
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            
            text = f"Roll {i}: {fullness}% full"
            draw.text((x_min, y_min - 40), text, fill="red", font=font)
        
        results_text = "Detection Results:\n\n"
        
        if not analyzed_rolls:
            results_text += "No toilet paper rolls detected.\n"
            results_text += "Try uploading a clearer image with toilet paper rolls facing the camera."
        else:
            results_text += f"Found {len(analyzed_rolls)} potential toilet paper roll(s):\n\n"
            
            for i, (roll_info, fullness) in enumerate(analyzed_rolls, 1):
                results_text += f"Roll {i}:\n"
                results_text += f"  - Confidence: {roll_info['score']:.2f}\n"
                results_text += f"  - Fullness: {fullness}%\n"
                results_text += f"  - Status: {self.get_fullness_status(fullness)}\n\n"
        
        self.display_image(result_image)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
    
    def get_fullness_status(self, fullness):
        if fullness >= 80:
            return "Very Full"
        elif fullness >= 60:
            return "Mostly Full"
        elif fullness >= 40:
            return "Half Full"
        elif fullness >= 20:
            return "Getting Low"
        else:
            return "Almost Empty"

def main():
    print("Starting Toilet Paper Fullness Detector...")
    app = ToiletPaperDetector()
    app.root.mainloop()

if __name__ == "__main__":
    main()
