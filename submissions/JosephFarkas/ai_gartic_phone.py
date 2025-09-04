"""
One-player Gartic Phone

A game where users draw pictures, AI interprets them, and generates new pictures.
Uses Hugging Face models for image captioning and Stable Diffusion for generation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import io
import threading
from PIL import Image, ImageTk, ImageDraw
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class AIPictographyGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Gartic Phone")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.current_image = None
        self.generated_image = None
        
        # Model loading status
        self.models_loaded = False
        self.captioning_model = None
        self.captioning_processor = None
        self.generation_pipeline = None
        
        self.setup_ui()
        self.load_models_async()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Gartic Phone", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Draw a picture → AI interprets it → AI draws a new picture!",
                                font=("Arial", 12))
        instructions.pack(pady=(0, 10))
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Drawing area
        left_panel = ttk.LabelFrame(content_frame, text="Your Drawing", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Drawing canvas
        self.canvas = tk.Canvas(left_panel, bg="white", width=400, height=400)
        self.canvas.pack(pady=(0, 10))
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Drawing controls
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X)
        
        ttk.Button(controls_frame, text="Clear Canvas", 
                  command=self.clear_canvas).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Interpret Drawing", 
                  command=self.interpret_and_generate).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Save Drawing", 
                  command=self.save_drawing).pack(side=tk.LEFT)
        
        # Middle panel - AI Interpretation
        middle_panel = ttk.LabelFrame(content_frame, text="AI Interpretation", padding=10)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Caption display
        self.caption_text = tk.Text(middle_panel, height=8, width=30, wrap=tk.WORD)
        self.caption_text.pack(pady=(0, 10))
        
        # Status label - ensuring models are loaded
        self.status_label = ttk.Label(middle_panel, text="Load models first...")
        self.status_label.pack()
        
        # Right panel - AI Generated Image
        right_panel = ttk.LabelFrame(content_frame, text="AI Generated Image", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Generated image display
        self.generated_image_label = ttk.Label(right_panel, text="Generated image will appear here - be patient!")
        self.generated_image_label.pack(expand=True)
        
        # Generated image controls
        gen_controls_frame = ttk.Frame(right_panel)
        gen_controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(gen_controls_frame, text="Save Generated Image", 
                  command=self.save_generated_image).pack(side=tk.LEFT)
        ttk.Button(gen_controls_frame, text="Use as New Drawing", 
                  command=self.use_generated_as_drawing).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(gen_controls_frame, text="Start New Round", 
                  command=self.start_new_round).pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def load_models_async(self):
        """Load AI models in a separate thread"""
        def load_models():
            try:
                self.update_status("Loading image captioning model...")
                self.progress.start()
                
                # Load BLIP model for image captioning
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.captioning_model.to(device)
                
                self.update_status("Loading Stable Diffusion model...")
                
                # Load Stable Diffusion pipeline
                model_id = "runwayml/stable-diffusion-v1-5"
                self.generation_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.generation_pipeline.to(device)
                
                # Enable memory efficient attention if available
                if hasattr(self.generation_pipeline, 'enable_attention_slicing'):
                    self.generation_pipeline.enable_attention_slicing()
                
                self.models_loaded = True
                self.progress.stop()
                self.update_status("Models loaded! Ready to play!")
                
            except Exception as e:
                self.progress.stop()
                self.update_status(f"Error loading models: {str(e)}")
                messagebox.showerror("Error", f"Failed to load models: {str(e)}")
        
        # Start loading in a separate thread
        threading.Thread(target=load_models, daemon=True).start()
    
    def update_status(self, message):
        # Ensures thread safety
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                  width=3, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
    
    def canvas_to_image(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Bg
        image = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(image)
        
        # recreates items from canvas
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:
                draw.line(coords, fill="black", width=3)
        
        return image
    
    def interpret_and_generate(self):
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Models are still loading. Please wait.")
            return
        
        try:
            # Convert canvas to image
            drawing_image = self.canvas_to_image()
            
            if not drawing_image:
                messagebox.showwarning("Warning", "Please draw something first!")
                return
            
            self.update_status("Interpreting your drawing...")
            self.progress.start()
            
            # Run interpretation and generation in a separate thread
            threading.Thread(target=self._process_image, args=(drawing_image,), daemon=True).start()
            
        except Exception as e:
            self.progress.stop()
            self.update_status("Ready to play!")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    def _process_image(self, image):
        try:
            # caption generation
            inputs = self.captioning_processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.captioning_model.generate(**inputs, max_length=50)
            
            caption = self.captioning_processor.decode(out[0], skip_special_tokens=True)
            
            self.root.after(0, lambda: self.caption_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.caption_text.insert(tk.END, f"I see: {caption}\n\n"))
            
            self.update_status("Generating new image...")
            
            # new image generation
            prompt = f"A beautiful artistic illustration of {caption}"
            self.root.after(0, lambda: self.caption_text.insert(tk.END, f"Generating: {prompt}"))
            
            with torch.no_grad():
                generated = self.generation_pipeline(
                    prompt, 
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
            
            self.generated_image = generated
            self.display_generated_image(generated)
            
            self.progress.stop()
            self.update_status("Ready to play!")
            
        except Exception as e:
            self.progress.stop()
            self.update_status("Ready to play!")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing image: {str(e)}"))
    
    def display_generated_image(self, image):
        # Resize ti f
        display_image = image.copy()
        display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(display_image)
        self.generated_image_label.configure(image=photo, text="")
        self.generated_image_label.image = photo  # keep a reference
    
    def save_drawing(self):
        try:
            image = self.canvas_to_image()
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if filename:
                image.save(filename)
                messagebox.showinfo("Success", f"Drawing saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving drawing: {str(e)}")
    
    def save_generated_image(self):
        if self.generated_image is None:
            messagebox.showwarning("Warning", "No generated image to save!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if filename:
                self.generated_image.save(filename)
                messagebox.showinfo("Success", f"Generated image saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving generated image: {str(e)}")
    
    def use_generated_as_drawing(self):
        if self.generated_image is None:
            messagebox.showwarning("Warning", "No generated image to use!")
            return
        
        try:
            self.canvas.delete("all")
            
            line_drawing = self.image_to_line_drawing(self.generated_image)
            
            self.draw_image_on_canvas(line_drawing)
            
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, "Previous AI image converted to drawing!\nYou can now modify it and continue the game.")
            
            messagebox.showinfo("Success", "AI image converted to drawing! You can now modify it and continue the game.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error converting image to drawing: {str(e)}")
    
    def image_to_line_drawing(self, image):
        """Convert a color image to a line drawing using edge detection"""
        import cv2
        import numpy as np
        
        # image as np array
        img_array = np.array(image)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # dilate edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # convert back
        line_image = Image.fromarray(edges, mode='L')
        
        return line_image
    
    def draw_image_on_canvas(self, line_image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # def dimensions
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 400
        
        resized_image = line_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(resized_image)
        
        self.draw_pixels_as_dots(img_array)
    
    def draw_pixels_as_dots(self, img_array):
        """Draw edge pixels as small dots on canvas"""
        height, width = img_array.shape
        
        # sample every few for performance
        step = max(2, min(width, height) // 150) 
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                if img_array[y, x] > 128:  # if pixel bright (edge detected)
                    # draw a dot
                    self.canvas.create_oval(x-1, y-1, x+1, y+1, 
                                          fill="black", outline="black")
    
    def start_new_round(self):
        self.clear_canvas()
        self.caption_text.delete(1.0, tk.END)
        self.generated_image = None
        self.generated_image_label.configure(image="", text="Generated image will appear here - be patient!")
        self.update_status("Ready to draw!")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = AIPictographyGame()
    game.run()
