import tkinter as tk
from tkinter import scrolledtext, ttk
import requests
import json
import threading
import time

class SimpleSentimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Sentiment Analyzer")
        self.root.geometry("500x400")
        
        self.api_url = "https://api.tabularis.ai/"
        self.headers = {"Content-Type": "application/json"}
        
        self.setup_ui()
        
    def setup_ui(self):
        title = tk.Label(self.root, text="Simple Sentiment Analyzer", font=("Arial", 16, "bold"))
        title.pack(pady=20)
        
        input_label = tk.Label(self.root, text="Enter text to analyze:")
        input_label.pack()
        
        self.text_input = tk.Text(self.root, height=3, width=50)
        self.text_input.pack(pady=10)
        self.text_input.insert("1.0", "I love this amazing product!")
        
        self.analyze_btn = tk.Button(
            self.root, 
            text="Analyze", 
            command=self.start_analysis,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            relief='flat',
            padx=25,
            pady=8
        )
        self.analyze_btn.pack(pady=10)
        
        self.status_label = tk.Label(self.root, text="Ready", fg='green')
        self.status_label.pack()
        
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0', relief='groove', bd=2)
        self.results_frame.pack(pady=(20,5), padx=20, fill='both', expand=True)
        
        self.results_title = tk.Label(self.results_frame, text="Results will appear here", font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50')
        self.results_title.pack(pady=20)
        
        self.sentiment_labels = {}
        self.sentiment_bars = {}
        self.sentiment_percentages = {}
        
        self.model_info_frame = tk.Frame(self.results_frame, bg='#f0f0f0')
        self.model_info_frame.pack(pady=10, fill='x')
        
        self.model_label = tk.Label(self.model_info_frame, text="", bg='#f0f0f0', font=("Arial", 9), fg='#34495e')
        self.model_label.pack()
        
        self.time_label = tk.Label(self.model_info_frame, text="", bg='#f0f0f0', font=("Arial", 9), fg='#34495e')
        self.time_label.pack()
        
    def start_analysis(self):
        self.analyze_btn.config(state='disabled', text="Analyzing...")
        self.status_label.config(text="Making API request...", fg='orange')
        
        self.clear_results()
        
        self.results_title.config(text="Analyzing... Please wait...")
        
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", "Please enter some text!")
            self.reset_ui()
            return
        
        thread = threading.Thread(target=self.make_api_call, args=(text,))
        thread.daemon = True
        thread.start()
    
    def make_api_call(self, text):
        try:
            payload = {
                "text": text,
                "return_all_scores": True
            }
            
            self.root.after(0, lambda: self.status_label.config(text="API request in progress...", fg='orange'))
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            self.root.after(0, lambda: self.handle_response(response, text))
            
        except Exception as e:
            self.root.after(0, lambda: self.handle_error(str(e)))
    
    def handle_response(self, response, text):
        if response.status_code == 200:
            try:
                result = response.json()
                self.status_label.config(text="Success!", fg='green')
                
                self.results_title.config(text="Sentiment Analysis Results")
                
                text_label = tk.Label(self.results_frame, text=f"Text: {text}", font=("Arial", 10), bg='#f0f0f0', fg='#2c3e50', wraplength=400)
                text_label.pack(pady=(10,20))
                
                if 'output' in result and 'results' in result['output']:
                    results = result['output']['results']
                    
                    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
                    
                    results_title = tk.Label(self.results_frame, text="Sentiment Analysis Results", font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50')
                    results_title.pack(pady=(0,15))
                    
                    for i, item in enumerate(sorted_results):
                        label = item['label']
                        score = item['score']
                        percentage = score * 100
                        
                        sentiment_frame = tk.Frame(self.results_frame, bg='#f0f0f0')
                        sentiment_frame.pack(pady=5, fill='x', padx=20)
                        
                        label_text = label
                        if i == 0:
                            label_text += " (Highest)"
                        
                        sentiment_label = tk.Label(sentiment_frame, text=label_text, font=("Arial", 11, "bold"), 
                                                 bg='#f0f0f0', fg=self.get_sentiment_color(label))
                        sentiment_label.pack(anchor='w')
                        
                        bar_frame = tk.Frame(sentiment_frame, bg='#ecf0f1', height=20, relief='sunken', bd=1)
                        bar_frame.pack(fill='x', pady=(5,0))
                        bar_frame.pack_propagate(False)
                        
                        bar_width = int((percentage / 100) * 300)
                        progress_bar = tk.Frame(bar_frame, bg=self.get_sentiment_color(label), width=bar_width, height=18)
                        progress_bar.pack(side='left', fill='y', padx=1, pady=1)
                        
                        percentage_label = tk.Label(sentiment_frame, text=f"{percentage:.1f}%", font=("Arial", 10), 
                                                  bg='#f0f0f0', fg=self.get_sentiment_color(label))
                        percentage_label.pack(anchor='e')
                    
                    if 'model' in result['output']:
                        self.model_label.config(text=f"Model: {result['output']['model']}")
                    
                    if 'inference_time' in result['output']:
                        self.time_label.config(text=f"Inference Time: {result['output']['inference_time']} | Total: {result.get('executionTime', 'N/A')}ms")
                        
                else:
                    error_label = tk.Label(self.results_frame, text="No sentiment results found in the response.", 
                                         font=("Arial", 10), bg='#f0f0f0', fg='#e74c3c')
                    error_label.pack(pady=20)
                
            except json.JSONDecodeError:
                error_label = tk.Label(self.results_frame, text=f"Response received but couldn't parse JSON:\n{response.text}", 
                                     font=("Arial", 10), bg='#f0f0f0', fg='#e74c3c')
                error_label.pack(pady=20)
        else:
            self.status_label.config(text=f"API Error: {response.status_code}", fg='red')
            error_label = tk.Label(self.results_frame, text=f"Error {response.status_code}:\n{response.text}", 
                                 font=("Arial", 10), bg='#f0f0f0', fg='#e74c3c')
            error_label.pack(pady=20)
        
        self.reset_ui()
    
    def handle_error(self, error_msg):
        self.status_label.config(text=f"Error: {error_msg}", fg='red')
        
        self.results_title.config(text="Error Occurred")
        
        error_label = tk.Label(self.results_frame, text=f"Exception occurred:\n{error_msg}", 
                             font=("Arial", 10), bg='#f0f0f0', fg='#e74c3c')
        error_label.pack(pady=20)
        
        self.reset_ui()
    
    def reset_ui(self):
        self.analyze_btn.config(state='normal', text="Analyze")
    
    def clear_results(self):
        for widget in self.results_frame.winfo_children():
            if widget != self.results_title and widget != self.model_info_frame:
                widget.destroy()
        
        self.model_label.config(text="")
        self.time_label.config(text="")
        
        self.results_title.config(text="Results will appear here")
    
    def get_sentiment_color(self, label):
        color_map = {
            "Very Positive": "#27ae60",
            "Positive": "#2ecc71",
            "Neutral": "#95a5a6",
            "Negative": "#e74c3c",
            "Very Negative": "#c0392b"
        }
        return color_map.get(label, "#000000")

def main():
    root = tk.Tk()
    app = SimpleSentimentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
