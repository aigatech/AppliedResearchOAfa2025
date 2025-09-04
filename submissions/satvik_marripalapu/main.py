from transformers import pipeline, AutoTokenizer
import textwrap
import argparse
import re

# summarize research papers!
class PaperSummarizer:
    
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            tokenizer="sshleifer/distilbart-cnn-12-6"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        print("success? yessir")
    
    def filter_text(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'©.*?\d{4}', '', text)
        text = re.sub(r'arXiv:\d+\.\d+', '', text)
        return text.strip()
    
    def chunk_text(self, text, max_chunk_length = 900):
        words = text.split()
        chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) * 1.3 > max_chunk_length:
                chunks.append(' '.join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_text(self, text, max_length = 150, min_length = 50):
        text = self.filter_text(text)
        
        if len(text.split()) < 50:
            return "Bro, you wanted a summary? It's not even 50 words just read it smh " + text
        
        chunks = self.chunk_text(text)
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Overanalyzing chunk {i+1} / {len(chunks)}...")
            
            try:
                summary = self.summarizer(
                    chunk,
                    max_length = max_length,
                    min_length = min_length,
                    do_sample = False,
                    truncation = True
                )
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Discombobulated by chunk {i+1}: {e}")
                continue
        
        if not summaries:
            return "Couldn't do this one gng."
        
        if len(summaries) > 1:
            combined = ' '.join(summaries)
            try:
                final_summary = self.summarizer(
                    combined,
                    max_length = max_length,
                    min_length = min_length,
                    do_sample = False,
                    truncation = True
                )
                return final_summary[0]['summary_text']
            except:
                return ' '.join(summaries)
        
        return summaries[0]
    
    def extract_key_points(self, text):
        keywords = re.findall(r'\b(?:method|approach|algorithm|model|result|finding|conclusion|significant|improve|novel|state-of-the-art|performance|accuracy|dataset)\w*\b', text.lower())
        return list(set(keywords))[:10]

def main():
    parser = argparse.ArgumentParser(description = "Summarize research papers")
    parser.add_argument("--file", "-f", help = "Path to text file containing paper content")
    parser.add_argument("--text", "-t", help = "Text to summarize directly")
    parser.add_argument("--max_length", type = int, default = 150, help = "Maximum summary length")
    parser.add_argument("--min_length", type = int, default = 50, help = "Minimum summary length")
    args = parser.parse_args()
    
    summarizer = PaperSummarizer()
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.text:
        input_text = args.text
    else:
        print("Welcome to my goated Research Paper Summarizer")
        print("Enter the research paper text (press Enter twice to finish):\n")
        
        lines = []
        empty_line_count = 0
        
        while True:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                lines.append(line)
        
        input_text = '\n'.join(lines)
    
    if not input_text.strip():
        print("You gotta give me text buddy.")
        return
    
    summary = summarizer.summarize_text(
        input_text, 
        max_length=args.max_length, 
        min_length=args.min_length
    )
    
    key_points = summarizer.extract_key_points(input_text)

    print("Here's your RESEARCH PAPER SUMMARY!!!")
    print(textwrap.fill(summary, width = 80))
    
    print("\n Plus some KEY THEMES WE IDENTIFIED!!!")
    if key_points:
        for point in key_points:
            print(f"• {point}")
    else:
        print("Nevermind, we didn't get any key themes this time.")

    print("I hope you're happy with the results! Go AI@GT and Go Jackets!")

if __name__ == "__main__":
    if len(__import__('sys').argv) == 1:
        print("Research Paper Summarizer")
        print("Usage examples: (also found in readme)")
        print("  python main.py --file paper.txt")
        print("  python main.py --text 'Your paper text here'")
        print("  python main.py  # Interactive mode")
        main()
    else:
        main()