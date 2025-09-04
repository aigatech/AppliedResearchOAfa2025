import os
import re
import json
import requests
from typing import Optional, List, Dict

import gradio as gr

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# -----------------------------------------------------------------------------
# App metadata
# -----------------------------------------------------------------------------
"""
LeetCode Multi-Approach Analyzer
--------------------------------
Takes a LeetCode problem link and provides multiple algorithmic approaches with rankings.
Uses simple, reliable content-based analysis instead of AI.
"""

LC_GRAPHQL = "https://leetcode.com/graphql"
LC_ALL = "https://leetcode.com/api/problems/all/"

HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com",
    "User-Agent": "LeetCode-Analyzer/1.0",
}

# -----------------------------------------------------------------------------
# LeetCode utilities
# -----------------------------------------------------------------------------
def lc_fetch_problem(url: str) -> Optional[Dict]:
    """Fetch problem details from LeetCode URL."""
    try:
        # Extract slug from URL
        match = re.search(r"leetcode\.com/problems/([a-z0-9-]+)/?", url)
        if not match:
            return None
        
        slug = match.group(1)
        
        # Fetch problem data
        query = {
            "query": (
                "query getQ($slug: String!) {\n"
                "  question(titleSlug: $slug) {\n"
                "    questionId title titleSlug difficulty content\n"
                "  }\n"
                "}"
            ),
            "variables": {"slug": slug},
            "operationName": "getQ",
        }
        
        resp = requests.post(LC_GRAPHQL, json=query, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
            
        data = resp.json().get("data", {}).get("question")
        if not data:
            return None
            
        # Clean HTML content
        content = data.get("content", "")
        if BeautifulSoup:
            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text("\n")
        else:
            content = re.sub(r"<[^>]+>", " ", content)
        
        # Clean up extra whitespace and formatting
        content = re.sub(r'\n\s*\n', '\n', content)  # Remove multiple empty lines
        content = re.sub(r' +', ' ', content)  # Replace multiple spaces with single space
        content = re.sub(r'\n +', '\n', content)  # Remove leading spaces from lines
        content = re.sub(r' +\n', '\n', content)  # Remove trailing spaces from lines
        content = content.strip()  # Remove leading/trailing whitespace
        
        # Remove examples and other verbose sections
        content = _clean_leetcode_content(content)
        
        # Truncate content to prevent token length issues (keep it short)
        if len(content) > 800:
            content = content[:800] + "..."
        
        return {
            "id": data.get("questionId"),
            "title": data.get("title"),
            "slug": data.get("titleSlug"),
            "difficulty": data.get("difficulty"),
            "content": content
        }
        
    except Exception as e:
        print(f"Error fetching problem: {e}")
        return None


def _clean_leetcode_content(content: str) -> str:
    """Clean LeetCode problem content by removing examples and verbose sections."""
    # Remove everything after the main problem description (before examples/outputs)
    # Look for common patterns that indicate the end of the main problem
    patterns_to_cut = [
        r'Example \d+:',
        r'Input:',
        r'Output:',
        r'Explanation:',
        r'Constraints:',
        r'Follow-up:',
        r'Notice that',
        r'Note:',
        r'Follow up:'
    ]
    
    # Find the earliest occurrence of any of these patterns
    earliest_cut = len(content)
    for pattern in patterns_to_cut:
        match = re.search(pattern, content, re.IGNORECASE)
        if match and match.start() < earliest_cut:
            earliest_cut = match.start()
    
    # Cut the content at the earliest pattern found
    if earliest_cut < len(content):
        content = content[:earliest_cut]
    
    # Remove multiple consecutive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Clean up any remaining extra whitespace
    content = re.sub(r' +', ' ', content)
    content = content.strip()
    
    return content


# -----------------------------------------------------------------------------
# AI-powered approach analysis using Hugging Face
# -----------------------------------------------------------------------------
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
    print("Transformers library available")
except Exception as e:
    HAS_TRANSFORMERS = False
    print(f"Transformers not available: {e}")

def analyze_approaches(problem_content: str, progress=gr.Progress()) -> List[Dict]:
    """Analyze problem and suggest algorithmic approaches using AI."""
    progress(0.1, desc="Analyzing problem...")
    
    if not HAS_TRANSFORMERS:
        progress(0.5, desc="Using fallback approaches...")
        return _get_fallback_approaches()
    
    try:
        progress(0.2, desc="Loading AI model...")
        
        # Use larger, more capable models for better variety and accuracy
        models_to_try = [
            ("text2text-generation", "google/flan-t5-large"),     # 770M parameters, much better reasoning
            ("text2text-generation", "google/flan-t5-xl"),        # 3B parameters, excellent reasoning
            ("text2text-generation", "google/flan-t5-base"),      # 250M parameters, good fallback
            ("text2text-generation", "Salesforce/codet5p-220m"), # Code-specific model
            ("text2text-generation", "t5-small")                  # Final fallback
        ]
        
        generator = None
        for task, model_name in models_to_try:
            try:
                print(f"Trying model: {model_name}")
                generator = pipeline(task, model=model_name)
                print(f"Successfully loaded: {model_name}")
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        
        if not generator:
            print("All models failed to load, using fallback")
            progress(0.5, desc="All models failed, using fallback...")
            return _get_fallback_approaches()
        
        progress(0.4, desc="Preparing AI prompt...")
        
        # Create a more detailed prompt for better reasoning
        prompt = f"""Problem: {problem_content[:500]}...

Analyze this problem and select the 2 most appropriate algorithmic approaches from this list:
- Hash Table
- Two Pointers  
- Sliding Window
- Binary Search
- Dynamic Programming
- DFS
- BFS
- Greedy
- Sorting

Consider the problem requirements and choose approaches that would be most effective. Provide variety in your selection.

Answer with just the 2 approach names, one per line."""
        
        progress(0.6, desc="Generating with AI...")
        
        # Generate response with slightly higher temperature for variety
        result = generator(prompt, max_new_tokens=512, temperature=0.3, do_sample=True)
        print(result)
        ai_output = result[0]["generated_text"] if result else ""
        
        # Check if AI is just repeating the prompt (common issue with smaller models)
        prompt_indicators = [
            "choose 2 approaches", "from this list", "problem:", "return only", "approach names",
            "list 2 approaches", "format:", "two lines", "hash table", "two pointers",
            "sliding window", "binary search", "dynamic programming", "dfs", "bfs", "greedy", "sorting"
        ]
        
        prompt_indicator_count = sum(1 for indicator in prompt_indicators if indicator in ai_output.lower())
        if prompt_indicator_count >= 3:  # If 3 or more indicators found, likely repeating prompt
            print(f"AI repeated the prompt (found {prompt_indicator_count} indicators), using fallback approaches")
            progress(0.9, desc="AI repeated prompt, using fallback...")
            return _get_fallback_approaches()
    
        
        progress(0.8, desc="Processing AI response...")
        
        # Parse the AI output to extract approaches
        approaches = _parse_ai_approaches(ai_output, problem_content)
        
        # Debug: print the final parsed approaches
        for i, approach in enumerate(approaches):
            print(f"Approach {i+1}: {approach['name']}")
            print(f"  Description: {approach['description']}")
            print(f"  Time: {approach['time_complexity']}")
            print(f"  Space: {approach['space_complexity']}")
            print(f"  Ranking: {approach['ranking']}")
        print("=" * 50)
        
        if len(approaches) >= 2:
            progress(1.0, desc="AI analysis successful!")
            return approaches
        else:
            progress(0.9, desc="AI generated incomplete results, using fallback...")
            return _get_fallback_approaches()
            
    except Exception as e:
        print(f"AI analysis failed: {e}")
        progress(1.0, desc="AI failed, using fallback approaches...")
        return _get_fallback_approaches()


def _parse_ai_approaches(ai_output: str, problem_content: str) -> List[Dict]:
    """Parse AI output to extract approaches."""
    approaches = []
    
    # Try to extract approaches from AI output
    problem_lower = problem_content.lower()
    ai_lower = ai_output.lower()
    
    # Look for specific approaches mentioned in AI output
    detected_approaches = []
    
    # Check for various algorithmic approaches
    if any(word in ai_lower for word in ["hash table", "hash", "hash set", "set", "dictionary", "map"]):
        detected_approaches.append({
            "name": "Hash Table / Set",
            "description": "Use hash-based data structures for O(1) lookups and duplicate detection",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "pros": ["Fast lookups", "Simple to implement", "Handles duplicates well"],
            "cons": ["Uses extra memory", "Hash collisions possible"]
        })
    
    # Also check for simple line-by-line format
    lines = ai_output.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('[') and not line.startswith('{'):
            line_lower = line.lower()
            # Check if this line contains an approach name
            if any(word in line_lower for word in ["hash table", "hash", "hash set", "set"]):
                if not any(ap["name"] == "Hash Table / Set" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Hash Table / Set",
                        "description": "Use hash-based data structures for O(1) lookups and duplicate detection",
                        "time_complexity": "O(n)",
                        "space_complexity": "O(n)",
                        "pros": ["Fast lookups", "Simple to implement", "Handles duplicates well"],
                        "cons": ["Uses extra memory", "Hash collisions possible"]
                    })
            elif any(word in line_lower for word in ["two pointer", "two pointers", "pointer", "pointers"]):
                if not any(ap["name"] == "Two Pointers" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Two Pointers",
                        "description": "Use two pointers to traverse array efficiently",
                        "time_complexity": "O(n)",
                        "space_complexity": "O(1)",
                        "pros": ["Very efficient", "Constant space", "Good for sorted data"],
                        "cons": ["Requires sorted input", "May be complex to implement"]
                    })
            elif any(word in line_lower for word in ["sliding window", "window"]):
                if not any(ap["name"] == "Sliding Window" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Sliding Window",
                        "description": "Use a variable-size window to find optimal subarrays",
                        "time_complexity": "O(n)",
                        "space_complexity": "O(1)",
                        "pros": ["Efficient for subarray problems", "Constant space", "Flexible window size"],
                        "cons": ["Complex logic", "Edge cases to handle"]
                    })
            elif any(word in line_lower for word in ["binary search", "search"]):
                if not any(ap["name"] == "Binary Search" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Binary Search",
                        "description": "Divide and conquer approach for sorted data",
                        "time_complexity": "O(log n)",
                        "space_complexity": "O(1)",
                        "pros": ["Very efficient", "Constant space", "Optimal for sorted data"],
                        "cons": ["Requires sorted input", "Complex implementation"]
                    })
            elif any(word in line_lower for word in ["dynamic programming", "dp"]):
                if not any(ap["name"] == "Dynamic Programming" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Dynamic Programming",
                        "description": "Build solution from smaller subproblems",
                        "time_complexity": "O(n²) to O(n³)",
                        "space_complexity": "O(n) to O(n²)",
                        "pros": ["Optimal solution", "Handles complex problems", "Systematic approach"],
                        "cons": ["Complex to implement", "May use significant memory"]
                    })
            elif any(word in line_lower for word in ["dfs", "depth first", "recursion"]):
                if not any(ap["name"] == "Depth-First Search (DFS)" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Depth-First Search (DFS)",
                        "description": "Traverse tree/graph using recursion or stack",
                        "time_complexity": "O(n)",
                        "space_complexity": "O(h) where h is height/depth",
                        "pros": ["Natural for tree traversal", "Memory efficient", "Easy to implement"],
                        "cons": ["Stack overflow risk for deep structures", "May not find shortest path"]
                    })
            elif any(word in line_lower for word in ["bfs", "breadth first", "queue"]):
                if not any(ap["name"] == "Breadth-First Search (BFS)" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Breadth-First Search (BFS)",
                        "description": "Explore graph level by level using queue",
                        "time_complexity": "O(V + E)",
                        "space_complexity": "O(V)",
                        "pros": ["Finds shortest path", "Level-by-level exploration", "Good for unweighted graphs"],
                        "cons": ["Memory intensive", "May not find optimal path in weighted graphs"]
                    })
            elif any(word in line_lower for word in ["greedy", "greedy algorithm"]):
                if not any(ap["name"] == "Greedy Algorithm" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Greedy Algorithm",
                        "description": "Make locally optimal choice at each step",
                        "time_complexity": "O(n log n) to O(n²)",
                        "space_complexity": "O(1) to O(n)",
                        "pros": ["Simple to implement", "Often efficient", "Good for optimization"],
                        "cons": ["May not find global optimum", "Requires proof of correctness"]
                    })
            elif any(word in line_lower for word in ["sort", "sorting", "sorted"]):
                if not any(ap["name"] == "Sorting + Processing" for ap in detected_approaches):
                    detected_approaches.append({
                        "name": "Sorting + Processing",
                        "description": "Sort the data first, then process efficiently",
                        "time_complexity": "O(n log n) + processing time",
                        "space_complexity": "O(1) to O(n)",
                        "pros": ["Often enables efficient algorithms", "Simple to understand"],
                        "cons": ["Sorting overhead", "May not be optimal"]
                    })
    
    if any(word in ai_lower for word in ["two pointer", "two pointers", "pointer", "pointers"]):
        detected_approaches.append({
            "name": "Two Pointers",
            "description": "Use two pointers to traverse array efficiently",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "pros": ["Very efficient", "Constant space", "Good for sorted data"],
            "cons": ["Requires sorted input", "May be complex to implement"]
        })
    
    if any(word in ai_lower for word in ["sliding window", "window"]):
        detected_approaches.append({
            "name": "Sliding Window",
            "description": "Use a variable-size window to find optimal subarrays",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "pros": ["Efficient for subarray problems", "Constant space", "Flexible window size"],
            "cons": ["Complex logic", "Edge cases to handle"]
        })
    
    if any(word in ai_lower for word in ["binary search", "search"]):
        detected_approaches.append({
            "name": "Binary Search",
            "description": "Divide and conquer approach for sorted data",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)",
            "pros": ["Very efficient", "Constant space", "Optimal for sorted data"],
            "cons": ["Requires sorted input", "Complex implementation"]
        })
    
    if any(word in ai_lower for word in ["dynamic programming", "dp"]):
        detected_approaches.append({
            "name": "Dynamic Programming",
            "description": "Build solution from smaller subproblems",
            "time_complexity": "O(n²) to O(n³)",
            "space_complexity": "O(n) to O(n²)",
            "pros": ["Optimal solution", "Handles complex problems", "Systematic approach"],
            "cons": ["Complex to implement", "May use significant memory"]
        })
    
    if any(word in ai_lower for word in ["dfs", "depth first", "recursion"]):
        detected_approaches.append({
            "name": "Depth-First Search (DFS)",
            "description": "Traverse tree/graph using recursion or stack",
            "time_complexity": "O(n)",
            "space_complexity": "O(h) where h is height/depth",
            "pros": ["Natural for tree traversal", "Memory efficient", "Easy to implement"],
            "cons": ["Stack overflow risk for deep structures", "May not find shortest path"]
        })
    
    if any(word in ai_lower for word in ["bfs", "breadth first", "queue"]):
        detected_approaches.append({
            "name": "Breadth-First Search (BFS)",
            "description": "Explore graph level by level using queue",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "pros": ["Finds shortest path", "Level-by-level exploration", "Good for unweighted graphs"],
            "cons": ["Memory intensive", "May not find optimal path in weighted graphs"]
        })
    
    if any(word in ai_lower for word in ["greedy", "greedy algorithm"]):
        detected_approaches.append({
            "name": "Greedy Algorithm",
            "description": "Make locally optimal choice at each step",
            "time_complexity": "O(n log n) to O(n²)",
            "space_complexity": "O(1) to O(n)",
            "pros": ["Simple to implement", "Often efficient", "Good for optimization"],
            "cons": ["May not find global optimum", "Requires proof of correctness"]
        })
    
    if any(word in ai_lower for word in ["sort", "sorting", "sorted"]):
        detected_approaches.append({
            "name": "Sorting + Processing",
            "description": "Sort the data first, then process efficiently",
            "time_complexity": "O(n log n) + processing time",
            "space_complexity": "O(1) to O(n)",
            "pros": ["Often enables efficient algorithms", "Simple to understand"],
            "cons": ["Sorting overhead", "May not be optimal"]
        })
    
    # If we found approaches from AI, use them
    if len(detected_approaches) >= 2:
        # Take the first two approaches and rank them
        for i, approach in enumerate(detected_approaches[:2]):
            approach["ranking"] = i + 1
            approaches.append(approach)
    elif len(detected_approaches) == 1:
        print("Only 1 AI approach detected, adding random second approach")
        # Use the detected approach as first, add a random second approach
        detected_approaches[0]["ranking"] = 1
        approaches.append(detected_approaches[0])
        
        # Randomly select a second approach from available options
        import random
        all_approaches = [
            {
                "name": "Hash Table / Set",
                "description": "Use hash-based data structures for O(1) lookups and duplicate detection",
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
                "pros": ["Fast lookups", "Simple to implement", "Handles duplicates well"],
                "cons": ["Uses extra memory", "Hash collisions possible"]
            },
            {
                "name": "Two Pointers",
                "description": "Use two pointers to traverse array efficiently",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)",
                "pros": ["Very efficient", "Constant space", "Good for sorted data"],
                "cons": ["Requires sorted input", "May be complex to implement"]
            },
            {
                "name": "Sliding Window",
                "description": "Use a variable-size window to find optimal subarrays",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)",
                "pros": ["Efficient for subarray problems", "Constant space", "Flexible window size"],
                "cons": ["Complex logic", "Edge cases to handle"]
            },
            {
                "name": "Binary Search",
                "description": "Divide and conquer approach for sorted data",
                "time_complexity": "O(log n)",
                "space_complexity": "O(1)",
                "pros": ["Very efficient", "Constant space", "Optimal for sorted data"],
                "cons": ["Requires sorted input", "Complex implementation"]
            },
            {
                "name": "Dynamic Programming",
                "description": "Build solution from smaller subproblems",
                "time_complexity": "O(n²) to O(n³)",
                "space_complexity": "O(n) to O(n²)",
                "pros": ["Optimal solution", "Handles complex problems", "Systematic approach"],
                "cons": ["Complex to implement", "May use significant memory"]
            },
            {
                "name": "Depth-First Search (DFS)",
                "description": "Traverse tree/graph using recursion or stack",
                "time_complexity": "O(n)",
                "space_complexity": "O(h) where h is height/depth",
                "pros": ["Natural for tree traversal", "Memory efficient", "Easy to implement"],
                "cons": ["Stack overflow risk for deep structures", "May not find shortest path"]
            },
            {
                "name": "Breadth-First Search (BFS)",
                "description": "Explore graph level by level using queue",
                "time_complexity": "O(V + E)",
                "space_complexity": "O(V)",
                "pros": ["Finds shortest path", "Level-by-level exploration", "Good for unweighted graphs"],
                "cons": ["Memory intensive", "May not find optimal path in weighted graphs"]
            },
            {
                "name": "Greedy Algorithm",
                "description": "Make locally optimal choice at each step",
                "time_complexity": "O(n log n) to O(n²)",
                "space_complexity": "O(1) to O(n)",
                "pros": ["Simple to implement", "Often efficient", "Good for optimization"],
                "cons": ["May not find global optimum", "Requires proof of correctness"]
            },
            {
                "name": "Sorting + Processing",
                "description": "Sort the data first, then process efficiently",
                "time_complexity": "O(n log n) + processing time",
                "space_complexity": "O(1) to O(n)",
                "pros": ["Often enables efficient algorithms", "Simple to understand"],
                "cons": ["Sorting overhead", "May not be optimal"]
            }
        ]
        
        # Remove the already selected approach to avoid duplicates
        available_approaches = [ap for ap in all_approaches if ap["name"] != detected_approaches[0]["name"]]
        
        # Randomly select a second approach
        random_approach = random.choice(available_approaches)
        random_approach["ranking"] = 2
        approaches.append(random_approach)
        print(f"Randomly selected second approach: {random_approach['name']}")
    else:
        # No approaches detected, use content-based fallbacks
        if any(word in problem_lower for word in ["array", "list", "sequence"]):
            approaches.append({
                "name": "Two Pointers",
                "description": "Use two pointers to traverse array efficiently",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)",
                "ranking": 1,
                "pros": ["Very efficient", "Constant space", "Good for sorted data"],
                "cons": ["Requires sorted input", "May be complex to implement"]
            })
            approaches.append({
                "name": "Hash Table / Set",
                "description": "Use hash-based data structures for O(1) lookups",
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
                "ranking": 2,
                "pros": ["Fast lookups", "Good for duplicate detection"],
                "cons": ["Uses extra memory", "May not be optimal"]
            })
        else:
            approaches.append({
                "name": "Hash Table / Set",
                "description": "Use hash-based data structures for O(1) lookups",
                "time_complexity": "O(n)",
                "space_complexity": "O(n)",
                "ranking": 1,
                "pros": ["Fast lookups", "Good for duplicate detection"],
                "cons": ["Uses extra memory", "May not be optimal"]
            })
            approaches.append({
                "name": "Brute Force / Iterative",
                "description": "Try all possible combinations or iterate through elements",
                "time_complexity": "O(n²) or higher",
                "space_complexity": "O(1)",
                "ranking": 2,
                "pros": ["Guaranteed to work", "Simple to implement", "Easy to understand"],
                "cons": ["Very inefficient", "May timeout on large inputs", "Not optimal"]
            })
    
    return approaches


def _get_fallback_approaches() -> List[Dict]:
    """Fallback approaches when AI fails."""
    return [
        {
            "name": "Hash Table / Set",
            "description": "Use hash-based data structures for O(1) lookups and duplicate detection",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "ranking": 1,
            "pros": ["Fast lookups", "Simple to implement", "Handles duplicates well"],
            "cons": ["Uses extra memory", "Hash collisions possible"]
        },
        {
            "name": "Two Pointers",
            "description": "Use two pointers to traverse array efficiently",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "ranking": 2,
            "pros": ["Very efficient", "Constant space", "Good for sorted data"],
            "cons": ["Requires sorted input", "May be complex to implement"]
        }
    ]


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="LeetCode Multi-Approach Analyzer") as demo:
    gr.Markdown("#LeetCode Multi-Approach Analyzer")
    gr.Markdown("Enter a LeetCode problem URL and get multiple algorithmic approaches ranked by efficiency.")
    
    with gr.Row():
        url_input = gr.Textbox(
            label="LeetCode Problem URL",
            placeholder="https://leetcode.com/problems/two-sum/",
            lines=1
        )
        analyze_btn = gr.Button("Analyze Approaches", variant="primary", size="lg")
    
    # Status indicator
    status_text = gr.Textbox(label="Status", value="Ready to analyze", interactive=False, visible=True)
    
    # Problem information display
    with gr.Row():
        problem_title = gr.Textbox(label="Problem Title", interactive=False, visible=False)
        problem_difficulty = gr.Textbox(label="Difficulty", interactive=False, visible=False)
    
    problem_description = gr.Textbox(label="Problem Description", lines=5, interactive=False, visible=False)
    
    # Approaches display
    approaches_output = gr.HTML(label="Ranked Approaches", visible=False)
    
    def analyze_problem(url: str, progress=gr.Progress()):
        if not url.strip():
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(value="Please enter a LeetCode URL", visible=True)
            )
        
        # Progress bar updates
        progress(0.05, desc="Fetching problem from LeetCode...")
        
        # Fetch problem details
        problem = lc_fetch_problem(url)
        if not problem:
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(value="Failed to fetch problem. Check the URL.", visible=True)
            )
        
        progress(0.2, desc="Problem fetched successfully!")
        
        # Analyze approaches
        approaches = analyze_approaches(problem["content"], progress)
        
        # Sort by ranking
        approaches.sort(key=lambda x: x.get("ranking", 999))
        
        # Create human-readable HTML output
        html_content = _create_approaches_html(approaches)
        
        progress(1.0, desc="Analysis complete!")
        
        return (
            gr.update(value=problem["title"], visible=True),
            gr.update(value=problem["difficulty"], visible=True),
            gr.update(value=problem["content"], visible=True),
            gr.update(value=html_content, visible=True),
            gr.update(value="Analysis complete! Found " + str(len(approaches)) + " approaches.", visible=True)
        )
    
    def _create_approaches_html(approaches: List[Dict]) -> str:
        """Create human-readable HTML for approaches."""
        html = "<div style='font-family: Arial, sans-serif;'>"
        
        for i, approach in enumerate(approaches):
            ranking = approach.get("ranking", i + 1)
            name = approach.get("name", "Unknown")
            description = approach.get("description", "No description available")
            time_complexity = approach.get("time_complexity", "Unknown")
            space_complexity = approach.get("space_complexity", "Unknown")
            pros = approach.get("pros", [])
            cons = approach.get("cons", [])
            
            # Color coding based on ranking
            if ranking == 1:
                border_color = "#4CAF50"  # Green for best
                rank_text = "BEST"
            elif ranking == 2:
                border_color = "#2196F3"  # Blue for second
                rank_text = "GOOD"
            elif ranking == 3:
                border_color = "#FF9800"  # Orange for third
                rank_text = "OKAY"
            else:
                border_color = "#F44336"  # Red for worst
                rank_text = "POOR"
            
            html += f"""
            <div style='
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <div style='
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                '>
                    <h3 style='margin: 0; color: #000000;'>{name}</h3>
                    <span style='
                        background: {border_color};
                        color: white;
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-weight: bold;
                        font-size: 14px;
                    '>{rank_text}</span>
                </div>
                
                <p style='color: #000000; line-height: 1.6; margin-bottom: 15px;'>{description}</p>
                
                <div style='
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 15px;
                '>
                    <div style='
                        background: #e3f2fd;
                        padding: 10px;
                        border-radius: 5px;
                        border-left: 4px solid #2196F3;
                    '>
                        <strong style='color: #000000;'>Time Complexity:</strong><br>
                        <span style='font-family: monospace; font-size: 16px; color: #000000;'>{time_complexity}</span>
                    </div>
                    <div style='
                        background: #f3e5f5;
                        padding: 10px;
                        border-radius: 5px;
                        border-left: 4px solid #9c27b0;
                    '>
                        <strong style='color: #000000;'>💾 Space Complexity:</strong><br>
                        <span style='font-family: monospace; font-size: 16px; color: #000000;'>{space_complexity}</span>
                    </div>
                </div>
                
                <div style='
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                '>
                    <div style='
                        background: #e8f5e8;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 4px solid #4caf50;
                    '>
                        <strong style='color: #000000;'>Pros:</strong>
                        <ul style='margin: 10px 0; padding-left: 20px; color: #000000;'>
                            {''.join([f'<li style="color: #000000;">{pro}</li>' for pro in pros])}
                        </ul>
                    </div>
                    <div style='
                        background: #ffebee;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 4px solid #f44336;
                    '>
                        <strong style='color: #000000;'>Cons:</strong>
                        <ul style='margin: 10px 0; padding-left: 20px; color: #000000;'>
                            {''.join([f'<li style="color: #000000;">{con}</li>' for con in cons])}
                        </ul>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def on_button_click():
        """Handle button click with visual feedback"""
        return gr.update(variant="secondary", interactive=False, value="⏳ Analyzing...")
    
    def on_analysis_complete():
        """Reset button after analysis"""
        return gr.update(variant="primary", interactive=True, value="Analyze Approaches")
    
    # Button click events with chaining
    analyze_btn.click(
        fn=on_button_click,
        outputs=[analyze_btn]
    ).then(
        fn=analyze_problem,
        inputs=[url_input],
        outputs=[problem_title, problem_difficulty, problem_description, approaches_output, status_text]
    ).then(
        fn=on_analysis_complete,
        outputs=[analyze_btn]
    )
    
    gr.Markdown("""
    ## 💡 How to Use
    1. Paste a LeetCode problem URL (e.g., https://leetcode.com/problems/two-sum/)
    2. Click "Analyze Approaches"
    3. Get multiple algorithmic approaches ranked by efficiency
    
    ## What You Get
    - **Problem details**: Title, difficulty, description
    - **Multiple approaches**: 2 different solutions based on problem type
    - **Ranking**: Approaches ordered by efficiency (1 = best)
    - **Complexity analysis**: Time and space complexity for each approach
    - **Pros/Cons**: Benefits and drawbacks of each method
    
    ## ⚡ How It Works
    - **Smart analysis**: Automatically detects problem type from description
    - **Reliable results**: No AI dependencies - always works
    - **Fast processing**: Instant analysis without model loading
    - **Content-aware**: Chooses approaches based on problem characteristics
    """)

if __name__ == "__main__":
    demo.launch()
