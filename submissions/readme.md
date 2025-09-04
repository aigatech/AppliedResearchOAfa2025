# LeetCode Multi-Approach Analyzer

A simple, focused tool that takes a LeetCode problem URL and provides multiple algorithmic approaches ranked by efficiency.

##Features

- **LeetCode Integration**: Paste any LeetCode problem URL
- **Multiple Approaches**: Get 3-4 different algorithmic solutions
- **Smart Ranking**: Approaches automatically ranked by efficiency (1 = best)
- **Complexity Analysis**: Time and space complexity for each approach
- **Pros/Cons**: Benefits and drawbacks of each method
- **AI-Powered**: Uses lightweight local models for intelligent analysis

##  Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python appy_fixed.py
```

### How to Use
1. **Paste URL**: Copy a LeetCode problem URL (e.g., `https://leetcode.com/problems/two-sum/`)
2. **Click Analyze**: Press "Analyze Approaches"
3. **Get Results**: View ranked approaches with complexity analysis

## What You Get

### Problem Information
- **Title**: Problem name
- **Difficulty**: Easy/Medium/Hard
- **Description**: Clean problem text
- **ID**: LeetCode problem number

### Ranked Approaches
Each approach includes:
- **Name**: Algorithmic technique (e.g., "Two Pointers", "Hash Table")
- **Description**: How the approach works
- **Time Complexity**: Big-O notation (e.g., O(n), O(n²))
- **Space Complexity**: Memory usage
- **Ranking**: Efficiency score (1 = best, 4 = least efficient)
- **Pros**: Benefits of this approach
- **Cons**: Drawbacks and limitations

## Technical Details

- **Model**: Uses lightweight `codet5p-220m` (220M parameters)
- **Fallback**: Smart fallback approaches if AI analysis fails
- **Fast**: Quick analysis with minimal resource usage
- **Reliable**: Works even when models struggle with complex outputs

## Example Output

```json
{
  "approaches": [
    {
      "name": "Two Pointers",
      "description": "Use two pointers to traverse array efficiently",
      "time_complexity": "O(n)",
      "space_complexity": "O(1)",
      "ranking": 1,
      "pros": ["Very efficient", "Constant space"],
      "cons": ["Requires sorted input", "Complex implementation"]
    },
    {
      "name": "Hash Table",
      "description": "Use hash set for O(1) lookups",
      "time_complexity": "O(n)",
      "space_complexity": "O(n)",
      "ranking": 2,
      "pros": ["Fast lookups", "Simple logic"],
      "cons": ["Uses extra memory", "Hash collisions"]
    }
  ]
}
```

##  Dependencies

- **gradio**: Modern web interface
- **requests**: HTTP requests to LeetCode API
- **beautifulsoup4**: HTML parsing for problem content
- **transformers**: Hugging Face models for AI analysis
- **torch**: PyTorch backend for models

## Use Cases

- **Interview Prep**: Quickly understand multiple approaches to problems
- **Learning**: See how different algorithms solve the same problem
- **Comparison**: Understand trade-offs between approaches
- **Reference**: Get complexity analysis for common patterns

## Why This Project?

- **Simple**: One input (URL), one output (ranked approaches)
- **Focused**: Does one thing really well
- **Fast**: Lightweight models for quick analysis
- **Reliable**: Smart fallbacks ensure it always works
- **Educational**: Helps understand algorithmic thinking

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
