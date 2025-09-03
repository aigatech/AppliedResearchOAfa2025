# Code Comment Generator

## What it does

This is a small tool that scans a Python file, looks for functions that don’t have docstrings, and then adds simple ones automatically. It uses the Python `ast` module to parse code and a lightweight Hugging Face model (`distilgpt2`) to suggest short descriptions. If the model output isn’t usable, it falls back to a few simple rules based on the function name and arguments.

## Features

- Finds Python functions that don’t have docstrings
- Adds a short summary (model-assisted or heuristic fallback)
- Keeps existing comments/docstrings untouched
- Preserves indentation and layout
- Runs on CPU, no GPU needed

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python comment_generator.py input_file.py
```

Save to a specific output file:

```bash
python comment_generator.py input_file.py -o output_file.py
```

Overwrite the original file:

```bash
python comment_generator.py input_file.py --overwrite
```

## Example

Suppose you have this function in `test_functions.py`:

```python
def filter_active_users(users):
    active_users = []
    for user in users:
        if user.get('status') == 'active' and user.get('last_login'):
            active_users.append(user)
    return active_users
```

Running:

```bash
python comment_generator.py test_functions.py -o test_functions_commented.py
```

Produces:

```python
def filter_active_users(users):
    """Filters the list of users and returns only those who are active and have a last login."""
    active_users = []
    for user in users:
        if user.get('status') == 'active' and user.get('last_login'):
            active_users.append(user)
    return active_users
```

## Notes

This was built for the AI@GT Applied Research application by Zain. The goal was to show how code parsing and lightweight text generation can be combined into a simple, useful tool.

I have provided a test_functions.py file for testing the code commenter's performance but feel free to add your own code files as well!
