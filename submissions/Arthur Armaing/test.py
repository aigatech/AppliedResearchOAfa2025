import os

file_path = "prompt.txt"  # Replace with the actual path to your TXT file

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)
