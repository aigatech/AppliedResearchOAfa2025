#!/usr/bin/env python3
"""
Code Comment Generator
Generates succinct docstrings for Python functions detected via AST.
"""

import re
import ast
import argparse
from typing import List, Dict, Tuple
from transformers import pipeline

class CodeCommentGenerator:
    def __init__(self):
        # load model or fallback
        print("Loading language model...")
        try:
            model_name = "distilgpt2"
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=25,
                do_sample=True,
                temperature=0.6,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            print("Model loaded.")
        except Exception as e:
            print(f"Model failed to load ({e}). Falling back to heuristic mode.")
            self.generator = None
    
    def extract_functions(self, code: str) -> List[Dict]:
        functions = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno - 1
                    func_lines = code.split('\n')[func_start:]
                    
                    func_code_lines = []
                    indent_level = None
                    
                    for i, line in enumerate(func_lines):
                        if i == 0:
                            func_code_lines.append(line)
                            stripped = line.lstrip()
                            if stripped:
                                indent_level = len(line) - len(stripped)
                        else:
                            if line.strip() == "":
                                func_code_lines.append(line)
                                continue
                            current_indent = len(line) - len(line.lstrip())
                            if line.strip() and current_indent <= indent_level:
                                break
                            func_code_lines.append(line)
                    
                    func_code = '\n'.join(func_code_lines)
                    has_docstring = self._has_docstring(node)
                    
                    functions.append({
                        'name': node.name,
                        'code': func_code,
                        'line_start': func_start,
                        'args': [arg.arg for arg in node.args.args],
                        'has_docstring': has_docstring
                    })
        except SyntaxError as e:
            print(f"Warning: could not parse file: {e}")
        
        return functions
    
    def _has_docstring(self, func_node) -> bool:
        if (
            func_node.body and 
            isinstance(func_node.body[0], ast.Expr) and 
            isinstance(func_node.body[0].value, ast.Constant) and 
            isinstance(func_node.body[0].value.value, str)
        ):
            return True
        return False
    
    def generate_comment(self, func_info: Dict) -> str:
        if self.generator is not None:
            try:
                return self._generate_with_model(func_info)
            except Exception:
                pass
        return self._generate_intelligent_fallback(func_info)
    
    def _generate_with_model(self, func_info: Dict) -> str:
        prompt = self._create_prompt(func_info)
        result = self.generator(
            prompt,
            max_new_tokens=20,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.6,
            pad_token_id=50256  # GPT-2 EOS
        )
        generated_text = result[0]['generated_text'].strip()
        comment = self._clean_and_format_comment(generated_text, func_info)
        return comment
    
    def _create_prompt(self, func_info: Dict) -> str:
        func_name = func_info['name']
        if func_name.startswith('calculate_'):
            action = "calculates"; subject = func_name[10:]
        elif func_name.startswith('process_'):
            action = "processes"; subject = func_name[8:]
        elif func_name.startswith('get_'):
            action = "gets"; subject = func_name[4:]
        elif func_name.startswith('transform_'):
            action = "transforms"; subject = func_name[10:]
        elif func_name.startswith(('is_', 'has_')):
            action = "checks if"; subject = func_name[3:] if func_name.startswith('is_') else func_name[4:]
        else:
            action = "handles"; subject = func_name.replace('_', ' ')
        return f"Python docstring: This function {action} {subject.replace('_', ' ')}"
    
    def _clean_and_format_comment(self, generated_text: str, func_info: Dict) -> str:
        comment = generated_text.strip()
        comment = re.sub(r'\n+', ' ', comment)
        comment = re.sub(r'\s+', ' ', comment)
        comment = comment.replace('"""', '').replace("'''", '')
        
        sentences = comment.split('.')
        if len(sentences) > 0 and len(sentences[0]) > 10:
            comment = sentences[0].strip()
        elif len(comment) > 80:
            comment = comment[:80]
        
        if comment and not comment.endswith('.'):
            comment += '.'
        if comment:
            comment = comment[0].upper() + comment[1:]
        
        keywords = ['function', 'calculates', 'processes', 'returns', 'gets', 'handles', 'checks', 'transforms']
        if len(comment) < 15 or not any(w in comment.lower() for w in keywords):
            return self._generate_intelligent_fallback(func_info)
        
        comment = f'"""{comment}"""'
        base_indent = self._get_function_indent(func_info['code'])
        return self._indent_comment(comment, base_indent + "    ")
    
    def _generate_intelligent_fallback(self, func_info: Dict) -> str:
        func_name = func_info['name']
        args = func_info['args']
        code = func_info['code']
        
        if func_name.startswith('calculate_'):
            summary = f"Calculates {func_name[10:].replace('_', ' ')}"
        elif func_name.startswith('process_'):
            summary = f"Processes {func_name[8:].replace('_', ' ')}"
        elif func_name.startswith('get_'):
            summary = f"Retrieves {func_name[4:].replace('_', ' ')}"
        elif func_name.startswith('transform_'):
            summary = f"Transforms {func_name[10:].replace('_', ' ')}"
        elif func_name.startswith(('is_', 'has_')):
            summary = f"Checks if {func_name.split('_', 1)[1].replace('_', ' ')}"
        elif func_name == '__init__':
            summary = "Initialize the class instance"
        else:
            summary = f"Handles {func_name.replace('_', ' ')}"
        
        non_self_args = [arg for arg in args if arg != 'self']
        if non_self_args:
            if len(non_self_args) == 1:
                summary += f" for the given {non_self_args[0]}"
            elif len(non_self_args) == 2:
                summary += f" using {non_self_args[0]} and {non_self_args[1]}"
            else:
                summary += f" with {len(non_self_args)} parameters"
        
        if 'return True' in code or 'return False' in code:
            summary += " and returns a boolean result"
        elif 'return []' in code or 'return list' in code:
            summary += " and returns a list"
        elif code.count('return') > 1:
            summary += " with conditional returns"
        
        summary += "."
        doc = f'"""{summary}"""'
        base_indent = self._get_function_indent(func_info['code'])
        return self._indent_comment(doc, base_indent + "    ")
    
    def _get_function_indent(self, func_code: str) -> str:
        first_line = func_code.split('\n')[0]
        return first_line[:len(first_line) - len(first_line.lstrip())]
    
    def _indent_comment(self, comment: str, indent: str) -> str:
        lines = comment.split('\n')
        return '\n'.join([indent + line if line.strip() else line for line in lines])
    
    def add_comments_to_code(self, code: str) -> str:
        functions = self.extract_functions(code)
        if not functions:
            print("No functions found.")
            return code
        
        print(f"Processing {len(functions)} function(s)...")
        lines = code.split('\n')
        
        for func_info in reversed(functions):
            if func_info['has_docstring']:
                continue
            comment = self.generate_comment(func_info)
            insert_line = func_info['line_start'] + 1
            lines.insert(insert_line, comment)
        
        return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description="Insert concise docstrings into Python code.")
    parser.add_argument("input_file", help="Python file to process")
    parser.add_argument("-o", "--output", help="Output file (default: adds '_commented' suffix)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the input file")
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.input_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    generator = CodeCommentGenerator()
    commented_code = generator.add_comments_to_code(original_code)
    
    if args.overwrite:
        output_file = args.input_file
    elif args.output:
        output_file = args.output
    else:
        base_name = args.input_file.rsplit('.', 1)[0]
        output_file = f"{base_name}_commented.py"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(commented_code)
        print(f"Saved: {output_file}")
        if generator.generator:
            print("Mode: model-assisted")
        else:
            print("Mode: heuristic")
    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    main()