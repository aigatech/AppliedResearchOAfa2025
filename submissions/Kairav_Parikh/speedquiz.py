import json
import os
import PyPDF2
from transformers import pipeline
import random
import re

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def create_multiple_choice_question(text, question):
    try:
        answer_result = qa_pipeline(question=question, context=text)
        correct_answer = answer_result['answer']
        
        wrong_answers = generate_distractors(text, correct_answer)
        
        options = [correct_answer] + wrong_answers[:3]  
        random.shuffle(options)   
        
        correct_index = options.index(correct_answer)
        
        return {
            "question": question,
            "options": options,
            "correct_answer": correct_index,
            "correct_text": correct_answer
        }
    except:
        return None

def generate_distractors(text, correct_answer):
    words = text.split()
    distractors = []
    
    correct_words = correct_answer.split()
    
    for i, word in enumerate(words):
        if (word.lower() != correct_answer.lower() and 
            len(word) >= 3 and 
            word.isalpha()):
            
            if (word.istitle() == correct_answer.istitle() or
                word.isupper() == correct_answer.isupper()):
                distractors.append(word)
    
    if any(char.isdigit() for char in correct_answer):
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            if num != correct_answer and num not in distractors:
                distractors.append(num)
    
    generic_distractors = [
        "None of the above",
        "All of the above", 
        "Cannot be determined",
        "Not mentioned in the text"
    ]
    
    distractors = list(set(distractors))
    distractors = [d for d in distractors if d.lower() != correct_answer.lower()]
    
    while len(distractors) < 3:
        for generic in generic_distractors:
            if generic not in distractors and len(distractors) < 3:
                distractors.append(generic)
    
    return distractors[:3]

def generate_simple_questions(text, num_questions=3):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    questions = []
    
    for sentence in sentences[:num_questions*3]:  
        sentence = sentence.strip()
        
        if any(word in sentence.lower() for word in ['is', 'was', 'are', 'were']):
            words = sentence.split()
            for i, word in enumerate(words):
                if word.lower() in ['is', 'was'] and i > 0:
                    subject = ' '.join(words[:i]).strip()
                    if len(subject) > 2 and len(subject) < 50:
                        question = f"What {word.lower()} {subject}?"
                        questions.append(question)
                        break
        
        if len(questions) >= num_questions:
            break
    
    if len(questions) < num_questions:
        generic_questions = [
            "What is the main topic of this document?",
            "What information is provided in the text?",
            "What is discussed in this document?"
        ]
        for q in generic_questions:
            if len(questions) < num_questions:
                questions.append(q)
    
    return questions[:num_questions]

def create_quiz(text, num_questions=3):
    print("Creating questions...")
    questions = generate_simple_questions(text, num_questions)
    
    quiz = []
    print("Generating multiple choice options...")
    
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}")
        mc_question = create_multiple_choice_question(text, question)
        if mc_question:
            quiz.append(mc_question)
    
    return quiz

def run_quiz(quiz):
    score = 0
    total = len(quiz)
    
    print(f"\nMULTIPLE CHOICE QUIZ")
    print(f"Answer by typing A, B, C, or D\n")
    
    for i, q in enumerate(quiz, 1):
        print(f"Question {i}: {q['question']}")
        print()
        
        letters = ['A', 'B', 'C', 'D']
        for j, option in enumerate(q['options']):
            print(f"{letters[j]}. {option}")
        
        print()
        user_answer = input("Your answer (A/B/C/D): ").strip().upper()
        
        if user_answer in letters:
            answer_index = letters.index(user_answer)
            if answer_index == q['correct_answer']:
                print("Correct!")
                score += 1
            else:
                correct_letter = letters[q['correct_answer']]
                print(f"Wrong! The correct answer is {correct_letter}: {q['correct_text']}")
        else:
            print("Invalid answer! Please use A, B, C, or D")
            correct_letter = letters[q['correct_answer']]
            print(f"The correct answer is {correct_letter}: {q['correct_text']}")
        
        print("-" * 50)
    
    return score, total

def save_results(score, total, pdf_name):
    result = {
        "pdf": pdf_name,
        "score": score,
        "total": total,
        "percentage": round((score/total)*100, 1),
        "date": str(__import__('datetime').datetime.now())[:19]
    }
    
    history = []
    if os.path.exists("quiz_history.json"):
        try:
            with open("quiz_history.json", "r") as f:
                history = json.load(f)
        except:
            history = []
    
    history.append(result)
    
    with open("quiz_history.json", "w") as f:
        json.dump(history, f, indent=2)

def main():
    pdf_path = input("Enter PDF file path: ").strip()
    
    try:
        print("Reading PDF...")
        text = extract_text_from_pdf(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        
        if not text:
            print("Could not extract text from PDF!")
            return
        
        quiz = create_quiz(text, num_questions=3)
        
        if not quiz:
            print("Could not create quiz questions!")
            return
        
        score, total = run_quiz(quiz)
        
        percentage = round((score/total)*100, 1)
        print(f"\n🎯 FINAL SCORE: {score}/{total} ({percentage}%)")
        
        save_results(score, total, pdf_name)
        print("Results saved!")
        
    except FileNotFoundError:
        print("PDF file not found!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()