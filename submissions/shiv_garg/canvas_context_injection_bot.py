import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 8192

def fetch_canvas_data(canvas_base, token):
    headers = {"Authorization": f"Bearer {token}"}
    courses_url = f"{canvas_base}/api/v1/courses"
    courses = requests.get(courses_url, headers=headers).json()

    info_text = "Here is course and assignment information for a student:\n\n"

    for course in courses:
        if "name" not in course:
            continue
        course_name = course["name"]
        info_text += f"Course: {course_name}\n"

        assignments_url = f"{canvas_base}/api/v1/courses/{course['id']}/assignments?per_page=100"
        assignments = requests.get(assignments_url, headers=headers).json()

        if isinstance(assignments, list):
            for a in assignments:
                due = a.get("due_at", "No due date")
                submitted = "submitted" if a.get("has_submitted_submissions") else "not submitted"
                info_text += f"  - {a['name']} (Due: {due}, Status: {submitted})\n"
        info_text += "\n"
    return info_text

def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    gen = pipeline("text-generation", model=model, tokenizer=tok)
    return gen

def main():
    canvas_base = "https://gatech.instructure.com"
    token = input("Enter your Canvas access token: ").strip()

    print("\nFetching data from Canvas...")
    canvas_info = fetch_canvas_data(canvas_base, token)

    question = input("\nEnter your question: ").strip()

    prompt = f"{canvas_info}\nStudent's question: {question}\nAnswer this question based on the information provided above and your general knowledge."

    generator = load_llm()
    response = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, truncation=True)

    print("\n--- Answer ---")
    print(response[0]['generated_text'])

if __name__ == "__main__":
    main()
