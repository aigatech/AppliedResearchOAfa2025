import wikipediaapi
from transformers import pipeline
from typing import List, Dict


#constants
WIKI_CHAR_LIMIT = 1000 # limitng to avoid overflow
SUM_MODEL = "sshleifer/distilbart-cnn-12-6" # heard it was decent + lightweight
SUM_MAX_LEN = 200
SUM_MIN_LEN = 50
QG_MODEL  = "iarfmoose/t5-base-question-generator" # model for question generation
QG_NUM_QUESTIONS = 1   # asking a reading check question
USER_AGENT = "NannanAravazhi-AIatGT-AR-Fall2025/1.0 (contact: naravazhi3@gatech.edu)"



# intiializing pipeline here to use huggingface locally
summarizer_pipeline = pipeline("summarization", model=SUM_MODEL)
qg_pipeline = pipeline("text2text-generation", model=QG_MODEL)




# core functions
def fetch_wikipedia_content(topic: str) -> str:
   """
   fetching text, limit to 1000, basic error handling
   """
   wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent=USER_AGENT  # setting user agent
    )
   page = wiki.page(topic)
   if not page.exists():
       raise ValueError(f"Can't find this topic: '{topic}'.")
  
   text = (page.summary or "").strip()
   if not text:
        text = (page.text or "").strip()
   if not text:
        raise ValueError(f"No readable content on this topic: '{topic}'.")
   return text[:WIKI_CHAR_LIMIT]


def summarize_text(content: str) -> str:
   """
   creating short summary here
   """
   if not content or not content.strip():
       raise ValueError("Empty content; cannot summarize.")
   result = summarizer_pipeline(content, max_length=SUM_MAX_LEN, min_length=SUM_MIN_LEN, do_sample=False)
   return result[0]["summary_text"].strip()


def generate_questions(summary: str) -> List[str]:
   """
   quiz-style questions
   """
   if not summary or not summary.strip():
        return []
   
   prompt = f"Generate {QG_NUM_QUESTIONS} unique, short quiz-style questions based on this summary:\n{summary}"

   out = qg_pipeline(
       prompt,
       max_length=64,
       do_sample=True,           #sampling
       top_p=0.92,
       top_k=50,
       temperature=0.9,
       num_return_sequences=QG_NUM_QUESTIONS,
       num_beams=QG_NUM_QUESTIONS,
    )
   qs = [it["generated_text"].strip() for it in out if it["generated_text"].strip()]
   seen, deduped = set(), []
   for q in qs:
      if q not in seen:
          seen.add(q)
          deduped.append(q)
          if len(deduped) >= QG_NUM_QUESTIONS:
              break
   return deduped




# mani method
def main() -> None:
   print("=== AI@GT: Wikipedia Summarizer and QGen ===")
   topic = input("Enter a Wikipedia topic: ").strip() or "Georgia Institute of Technology"
   try:
       content = fetch_wikipedia_content(topic)
       summary = summarize_text(content)
       questions = generate_questions(summary)


       # hashmap to hold
       result: Dict[str, object] = {
           "topic": topic,
           "summary": summary,
           "questions": questions
       }


       print("\n=== Summary ===\n" + result["summary"])
       print("\n=== Questions ===")
       if questions:
           for i, q in enumerate(questions, 1):
               print(f"{i}. {q}")
       else:
           print("(No questions generated)")


   except Exception as e:
       print(f"\n[Error] {e}")


if __name__ == "__main__":
   main()
