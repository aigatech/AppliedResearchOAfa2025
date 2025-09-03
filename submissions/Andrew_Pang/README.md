# ResumeAgent: AI-Powered Resume Matcher  

ResumeAgent is an AI tool that compares resumes to job descriptions and generates insights such as:  
- **Relevance Score** – how well the resume matches the job  
- **Matching Skills** – skills/keywords found in both documents  
- **Missing Skills** – skills the resume lacks compared to the job description  
- **Named Entity Recognition (NER)** – highlights skills, organizations, and experiences extracted from resumes  
- **Question Answering** - answers user questions about the job description

Built with HuggingFace, PyTorch, spaCy, and SentenceTransformers.  

---

## Usage
- change resume_path, job_path, and question in main.py to get different responses
- try different job resumes (included in docs) for different results

---

## Installation  

Clone the repo and install dependencies:  

```bash
pip install -r requirements.txt
python main.py