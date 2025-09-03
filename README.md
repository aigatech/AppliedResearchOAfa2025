# Personal Information Redaction Tool
## AI-powered privacy protection using HuggingFace NER models

## What It Does
A practical privacy tool that detects and redacts personal information in text using:
- **HuggingFace NER**: Uses `dslim/bert-base-NER` to detect names, organizations, and locations
- **Regex Patterns**: Detects emails, phone numbers, SSNs, credit cards, URLs, IP addresses
- **Multiple Mask Styles**: Choose from `[REDACTED]`, ★, █, •, #, X
- **Real-World Use Cases**: Medical records (HIPAA), employee data, customer feedback, research data, legal documents

**Detects**: Names, organizations, locations, emails, phones, SSNs, credit cards, URLs, IP addresses, dates, ZIP codes

## How to Run It

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage

**Demo with real-world scenarios:**
```bash
python privacy_redactor.py --demo
```

**Redact text:**
```bash
python privacy_redactor.py --text "Hi, I'm John Smith from Google. Email me at john@google.com or call (555) 123-4567."
```

**Process files:**
```bash
python privacy_redactor.py --file document.txt --output redacted_document.txt
```
Under 'sample_data' folder, there are some generated example files to try for yourself.

**Different mask styles:**
```bash
python privacy_redactor.py --text "Contact Sarah at sarah@company.com" --mask-style stars
python privacy_redactor.py --text "Contact Sarah at sarah@company.com" --mask-style blocks
```

**Batch process directory:**
```bash
python privacy_redactor.py --batch-dir sample_data --mask-style redacted
```

**Get detailed analysis:**
```bash
python privacy_redactor.py --text "My SSN is 123-45-6789" --format json --summary
```

### Example Output
```bash
Input: "Hi, my name is John Smith and I work at Google. Email me at john.smith@google.com or call (555) 123-4567."
```
```bash
Output: "Hi, my name is [REDACTED:PER] and I work at [REDACTED:ORG]. Email me at [REDACTED:EMAIL] or call [REDACTED:PHONE]."
```

### Example Real-World Use Cases
- **Medical Records**: HIPAA compliance for patient data
- **Employee Data**: HR privacy protection
- **Customer Feedback**: Data anonymization for analysis
- **Research Data**: Participant privacy protection
- **Legal Documents**: Case file protection