import argparse
import json
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from transformers import pipeline

class PrivacyRedactor:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        # init model
        print("Loading privacy redaction model...")
        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Could not load NER model: {e}")
            self.ner_pipeline = None
        
        # pattern detection for common personal info
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
        }
        
        # different mask options
        self.mask_styles = {
            'redacted': '[REDACTED:{type}]',
            'stars': '★' * 8,
            'blocks': '█' * 8,
            'dots': '•' * 8,
            'hash': '#' * 8,
            'x': 'X' * 8,
        }
    
    def detect_ner_entities(self, text: str) -> List[Dict]:
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            # Filter by confidence threshold
            filtered_entities = [
                entity for entity in entities 
                if entity['score'] >= self.confidence_threshold
            ]
            return filtered_entities
        except Exception as e:
            print(f"NER detection failed: {e}")
            return []
    
    def detect_regex_patterns(self, text: str) -> List[Dict]:
        detected = []
        
        for pattern_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected.append({
                    'entity_group': pattern_type.upper(),
                    'word': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'score': 1.0,  # regex match is 100% confidence
                    'source': 'regex'
                })
        
        return detected
    
    def merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        if not entities:
            return []
        
        entities.sort(key=lambda x: x['start'])
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            if next_entity['start'] < current['end']:
                if next_entity['score'] > current['score']:
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def redact_text(self, text: str, mask_style: str = 'redacted') -> Dict:
        """Redact personal information from text."""
        # Detect entities using NER
        ner_entities = self.detect_ner_entities(text)
        
        # Detect entities using regex
        regex_entities = self.detect_regex_patterns(text)
        
        # Combine and merge overlapping entities
        all_entities = ner_entities + regex_entities
        merged_entities = self.merge_overlapping_entities(all_entities)
        
        merged_entities.sort(key=lambda x: x['start'], reverse=True)
        
        # Apply redactions
        redacted_text = text
        redaction_log = []
        
        for entity in merged_entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['entity_group']
            original_text = entity['word']
            confidence = entity['score']
            
            if mask_style == 'redacted':
                mask = f"[REDACTED:{entity_type}]"
            else:
                mask = self.mask_styles.get(mask_style, self.mask_styles['redacted'])
            
            redacted_text = redacted_text[:start] + mask + redacted_text[end:]
            
            redaction_log.append({
                'type': entity_type,
                'original': original_text,
                'mask': mask,
                'confidence': confidence,
                'position': (start, end),
                'source': entity.get('source', 'ner')
            })
        
        return {
            'original_text': text,
            'redacted_text': redacted_text,
            'redactions': redaction_log,
            'total_redactions': len(redaction_log),
            'mask_style': mask_style,
            'confidence_threshold': self.confidence_threshold
        }
    
    def get_redaction_summary(self, result: Dict) -> str:
        """Generate a summary of redactions performed."""
        summary = f"Privacy Redaction Summary\n"
        summary += f"{'='*40}\n"
        summary += f"Total redactions: {result['total_redactions']}\n"
        summary += f"Mask style: {result['mask_style']}\n"
        summary += f"Confidence threshold: {result['confidence_threshold']}\n\n"
        
        if result['redactions']:
            type_counts = {}
            for redaction in result['redactions']:
                entity_type = redaction['type']
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            summary += "Redaction breakdown:\n"
            for entity_type, count in sorted(type_counts.items()):
                summary += f"  {entity_type}: {count}\n"
            
            summary += "\nDetailed redactions:\n"
            for i, redaction in enumerate(result['redactions'], 1):
                summary += f"  {i}. {redaction['type']}: '{redaction['original']}' → '{redaction['mask']}' (confidence: {redaction['confidence']:.2f})\n"
        else:
            summary += "No personal information detected.\n"
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Personal Information Redaction Tool")
    parser.add_argument("--text", type=str, help="Text to redact")
    parser.add_argument("--file", type=str, help="File to redact")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    parser.add_argument("--format", choices=['text', 'json'], default='text', help="Output format")
    parser.add_argument("--mask-style", choices=['redacted', 'stars', 'blocks', 'dots', 'hash', 'x'], 
                       default='redacted', help="Mask style for redacted text")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold for NER detection (0.0-1.0)")
    parser.add_argument("--summary", action="store_true", help="Show detailed redaction summary")
    parser.add_argument("--demo", action="store_true", help="Run demo with real-world scenarios")
    parser.add_argument("--batch-dir", type=str, help="Process all text files in a directory")
    
    args = parser.parse_args()
    
    # Initialize redactor
    redactor = PrivacyRedactor(confidence_threshold=args.confidence)
    
    def run_demo():
        """Run comprehensive demo with real-world scenarios."""
        print("Privacy Redaction Tool - Real-World Demo")
        print("=" * 60)
        print("This demo shows practical use cases for privacy protection")
        print("=" * 60)
        
        scenarios = [
            {
                "name": "Medical Records (HIPAA Compliance)",
                "text": "Patient: Sarah Johnson, DOB: 03/15/1985, SSN: 123-45-6789. Address: 456 Oak Street, Atlanta, GA 30309. Phone: (404) 555-0123. Email: sarah.johnson@email.com. Dr. Michael Chen from Emory University Hospital performed the examination.",
                "use_case": "Protecting patient information for HIPAA compliance"
            },
            {
                "name": "Employee Data (HR Privacy)",
                "text": "Employee: John Smith, Senior Software Engineer at TechCorp. Email: john.smith@techcorp.com, Phone: (555) 123-4567. Address: 123 Main Street, San Francisco, CA 94105. SSN: 987-65-4321. Manager: Jane Doe (jane.doe@techcorp.com).",
                "use_case": "Protecting employee personal information"
            },
            {
                "name": "Customer Feedback (Data Anonymization)",
                "text": "Customer: Jennifer Martinez, Email: jennifer.martinez@gmail.com, Phone: (305) 555-7890. Order #12345. 'I had an excellent experience! The representative Alex Thompson was very helpful. I live in Miami, FL. My credit card ending in 4532 was charged correctly.' Address: 567 Ocean Drive, Miami, FL 33139.",
                "use_case": "Anonymizing customer feedback for analysis"
            },
            {
                "name": "Research Data (Participant Privacy)",
                "text": "Participant: Emily Davis, Age: 22, University: Georgia Institute of Technology. Email: emily.davis@gatech.edu, Phone: (404) 555-9876. Address: 890 Tech Drive, Atlanta, GA 30332. Student ID: 901234567. 'I use social media platforms like Facebook and Instagram regularly.'",
                "use_case": "Protecting research participant information"
            },
            {
                "name": "Legal Documents (Case File Protection)",
                "text": "Case: Smith vs. TechCorp Inc. Plaintiff: Robert Smith, Address: 1234 Maple Street, Chicago, IL 60601, Phone: (312) 555-1234, Email: robert.smith@email.com, SSN: 111-22-3333. Defendant: TechCorp Inc., CEO: Jennifer Lee (jennifer.lee@techcorp.com). Attorney: Lisa Chen (lisa.chen@chenlaw.com).",
                "use_case": "Redacting sensitive information from legal documents"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print(f"Use Case: {scenario['use_case']}")
            print("-" * 50)
            print(f"Original: {scenario['text']}")
            
            result = redactor.redact_text(scenario['text'], args.mask_style)
            print(f"Redacted: {result['redacted_text']}")
            
            if args.summary:
                print(redactor.get_redaction_summary(result))
            else:
                print(f"Redactions: {result['total_redactions']}")
            print("=" * 60)
        
        print(f"\nDifferent Mask Styles Demo:")
        print("-" * 40)
        sample_text = "Contact John Smith at john.smith@company.com or call (555) 123-4567."
        
        for style in ['redacted', 'stars', 'blocks', 'dots']:
            result = redactor.redact_text(sample_text, style)
            print(f"{style.capitalize()}: {result['redacted_text']}")
        
        print(f"\nKey Benefits Demonstrated:")
        print("HIPAA compliance for medical data")
        print("Employee privacy protection") 
        print("Customer data anonymization")
        print("Research participant privacy")
        print("Legal document protection")
        print("Multiple redaction styles for different needs")
        print("Comprehensive PII detection (names, emails, phones, SSNs, etc.)")
        
        print(f"\n💡 Try these commands:")
        print("python privacy_redactor.py --text 'Your text here' --mask-style stars")
        print("python privacy_redactor.py --file your_document.txt --summary")
        print("python privacy_redactor.py --text 'John Smith at Google' --confidence 0.8")
    
    def batch_process_directory(directory: str):
        from pathlib import Path
        
        input_path = Path(directory)
        if not input_path.exists():
            print(f"Directory not found: {directory}")
            return
        
        # Find text files
        text_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))
        
        if not text_files:
            print(f"No text files found in {directory}")
            return
        
        print(f"Batch Processing Directory: {directory}")
        print(f"Files found: {len(text_files)}")
        print("=" * 50)
        
        processed = 0
        for file_path in text_files:
            print(f"\n📄 Processing: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                result = redactor.redact_text(content, args.mask_style)
                
                # Create output filename
                output_file = file_path.parent / f"redacted_{file_path.name}"
                
                if args.format == 'json':
                    output_content = json.dumps(result, indent=2, ensure_ascii=False)
                else:
                    output_content = result['redacted_text']
                    if args.summary:
                        output_content += "\n\n" + redactor.get_redaction_summary(result)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                
                print(f"Redacted: {output_file}")
                print(f"   Redactions: {result['total_redactions']}")
                processed += 1
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        
        print(f"\nBatch Processing Complete:")
        print(f"Successfully processed: {processed}/{len(text_files)}")
    
    if args.demo:
        run_demo()
        return
    
    if args.batch_dir:
        batch_process_directory(args.batch_dir)
        return
    
    # Get input text
    if args.text:
        input_text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except FileNotFoundError:
            print(f"File not found: {args.file}")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Please provide either --text or --file")
        parser.print_help()
        return
    
    # Perform redaction
    result = redactor.redact_text(input_text, args.mask_style)
    
    if args.format == 'json':
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = result['redacted_text']
        if args.summary:
            output += "\n\n" + redactor.get_redaction_summary(result)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Redacted text saved to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print(output)

if __name__ == "__main__":
    main()
