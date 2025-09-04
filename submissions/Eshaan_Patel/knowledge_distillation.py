import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeDistillationDataset(Dataset):
    def __init__(self, prompts: List[str], tokenizer, max_length: int = 128):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt
        }

class KnowledgeDistiller:
    def __init__(
        self, 
        teacher_model_name: str = "microsoft/DialoGPT-medium",
        student_model_name: str = "distilgpt2",
        device: str = "cpu"
    ):
        self.device = device
        self.temperature = 4.0
        self.alpha = 0.7
        
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading teacher model: {teacher_model_name}")
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name)
        self.teacher.to(self.device)
        self.teacher.eval()
        
        logger.info(f"Loading student model: {student_model_name}")
        self.student = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.student.to(self.device)
        
        if self.teacher.config.vocab_size != self.student.config.vocab_size:
            self.student.resize_token_embeddings(len(self.tokenizer))

    def get_teacher_predictions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        return soft_targets
    
    def distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_soft_targets: torch.Tensor, 
        true_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:

        student_soft_preds = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_soft_preds, 
            teacher_soft_targets, 
            reduction='none'
        ).sum(dim=-1)
        
        kl_loss = (kl_loss * attention_mask).sum() / attention_mask.sum()
        kl_loss *= (self.temperature ** 2)

        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = true_labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        ce_loss = ce_loss.view(shift_labels.size())
        ce_loss = (ce_loss * shift_mask).sum() / shift_mask.sum()
        
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss
    
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float, float]:
        self.student.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        teacher_soft_targets = self.get_teacher_predictions(input_ids, attention_mask)
        
        student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        
        total_loss, kl_loss, ce_loss = self.distillation_loss(
            student_logits, teacher_soft_targets, input_ids, attention_mask
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return total_loss.item(), kl_loss.item(), ce_loss.item()
    
    def train(
        self, 
        train_prompts: List[str], 
        num_epochs: int = 3, 
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100
    ):
        
        dataset = DistillationDataset(train_prompts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_kl_loss = 0
            total_ce_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                loss, kl_loss, ce_loss = self.train_step(batch, optimizer)
                scheduler.step()
                
                total_loss += loss
                total_kl_loss += kl_loss
                total_ce_loss += ce_loss
                
                progress_bar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'KL': f'{kl_loss:.4f}',
                    'CE': f'{ce_loss:.4f}'
                })
            
            avg_loss = total_loss / len(dataloader)
            avg_kl_loss = total_kl_loss / len(dataloader)
            avg_ce_loss = total_ce_loss / len(dataloader)
            
            logger.info(
                f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, "
                f"KL Loss: {avg_kl_loss:.4f}, CE Loss: {avg_ce_loss:.4f}"
            )
    
    def generate_text(self, prompt: str, max_length: int = 100, num_beams: int = 4) -> str:
        self.student.eval()
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.student.generate(
                inputs,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def compare_outputs(self, prompt: str) -> Dict[str, str]:
        teacher_inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            teacher_outputs = self.teacher.generate(
                teacher_inputs,
                max_length=100,
                num_beams=4,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        teacher_text = self.tokenizer.decode(teacher_outputs[0], skip_special_tokens=True)
        
        student_text = self.generate_text(prompt)
        
        return {
            'prompt': prompt,
            'teacher': teacher_text,
            'student': student_text
        }
    
    def save_model(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.student.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def create_training_data() -> List[str]:
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important skill for students today is",
        "Climate change affects our planet by",
        "The key to successful teamwork involves",
        "When facing difficult decisions, one should",
        "The role of education in society is to",
        "Innovation happens when people",
        "The relationship between humans and nature",
        "Effective communication requires",
        "The impact of social media on relationships",
        "Scientific breakthroughs often come from",
        "The importance of creativity in problem-solving",
        "Cultural diversity enriches communities by",
        "The future of work will likely involve",
        "Personal growth comes through",
        "The balance between tradition and progress",
        "Leadership in the modern world requires",
        "The power of storytelling lies in",
        "Understanding different perspectives helps us"
    ]
    return prompts

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = "cpu"
    logger.info(f"Using device: {device}")
    
    distiller = KnowledgeDistiller(device=device)
    
    training_prompts = create_training_data()
    
    distiller.train(
        train_prompts=training_prompts,
        num_epochs=2,
        batch_size=2,
        learning_rate=5e-5
    )
    
    test_prompts = [
        "The secret to happiness is",
        "Technology will change education by",
        "The most valuable lesson I've learned is"
    ]
    
    print("\n" + "="*60)
    print("COMPARING TEACHER VS STUDENT OUTPUTS")
    print("="*60)
    
    for prompt in test_prompts:
        comparison = distiller.compare_outputs(prompt)
        print(f"\nPrompt: {comparison['prompt']}")
        print(f"Teacher: {comparison['teacher']}")
        print(f"Student: {comparison['student']}")
        print("-" * 60)
    
    distiller.save_model("./distilled_model")
    
    print("\nTraining completed! The small model has learned to mimic the larger model's behavior.")
    print("Check the outputs above to see the knowledge transfer in action!")

if __name__ == "__main__":
    main()
