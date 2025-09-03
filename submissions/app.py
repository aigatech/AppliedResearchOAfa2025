import torch
from PIL import Image
import gradio as gr
from transformers import (
   AutoImageProcessor,
   AutoModelForImageClassification,
   AutoTokenizer,
   AutoModelForSeq2SeqLM,
)


processor = AutoImageProcessor.from_pretrained("nateraw/food")
clf_model = AutoModelForImageClassification.from_pretrained("nateraw/food")
clf_model.eval()
id2label = clf_model.config.id2label


RECIPE_MODEL_ID = "flax-community/t5-recipe-generation"
tok = AutoTokenizer.from_pretrained(RECIPE_MODEL_ID, use_fast=True)
t5 = AutoModelForSeq2SeqLM.from_pretrained(RECIPE_MODEL_ID)
t5.eval()


prefix = "items: "
gen_kwargs = {
   "max_length": 512,
   "min_length": 64,
   "no_repeat_ngram_size": 3,
   "do_sample": True,
   "top_k": 60,
   "top_p": 0.95,
}


SPECIAL_TOKENS = tok.all_special_tokens
TOKENS_MAP = {"<sep>": "--", "<section>": "\n"}


def _skip_special(text: str) -> str:
   for token in SPECIAL_TOKENS:
       text = text.replace(token, "")
   return text


def _postprocess_recipe(text: str) -> str:
   text = _skip_special(text)
   for k, v in TOKENS_MAP.items():
       text = text.replace(k, v)


   lines = [s.strip() for s in text.split("\n") if s.strip()]
   title, ingredients, steps = None, [], []
   for s in lines:
       low = s.lower()
       if low.startswith("title:"):
           title = s.split(":", 1)[1].strip()
       elif low.startswith("ingredients:"):
           ingredients = [it.strip(" -") for it in s.split(":", 1)[1].split("--") if it.strip()]
       elif low.startswith("directions:") or low.startswith("instructions:"):
           steps = [it.strip(" -") for it in s.split(":", 1)[1].split("--") if it.strip()]


   if not (title or ingredients or steps):
       return text 


   out = []
   if title:
       out.append(f"## {title}")
   if ingredients:
       out.append("**Ingredients**")
       out += [f"- {it}" for it in ingredients]
   if steps:
       out.append("\n**Steps**")
       out += [f"{i+1}. {st}" for i, st in enumerate(steps)]
   return "\n".join(out)


def _generate_recipe_from_label(label: str) -> str:
   prompt = prefix + label 
   enc = tok(prompt, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
   out = t5.generate(
       input_ids=enc.input_ids,
       attention_mask=enc.attention_mask,
       **gen_kwargs
   )
   raw = tok.batch_decode(out, skip_special_tokens=False)[0]
   return _postprocess_recipe(raw)


@torch.inference_mode()
def predict_and_recipe(img: Image.Image):
   inputs = processor(images=img, return_tensors="pt")
   logits = clf_model(**inputs).logits
   probs = logits.softmax(dim=1).squeeze(0)


   idx = probs.argmax().item()
   label = id2label[idx]
   conf = float(probs[idx].item())


   recipe_md = _generate_recipe_from_label(label)


   return {label: conf}, recipe_md


demo = gr.Interface(
   fn=predict_and_recipe,
   inputs=gr.Image(type="pil", label="Upload a food image"),
   outputs=[
       gr.Label(num_top_classes=1, label="Prediction"),
       gr.Markdown(label="How to Make It"),
   ],
   allow_flagging="never",
   title="Food Prediction + Recipe Generator",
   description="",
   examples=None,
)


if __name__ == "__main__":
   demo.launch()