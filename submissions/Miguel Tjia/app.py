import gradio as gr
from transformers import pipeline


zeroshot   = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ner        = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
sentiment  = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

RISK_LABELS = [
    "regulatory risk","team risk","market risk","competition risk",
    "technical risk","revenue risk","funding risk","legal risk",
    "product risk","go-to-market risk","operational risk"
]
OPPORTUNITY_LABELS = [
    "large market","recurring revenue","network effects","strong team",
    "proprietary tech","clear moat","capital efficiency","high gross margins",
    "enterprise demand","developer adoption","regulatory tailwinds"
]

def trim(text: str, max_chars: int = 2500) -> str:
    t = (text or "").strip()
    return t if len(t) <= max_chars else t[:max_chars]

def extract_entities(text: str):
    ents = ner(text)
    orgs   = sorted({e["word"] for e in ents if e.get("entity_group") in ["ORG","MISC"]})
    people = sorted({e["word"] for e in ents if e.get("entity_group") == "PER"})
    money  = sorted({e["word"] for e in ents if e.get("entity_group") == "MONEY"})
    return orgs, people, money

def zs_rank(text: str, candidate_labels, top_k: int = 5):
    out = zeroshot(text, candidate_labels=candidate_labels, multi_label=True)
    pairs = list(zip(out["labels"], out["scores"]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def verdict_from_scores(risks, opps):
    rmax = risks[0][1] if risks else 0.0
    omax = opps[0][1] if opps else 0.0
    if rmax >= 0.80 and omax < 0.60: return "Risky 🔴"
    if omax >= 0.80 and rmax < 0.60: return "Promising 🟢"
    return "Mixed 🟠"

def label_score(score: float, positive: bool) -> str:
    if positive:
        if score >= 0.80: return f"{score:.2f} ✅ High"
        if score >= 0.50: return f"{score:.2f} ⚠️ Medium"
        return f"{score:.2f} ❌ Low"
    else:
        if score >= 0.80: return f"{score:.2f} 🚩 Critical"
        if score >= 0.50: return f"{score:.2f} ⚠️ Moderate"
        return f"{score:.2f} 🟢 Low"

def make_dd(text: str):
    text = trim(text)
    if not text:
        return ["", [], [], [], [], "", ""]

    risks   = zs_rank(text, RISK_LABELS, top_k=6)
    opps    = zs_rank(text, OPPORTUNITY_LABELS, top_k=6)

    orgs, people, money = extract_entities(text)

    tone = sentiment(text[:1000])[0]
    tone_str = f'{tone["label"]} (score {tone["score"]:.2f})'

    verdict = verdict_from_scores(risks, opps)

    opp_rows  = [ [f"{lbl} — {label_score(score, True)}"]  for lbl, score in opps ]
    risk_rows = [ [f"{lbl} — {label_score(score, False)}"] for lbl, score in risks ]

    qmap = {
        "regulatory risk":"Which regulations apply and how will you stay compliant?",
        "team risk":"Which key hires are missing and what’s the plan to fill them?",
        "market risk":"What evidence proves strong, growing demand in the target segment?",
        "competition risk":"Top 3 competitors and your durable edge vs each?",
        "technical risk":"Hardest technical milestones and how you’ll de-risk them?",
        "revenue risk":"Pricing, sales cycle, and proof of willingness to pay?",
        "funding risk":"Runway, burn, and plan if the next round is delayed?",
        "legal risk":"Any IP/data/privacy exposures or contract obligations?",
        "product risk":"Evidence it’s a must-have, not a nice-to-have (retention/usage)?",
        "go-to-market risk":"Which channel works now and what are CAC/LTV metrics?",
        "operational risk":"Current bottlenecks and how you’ll scale processes?"
    }
    questions = "\n".join(f"• {qmap.get(lbl, f'Mitigation plan for {lbl}?')}" for lbl, _ in risks[:4])

    return (
        verdict,
        opp_rows,
        risk_rows,
        orgs,
        people,
        money,
        tone_str,
        questions
    )

with gr.Blocks(title="Investor Due-Diligence Helper") as demo:
    gr.Markdown("## Investor Due-Diligence Helper\nPaste a startup description. Get risks, opportunities, tone, entities, and key diligence questions.")

    txt = gr.Textbox(
    label="Startup Description",
    lines=12,
    placeholder="Example: We build a billing API for Southeast Asia SMBs. $120k ARR, 35% MoM growth, pilot with 3 regional banks, SOC2 in progress, 8-person team. Paste deck blurbs, memos, or homepage copy here."
)

    run = gr.Button("Analyze")

    with gr.Row():
        verdict = gr.Textbox(label="Quick Verdict (Promising / Mixed / Risky)")
        tone    = gr.Textbox(label="Tone (Sentiment)")

    with gr.Row():
        opps = gr.Dataframe(headers=["Opportunities"], row_count=6, col_count=1, wrap=True)
        risks = gr.Dataframe(headers=["Risks"], row_count=6, col_count=1, wrap=True)

    with gr.Row():
        orgs  = gr.Dataframe(headers=["Organizations"], row_count=3, col_count=1)
        ppl   = gr.Dataframe(headers=["People"],       row_count=3, col_count=1)
        money = gr.Dataframe(headers=["Money terms"],  row_count=3, col_count=1)

    qs = gr.Textbox(label="Questions to Ask (Auto-Generated)", lines=6)

    run.click(fn=make_dd, inputs=[txt],
              outputs=[verdict, opps, risks, orgs, ppl, money, tone, qs])

if __name__ == "__main__":
    demo.launch()
