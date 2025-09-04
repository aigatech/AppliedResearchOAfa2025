import argparse, json, os, random, hashlib
from pathlib import Path

from app import (
    get_text, sent_split, get_teacher_pipes, rank_sentences,
    teacher_make_item, list_entities, build_entity_pool, swap_entity
)

random.seed(42)

def norm_key(sentence: str, forced_entity):
    ent_txt, ent_typ = (forced_entity or ("", ""))
    raw = f"{sentence.strip()}||{(ent_txt or '').strip()}||{(ent_typ or '').strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def fabricate_item(sentence: str, ent_text: str, ent_type: str, pool_by_type: dict):
    choices = [x for x in list(pool_by_type.get(ent_type, [])) if x.lower() != (ent_text or "").lower()]
    random.shuffle(choices)
    if len(choices) < 2:
        return None

    wrong1 = swap_entity(sentence, ent_text, choices[0])
    wrong2 = swap_entity(sentence, ent_text, choices[1])

    expl = (
        f"'{choices[0]}' and '{choices[1]}' are {ent_type} entities that do not match the original fact in context. "
        f"The correct statement includes '{ent_text}' instead."
    )

    return {
        "context": sentence,
        "entity": ent_text,
        "correct": sentence,
        "distractors": [wrong1, wrong2],
        "explanations": [expl]
    }

def build_example(sentence: str, entity_pool, forced_entity=None):
    item = teacher_make_item(sentence, entity_pool=entity_pool, forced_entity=forced_entity)
    if item:
        inst = (
            "Task: Given a factual sentence, produce a JSON object with keys: "
            "correct (string), distractors (array of two strings), explanations (array with 1-2 sentences). "
            "The distractors must be plausible but incorrect and of the same entity type.\n"
            f"Sentence: {item['context']}\nOutput strictly valid JSON."
        )
        tgt = json.dumps({
            "correct": item["correct"],
            "distractors": item["distractors"][:2],
            "explanations": item.get("explanations", [])
        }, ensure_ascii=False)
        return {"input": inst, "target": tgt}

    if forced_entity is not None:
        ent_text, ent_type = forced_entity
    else:
        ent_text, ent_type = None, None

    if ent_text and ent_type:
        item2 = fabricate_item(sentence, ent_text, ent_type, entity_pool)
        if item2:
            inst = (
                "Task: Given a factual sentence, produce a JSON object with keys: "
                "correct (string), distractors (array of two strings), explanations (array with 1-2 sentences). "
                "The distractors must be plausible but incorrect and of the same entity type.\n"
                f"Sentence: {item2['context']}\nOutput strictly valid JSON."
            )
            tgt = json.dumps({
                "correct": item2["correct"],
                "distractors": item2["distractors"][:2],
                "explanations": item2.get("explanations", [])
            }, ensure_ascii=False)
            return {"input": inst, "target": tgt}

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="text file with one URL or local path per line")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--k", type=int, default=30, help="TARGET items to MAKE per source")
    ap.add_argument("--per_sentence_max", type=int, default=5, help="max entities to try per sentence")
    ap.add_argument("--max_ranked", type=int, default=400, help="how many ranked sents to consider per source")
    ap.add_argument("--target_total", type=int, default=1000, help="stop when total items reach this number")
    args = ap.parse_args()

    srcs = [x.strip() for x in Path(args.sources).read_text(encoding="utf-8").splitlines() if x.strip()]
    random.shuffle(srcs)

    os.makedirs(Path(args.out).parent, exist_ok=True)

    fe, ner, _ = get_teacher_pipes()
    n_ok = 0
    seen_keys = set()

    with open(args.out, "w", encoding="utf-8") as f:
        for si, src in enumerate(srcs, 1):
            if n_ok >= args.target_total:
                break
            try:
                text = get_text(src)
                sents = sent_split(text)
                if not sents:
                    print(f"[warn] no usable sentences in {src}; skipping")
                    continue

                ranked = rank_sentences(sents, fe)
                pool = build_entity_pool(ranked[:min(len(ranked), args.max_ranked)], ner)

                made_here = 0
                tried_sentences = 0

                for s in ranked[:args.max_ranked]:
                    if made_here >= args.k or n_ok >= args.target_total:
                        break
                    ents = list_entities(s, ner)[:args.per_sentence_max]
                    tried_sentences += 1
                    if not ents:
                        continue

                    for ent in ents:
                        if made_here >= args.k or n_ok >= args.target_total:
                            break
                        key = norm_key(s, ent)
                        if key in seen_keys:
                            continue

                        ex = build_example(s, pool, forced_entity=ent)
                        if ex:
                            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            seen_keys.add(key)
                            n_ok += 1
                            made_here += 1

                print(f"[info] [{si}/{len(srcs)}] {src}: made {made_here}/{args.k} (tried {tried_sentences} sents; total={n_ok})")

            except Exception as e:
                print(f"[warn] skipping {src}: {e}")

    print(f"Wrote {n_ok} examples to {args.out}")

if __name__ == "__main__":
    main()
