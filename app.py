import os, io, json, base64, re, uuid, datetime as dt
from flask import Flask, request, redirect, url_for, send_file, render_template, flash
import dateparser
from dateutil.tz import gettz
from ics import Calendar, Event, DisplayAlarm
from openai import OpenAI

CAL_TZ = "America/New_York"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    from dotenv import load_dotenv
    for fname in ("key.env", ".env"):
        p = os.path.join(BASE_DIR, fname)
        if os.path.exists(p):
            load_dotenv(p, override=False)
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.secret_key = FLASK_SECRET

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

HF_EXTRACT_MODEL = os.getenv("HF_EXTRACT_MODEL", "google/flan-t5-base")
HF_MODEL_CANDIDATES = [HF_EXTRACT_MODEL, "google/flan-t5-large", "google/flan-t5-base"]

try:
    import torch
    from transformers import pipeline
    DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    from transformers import pipeline
    DEVICE = -1

_HF_PIPES = {}
def _load_hf_extractor(model_name):
    if model_name not in _HF_PIPES:
        _HF_PIPES[model_name] = pipeline("text2text-generation", model=model_name, device=DEVICE)
    return _HF_PIPES[model_name]

# ====== OCR with GPT-4o ======
def ocr_with_gpt4o(image_path):
    if not client:
        raise RuntimeError("OPENAI_API_KEY not set.")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract ONLY the plain text from this image. No commentary."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        temperature=0
    )
    return (resp.choices[0].message.content or "").strip()

STRUCTURE_SCHEMA = {
  "appointments": [{"title":"","date":"","time":"","end_time":"","location":"","notes":""}],
  "medications": [{"name":"","dose":"","route":"","freq":"","times":"","start_date":"","end_date":"","instructions":""}],
  "reminders": [{"title":"","date":"","time":"","notes":""}],
  "instructions": ""
}

STRUCTURE_INSTRUCTION = (
    "You are an information extractor for doctor's notes. "
    "Return ONLY valid JSON that matches this exact schema and key names:\n\n"
    + json.dumps(STRUCTURE_SCHEMA, indent=2)
    + "\n\nRules:\n"
      "- Use \"\" for unknown/absent fields.\n"
      "- No extra keys, no explanations, no code fences.\n"
      "- Do not include trailing commas.\n"
      "- Keep strings short where possible.\n"
      "Output JSON only."
)

def _sanitize_json_like(s):
    s2 = (s or "").strip()
    s2 = s2.replace("“", '"').replace("”", '"').replace("’", "'")
    s2 = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s2, flags=re.IGNORECASE|re.MULTILINE)
    s2 = re.sub(r",\s*(?=[}\]])", "", s2)
    return s2

def _extract_first_json_obj(s):
    s2 = _sanitize_json_like(s)
    dec = json.JSONDecoder()
    for i, ch in enumerate(s2):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(s2[i:])
                return obj
            except json.JSONDecodeError:
                continue
    m = re.search(r"\{[\s\S]*\}", s2)
    if m:
        try:
            return dec.raw_decode(m.group(0))[0]
        except Exception:
            pass
    raise ValueError("No valid JSON object found in text")

def _coerce_schema(data):
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), {})
    if not isinstance(data, dict):
        data = {}
    for k in ["appointments","medications","reminders"]:
        if k not in data or not isinstance(data.get(k), list):
            data[k] = []
    if "instructions" not in data or not isinstance(data.get("instructions"), str):
        data["instructions"] = ""
    for appt in data["appointments"]:
        for k in ["title","date","time","end_time","location","notes"]:
            appt.setdefault(k,"")
    for med in data["medications"]:
        for k in ["name","dose","route","freq","times","start_date","end_date","instructions"]:
            med.setdefault(k,"")
    for r in data["reminders"]:
        for k in ["title","date","time","notes"]:
            r.setdefault(k,"")
    return data

def _is_empty_struct(d):
    return not (d.get("appointments") or d.get("medications") or d.get("reminders"))

def _merge_struct(a, b):
    a = _coerce_schema(a); b = _coerce_schema(b)
    out = {"appointments": [], "medications": [], "reminders": [], "instructions": a.get("instructions") or b.get("instructions","")}
    seen = set()
    def _add_many(key, items, fields):
        for it in items:
            sig = (key,) + tuple((it.get(f,"") or "").strip() for f in fields)
            if sig in seen: continue
            seen.add(sig)
            out[key].append({f: (it.get(f,"") or "") for f in fields})
    _add_many("appointments", a["appointments"]+b["appointments"], ["title","date","time","end_time","location","notes"])
    _add_many("medications", a["medications"]+b["medications"], ["name","dose","route","freq","times","start_date","end_date","instructions"])
    _add_many("reminders", a["reminders"]+b["reminders"], ["title","date","time","notes"])
    return out

_TIME_RE = re.compile(r"\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b", re.I)
_DURATION_RE = re.compile(r"\bfor\s+(\d+)\s+(day|days|week|weeks)\b", re.I)
_STARTING_RE = re.compile(r"\bstarting\s+(today|tomorrow|on\s+[A-Za-z]{3,}\s+\d{1,2}|\d{4}-\d{2}-\d{2}|next\s+[A-Za-z]+)\b", re.I)
_ROUTE_WORDS = r"(oral|by mouth|po|iv|im|subcut|sc|subcutaneous|topical|inhale|inhalation)"
_FREQ_WORDS = r"(?:once(?:\s+daily)?|1x/day|qd|twice(?:\s+daily)?|2x/day|bid|3x/day|tid|4x/day|qid|qhs|bedtime|night|qam|morning)"
_ROUTE_RE = re.compile(rf"(?<!\w){_ROUTE_WORDS}(?!\w)", re.I)
_FREQ_RE  = re.compile(rf"(?<!\w){_FREQ_WORDS}(?!\w)", re.I)
_FOLLOWUP_RE = re.compile(r"\bfollow[- ]?up(?:\s+with\s+(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))?", re.I)
_REMINDER_RE = re.compile(r"\bset\s+reminder\s+to\s+(.+)", re.I)
_ON_AT_RE = re.compile(r"\bon\s+([^,.;\n]+?)(?:\s+at\s+([^,.;\n]+))?(?:$|\b)", re.I)

def _parse_rel_date(text, base):
    p = dateparser.parse(text, settings={"RELATIVE_BASE": base})
    return p.date() if p else None

def heuristic_extract(note, base_dt=None):
    base_dt = base_dt or dt.datetime.now(gettz(CAL_TZ))
    out = {"appointments": [], "medications": [], "reminders": [], "instructions": ""}

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", note) if ln.strip()]
    windows = []
    for i in range(len(lines)):
        seg1 = lines[i]
        seg2 = f"{lines[i]} {lines[i+1]}" if i+1 < len(lines) else lines[i]
        seg3 = f"{lines[i]} {lines[i+1]} {lines[i+2]}" if i+2 < len(lines) else seg2
        windows.extend([seg1, seg2, seg3])
    windows.append(" ".join(lines))
    seen = set()

    for s in windows:
        if "follow" not in s.lower(): continue
        m = _FOLLOWUP_RE.search(s)
        if not m: continue
        who = (m.group(1) or "").strip()
        dm = _ON_AT_RE.search(s)
        date_str = dm.group(1).strip() if dm else ""
        time_str = (dm.group(2) or "").strip() if dm else ""
        d = _parse_rel_date(date_str, base_dt) if date_str else None
        sig = ("appt", who, date_str, time_str)
        if sig in seen: continue
        seen.add(sig)
        title = f"Follow-up: {who or 'Doctor'}".strip(": ")
        out["appointments"].append({"title": title, "date": d.isoformat() if d else (date_str or ""), "time": time_str, "end_time": "", "location": "", "notes": s.strip()})

    for s in windows:
        if not re.search(r"\b(start|begin|take)\b", s, re.I): continue
        m = re.search(r"\b(?:start|begin|take)\s+([A-Z][A-Za-z0-9\-]+)\s+(\d+(?:\.\d+)?)\s?(mg|mcg|g|ml|units|iu)\b", s, re.I)
        if not m: continue
        name = m.group(1); dose = f"{m.group(2)} {m.group(3)}"
        route = ""; freq = ""
        times = ", ".join(t.group(1) for t in _TIME_RE.finditer(s))
        mr = _ROUTE_RE.search(s); mf = _FREQ_RE.search(s)
        if mr: route = mr.group(1).lower().replace("by mouth", "oral").replace("po", "oral")
        if mf: freq = mf.group(0).lower()
        start_date = ""; end_date = ""
        dur_m = _DURATION_RE.search(s); start_m = _STARTING_RE.search(s)
        if start_m:
            sd_phrase = start_m.group(1)
            sd_phrase = sd_phrase if sd_phrase.lower().startswith(("today","tomorrow")) else sd_phrase.replace("on ", "")
            sd = _parse_rel_date(sd_phrase, base_dt)
            if sd: start_date = sd.isoformat()
        if dur_m:
            num = int(dur_m.group(1)); unit = dur_m.group(2).lower()
            days = num * (7 if "week" in unit else 1)
            if start_date:
                sd = dt.date.fromisoformat(start_date)
                end_date = (sd + dt.timedelta(days=days)).isoformat()
            else:
                end_date = f"in {days} days"
        if not start_date: start_date = "today"
        sig = ("med", name, dose, route, freq, start_date, end_date, times)
        if sig in seen: continue
        seen.add(sig)
        out["medications"].append({"name": name, "dose": dose, "route": route, "freq": freq, "times": times, "start_date": start_date, "end_date": end_date, "instructions": ""})

    for s in windows:
        rm = _REMINDER_RE.search(s)
        if not rm: continue
        rest = rm.group(1).strip()
        onm = _ON_AT_RE.search(rest)
        date_phrase = (onm.group(1) or "").strip() if onm else ""
        if not date_phrase:
            mlast = re.search(r"(next\s+[A-Za-z]+|tomorrow|today|on\s+[A-Za-z]{3,}\s+\d{1,2}|\d{4}-\d{2}-\d{2})\b", rest, re.I)
            if mlast: date_phrase = mlast.group(1).strip()
        d_obj = _parse_rel_date(date_phrase, base_dt) if date_phrase else None
        title = rest
        if date_phrase: title = title.replace(date_phrase, "").strip(",. ").strip()
        if title.lower().startswith(("to ","on ","by ","at ")): title = title.split(" ", 1)[-1].strip()
        if not title: title = "Reminder"
        sig = ("rem", title.lower(), (date_phrase or "").lower())
        if sig in seen: continue
        seen.add(sig)
        out["reminders"].append({"title": title[:120], "date": d_obj.isoformat() if d_obj else (date_phrase or ""), "time": "", "notes": ""})

    if _is_empty_struct(out): out["instructions"] = note[:2000]
    return _coerce_schema(out)

def _extract_with_hf_or_none(ocr_text):
    note = (ocr_text or "").strip()
    prompts = [
        f"{STRUCTURE_INSTRUCTION}\n\nNote:\n{note}\n\nJSON:",
        (f"{STRUCTURE_INSTRUCTION}\n\n"
         "Example:\n"
         "{\n"
         "  \"appointments\": [{\"title\":\"Dr. Lee\",\"date\":\"Sep 12\",\"time\":\"3 pm\",\"end_time\":\"\",\"location\":\"\",\"notes\":\"\"}],\n"
         "  \"medications\": [{\"name\":\"Amoxicillin\",\"dose\":\"500 mg\",\"route\":\"oral\",\"freq\":\"3x/day\",\"times\":\"8am,2pm,9pm\",\"start_date\":\"today\",\"end_date\":\"in 7 days\",\"instructions\":\"with food\"}],\n"
         "  \"reminders\": [{\"title\":\"Book lab tests\",\"date\":\"next Monday\",\"time\":\"\",\"notes\":\"\"}],\n"
         "  \"instructions\": \"\"\n"
         "}\n\n"
         f"Note:\n{note}\n\nJSON:"),
        ("Return strictly one JSON object with ONLY these keys: "
         "\"appointments\",\"medications\",\"reminders\",\"instructions\". "
         "Empty strings for unknowns.\nSchema:\n" + json.dumps(STRUCTURE_SCHEMA) + f"\nNote:\n{note}\nJSON:")
    ]
    for model_name in HF_MODEL_CANDIDATES:
        try:
            pipe = _load_hf_extractor(model_name)
        except Exception:
            continue
        for pr in prompts:
            try:
                out_obj = pipe(pr, max_new_tokens=512, num_beams=4, do_sample=False)[0]
                out = (out_obj.get("generated_text") or out_obj.get("text") or "").strip()
                data = _extract_first_json_obj(out)
                return _coerce_schema(data)
            except Exception:
                continue
    return None

def structure_note_to_json_hf(ocr_text):
    data = _extract_with_hf_or_none(ocr_text)
    if data is None or _is_empty_struct(data):
        heur = heuristic_extract(ocr_text)
        return heur if (data is None or _is_empty_struct(data)) else _merge_struct(data, heur)
    return _coerce_schema(data)

structure_note_to_json = structure_note_to_json_hf

def parse_date(date_str, base=None):
    if not date_str or not date_str.strip(): return None
    base_dt = base or dt.datetime.now(gettz(CAL_TZ))
    parsed = dateparser.parse(date_str, settings={"RELATIVE_BASE": base_dt})
    return parsed.date() if parsed else None

def parse_time(time_str, base=None):
    if not time_str or not time_str.strip(): return None
    base_dt = base or dt.datetime.now(gettz(CAL_TZ))
    parsed = dateparser.parse(time_str, settings={"RELATIVE_BASE": base_dt})
    return parsed.time() if parsed else None

def parse_times_csv(times_str, base=None):
    if not times_str or not times_str.strip(): return []
    parts = [p.strip() for p in times_str.split(",") if p.strip()]
    out = []
    for p in parts:
        t = parse_time(p, base=base)
        if t: out.append(t)
    return out

def coerce_dt(date_obj, time_obj):
    tz = gettz(CAL_TZ)
    if time_obj is None:
        time_obj = dt.time(hour=9, minute=0)
    return dt.datetime.combine(date_obj, time_obj).replace(tzinfo=tz)

def add_alarm(evt, minutes_before=30):
    alarm = DisplayAlarm(trigger=dt.timedelta(minutes=-abs(minutes_before)))
    evt.alarms.append(alarm)

def build_calendar(struct, out_path):
    cal = Calendar()
    now_tz = dt.datetime.now(gettz(CAL_TZ))

    for appt in struct.get("appointments", []):
        title = (appt.get("title") or "").strip() or "Appointment"
        d = parse_date(appt.get("date",""), base=now_tz)
        start_t = parse_time(appt.get("time",""), base=now_tz)
        end_t = parse_time(appt.get("end_time",""), base=now_tz)
        if not d: continue
        start_dt = coerce_dt(d, start_t)
        end_dt = coerce_dt(d, end_t) if end_t else (start_dt + dt.timedelta(hours=1))
        e = Event()
        e.name = title
        e.begin = start_dt
        e.end = end_dt
        location = (appt.get("location") or "").strip()
        notes = (appt.get("notes") or "").strip()
        desc_parts = []
        if location: desc_parts.append(f"Location: {location}")
        if notes: desc_parts.append(notes)
        e.description = "\n".join(desc_parts) if desc_parts else ""
        add_alarm(e, minutes_before=45)
        cal.events.add(e)

    for med in struct.get("medications", []):
        name = (med.get("name") or "").strip() or "Medication"
        dose = (med.get("dose") or "").strip()
        route = (med.get("route") or "").strip()
        instr = (med.get("instructions") or "").strip()
        start_d = parse_date(med.get("start_date","") or "today", base=now_tz)
        end_d = parse_date(med.get("end_date",""), base=now_tz)
        if not start_d: start_d = now_tz.date()
        if not end_d: end_d = start_d + dt.timedelta(days=7)
        times = parse_times_csv(med.get("times",""), base=now_tz)
        if not times:
            freq = (med.get("freq","") or "").lower()
            if any(k in freq for k in ["once","1x","qd"]): times = [dt.time(9,0)]
            elif any(k in freq for k in ["twice","2x","bid"]): times = [dt.time(9,0), dt.time(21,0)]
            elif any(k in freq for k in ["3x","three","tid"]): times = [dt.time(8,0), dt.time(14,0), dt.time(21,0)]
            elif any(k in freq for k in ["4x","four","qid"]): times = [dt.time(8,0), dt.time(12,0), dt.time(16,0), dt.time(20,0)]
            elif any(k in freq for k in ["night","bedtime","qhs"]): times = [dt.time(22,0)]
            elif any(k in freq for k in ["morning","qam"]): times = [dt.time(9,0)]
            else: times = [dt.time(9,0)]
        desc = f"{name}"
        if dose: desc += f" — {dose}"
        if route: desc += f" ({route})"
        if instr: desc += f"\nInstructions: {instr}"
        cur = start_d
        while cur <= end_d:
            for t in times:
                e = Event()
                e.name = f"Take {name}"
                start_dt = coerce_dt(cur, t)
                e.begin = start_dt
                e.end = start_dt + dt.timedelta(minutes=10)
                e.description = desc
                add_alarm(e, minutes_before=15)
                cal.events.add(e)
            cur += dt.timedelta(days=1)

    for r in struct.get("reminders", []):
        title = (r.get("title") or "").strip() or "Reminder"
        d = parse_date(r.get("date",""), base=now_tz)
        t = parse_time(r.get("time",""), base=now_tz)
        if not d: continue
        start_dt = coerce_dt(d, t)
        e = Event()
        e.name = title
        e.begin = start_dt
        e.end = start_dt + dt.timedelta(minutes=15)
        notes = (r.get("notes") or "").strip()
        e.description = notes
        add_alarm(e, minutes_before=30)
        cal.events.add(e)

    if len(cal.events) == 0:
        e = Event()
        start_dt = dt.datetime.now(gettz(CAL_TZ)) + dt.timedelta(hours=2)
        e.name = "Review doctor's note"
        e.begin = start_dt
        e.end = start_dt + dt.timedelta(minutes=15)
        e.description = "No structured items detected. Review OCR text and add details."
        add_alarm(e, minutes_before=15)
        cal.events.add(e)

    with open(out_path, "w") as f:
        f.writelines(cal)
    return os.path.abspath(out_path)

def process_doctor_note_to_calendar(image_path, out_ics):
    ocr_text = ocr_with_gpt4o(image_path)
    struct = structure_note_to_json(ocr_text)
    ics_path = build_calendar(struct, out_ics)
    return {"ocr_text": ocr_text, "structured": struct, "ics_path": ics_path}

def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

TOKENS = {}

# ====== Routes ======
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, token=None, ics_name=None, has_key=bool(OPENAI_API_KEY))

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("index"))
    f = request.files["image"]
    if f.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not _allowed(f.filename):
        flash("Unsupported file type. Please upload PNG/JPG/TIFF/WEBP/BMP.")
        return redirect(url_for("index"))
    if not OPENAI_API_KEY:
        flash("OPENAI_API_KEY is not set. Please put it in key.env and restart the server.")
        return redirect(url_for("index"))

    ext = f.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_DIR, f"note_{uid}.{ext}")
    f.save(img_path)
    ics_path = os.path.join(OUTPUT_DIR, f"doctor_note_{uid}.ics")
    try:
        result = process_doctor_note_to_calendar(img_path, ics_path)
    except Exception as e:
        flash(f"Processing failed: {type(e).__name__}: {e}")
        return redirect(url_for("index"))
    token = uuid.uuid4().hex
    TOKENS[token] = {"ics": result["ics_path"]}
    return render_template("index.html", result=result, token=token, ics_name=os.path.basename(result["ics_path"]), has_key=bool(OPENAI_API_KEY))

@app.route("/download/<token>", methods=["GET"])
def download_ics(token):
    rec = TOKENS.get(token)
    if not rec:
        return "Invalid or expired link.", 404
    ics_path = rec["ics"]
    if not os.path.exists(ics_path):
        return "File not found.", 404
    return send_file(ics_path, as_attachment=True, download_name=os.path.basename(ics_path), mimetype="text/calendar")

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "4444")), debug=True)
