import os
import argparse
import numpy as np
import librosa

from transformers import AutoProcessor, AutoModel
import torch

SAMPLE_RATE = 16000
MODEL_ID = os.environ.get("MODEL_ID", "facebook/wav2vec2-base")

PROC = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModel.from_pretrained(MODEL_ID)
MODEL.eval()


def load_audio_array(audio_input):
    if isinstance(audio_input, tuple):
        sr, y = audio_input
    elif isinstance(audio_input, str):
        y, sr = librosa.load(audio_input, sr=None, mono=False)
    elif isinstance(audio_input, np.ndarray):
        sr, y = SAMPLE_RATE, audio_input
    else:
        raise ValueError("Unsupported audio input")
    

    if y.ndim > 1:
        y = np.mean(y, 0)

    if sr != SAMPLE_RATE:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    
    return sr, y.astype(np.float32)

def extract(y, sr):
    with torch.no_grad():
        inputs = PROC(y, sampling_rate=sr, return_tensors="pt")
        out = MODEL(**inputs).last_hidden_state
    H = out.squeeze(0).cpu().numpy()

    duration = len(y)/float(sr)
    T = H.shape[0]
    hop_s = duration / float(T)

    dH = np.diff(H, 0)
    env = np.linalg.norm(dH, axis=1)
    env = np.concatenate([[0.0], env], axis=0)
    env = (env - env.mean()) / (env.std() + 1e-8)
    env = np.maximum(env, 0.0)

    return env.astype(np.float32), hop_s

def detect_peaks(env, hop_s):
    sr_env = 1.0 / hop_s
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr_env, hop_length=1, units="time", backtrack=False, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1)

    pruned = [peaks[0]]

    for t in peaks[1:]:
        if t - pruned[-1] >= 0.06:
            pruned.append(t)

    return np.array(pruned, dtype=float)

def preset_beats(duration, bpm):
    period = 60.0 / bpm
    n = int(np.ceil(duration / period)) + 1

    return np.arange(n, dtype=float) * period

def time_from_beat(events, beats):
    diffs = np.abs(events[:, None] - beats[None, :])
    nearest = np.min(diffs, 1)

    return nearest * 1000.0

def calculate_score(y, sr, bpm):
    duration = len(y) / float(sr)
    env, hop_s = extract(y, sr)
    events = detect_peaks(env, hop_s)
    beats = preset_beats(duration, bpm)
    drifts_ms = time_from_beat(events, beats)

    if len(drifts_ms) == 0:
        return 0.0, events.tolist(), []
    
    mean_ms = float(np.mean(drifts_ms))
    score = max(0.0, 100.0 - (mean_ms / 5.0))
    
    return round(score, 1), events.tolist(), drifts_ms.tolist()

def launch_ui():
    import gradio as gr
    def _fn(audio, bpm):
        if audio is None or not bpm or bpm <= 0:
            return "Please provide audio and a positive BPM."
        sr, y = load_audio_array(audio)
        score, events, drifts = calculate_score(y, sr, bpm)
        if not drifts:
            return "No clear onsets detected from transformer embeddings; try speaking more percussively or raise volume."
        lines = "\n".join(f"- onset @{t:.2f}s → drift {d:.0f} ms" for t, d in list(zip(events, drifts))[:50])
        return f"Rhythm Score: {score}/100\n\nOnsets & drifts (first 50):\n{lines}"

    ui = gr.Interface(
        fn=_fn,
        inputs=[
            gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload audio"),
            gr.Number(label="BPM", value=90),
        ],
        outputs=gr.Textbox(label="Rhythm feedback"),
        title="RhythmScore-HF — transformer-based onsets",
        description="Uses a Hugging Face speech transformer (Wav2Vec2) to derive onsets, then scores beat alignment."
    )
    ui.launch()

def run_cli(wav_path, bpm):
    sr, y = load_audio_array(wav_path)
    score, events, drifts = calculate_score(y, sr, bpm)
    if not drifts:
        print("No clear onsets detected; try a different sample or speak more percussively.")
        return
    print(f"Rhythm Score: {score}/100")
    print("Onsets (s) and drifts (ms):")
    for t, d in zip(events[:50], drifts[:50]):
        print(f"  {t:.2f}s -> {d:.0f} ms")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true", help="Run in CLI mode")
    ap.add_argument("--wav", type=str, help="Path to audio when using --cli")
    ap.add_argument("--bpm", type=float, default=90.0)
    args = ap.parse_args()

    if args.cli:
        if not args.wav or args.bpm <= 0:
            raise SystemExit("Usage: python main.py --cli --wav path/to.wav --bpm 90")
        run_cli(args.wav, args.bpm)
    else:
        launch_ui()
