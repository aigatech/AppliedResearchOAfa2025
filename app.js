:root {
  --bg: #0c1024;
  --bg-grad-1: #0c1024;
  --bg-grad-2: #121a3f;
  --fg: #e9ecf8;
  --muted: #a7b0d4;
  --card: #121936;
  --card-border: #202a53;
  --accent: #7aa2f7;
  --accent2: #89dceb;
  --danger: #f7768e;
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif;
  color: var(--fg);
  background: linear-gradient(180deg, var(--bg-grad-1), var(--bg-grad-2));
  padding: 24px 16px 48px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.site-header, .site-footer, .card, .actions, form { width: 100%; max-width: 920px; }

.site-header h1 {
  margin: 0 0 8px;
  font-size: 28px;
  letter-spacing: 0.2px;
}
.muted { color: var(--muted); }

.card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 18px;
  margin: 16px 0;
  box-shadow: 0 14px 30px rgba(0,0,0,.25);
}

.grid { display: grid; gap: 14px; }

.form-group label { font-weight: 600; display: block; margin-bottom: 8px; }

input[type=file],
textarea {
  width: 100%;
  background: #101738;
  color: var(--fg);
  border: 1px solid #273163;
  border-radius: 12px;
  padding: 10px 12px;
  outline: none;
}
textarea::placeholder { color: #6f79ab; }

.switch-label {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; font-weight: 600;
}

button, .btn {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #071129;
  font-weight: 700;
  border: 0;
  border-radius: 12px;
  padding: 10px 16px;
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
  transition: transform .05s ease, filter .15s ease;
}
button:hover, .btn:hover { filter: brightness(1.06); }
button:active { transform: translateY(1px); }
.actions { margin: 12px 0 24px; }

.alert {
  background: #2a0f16;
  border: 1px solid #5c1f2a;
  padding: 12px;
  border-radius: 12px;
  color: var(--danger);
  width: 100%;
  max-width: 920px;
}

pre.codebox {
  background: #0e1430;
  border: 1px solid #23305f;
  padding: 12px;
  border-radius: 12px;
  overflow: auto;
  margin: 0;
}
code { color: #c8d2ff; }

/* Toggle switch */
.switch { position: relative; display: inline-block; width: 48px; height: 28px; }
.switch input { display: none; }
.slider {
  position: absolute; cursor: pointer; inset: 0;
  background-color: #3a4165; transition: .2s; border-radius: 28px;
}
.slider:before {
  position: absolute; content: ""; height: 22px; width: 22px; left: 3px; bottom: 3px;
  background-color: white; transition: .2s; border-radius: 50%;
}
input:checked + .slider { background: var(--accent); }
input:checked + .slider:before { transform: translateX(20px); }

/* Processing overlay */
.overlay {
  position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
  background: rgba(6, 10, 24, 0.6); backdrop-filter: blur(3px); z-index: 9999;
}
.is-processing .overlay { display: flex; }
.overlay-box {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 18px 20px;
  display: flex; flex-direction: column; align-items: center; gap: 12px;
  width: min(90vw, 420px);
  text-align: center;
}
.spinner {
  width: 28px; height: 28px; border: 3px solid #2a356e; border-top-color: var(--accent);
  border-radius: 50%; animation: spin 0.9s linear infinite;
}
.overlay-text { font-weight: 700; }
.overlay-subtext { color: var(--muted); font-size: 14px; }

@keyframes spin { to { transform: rotate(360deg); } }

.site-footer { margin-top: 8px; }
