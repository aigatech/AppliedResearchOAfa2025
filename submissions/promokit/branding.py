import requests, base64, os

def remote_image(prompt):
    token = os.getenv("HF_TOKEN")
    url = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(url, headers=headers, json={"inputs": prompt})
    r.raise_for_status()
    with open("bg.png", "wb") as f:
        f.write(r.content)
    return "bg.png"