import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model_name = "wambugu71/crop_leaf_diseases_vit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.to(device).eval()

disease_keywords = ['rust', 'blight', 'spot', 'mosaic', 'wilt', 'scab', 'powdery', 'downy']
healthy_keywords = ['healthy', 'normal', 'good', 'fresh']

def generate_integrated_gradients(model, processor, image, target_class=None, steps=50):
    inputs = processor(images=image, return_tensors="pt").to(device)
    baseline = torch.zeros_like(inputs['pixel_values']).to(device)
    
    if target_class is None:
        with torch.no_grad():
            outputs = model(**inputs)
            target_class = outputs.logits.argmax(dim=-1).item()
    
    alphas = torch.linspace(0, 1, steps).to(device)
    
    integrated_grads = torch.zeros_like(inputs['pixel_values']).to(device)
    
    for alpha in alphas:
        interpolated = baseline + alpha * (inputs['pixel_values'] - baseline)
        interpolated.requires_grad_(True)
        
        outputs = model(interpolated)
        score = outputs.logits[0, target_class]
        
        model.zero_grad()
        score.backward()
        
        integrated_grads += interpolated.grad.detach()
    
    integrated_grads = integrated_grads / steps
    attribution = integrated_grads * (inputs['pixel_values'] - baseline)
    
    attribution = torch.sum(torch.abs(attribution), dim=1)[0]
    
    return attribution.cpu().numpy()

def generate_simple_gradients(model, processor, image, target_class=None):
    inputs = processor(images=image, return_tensors="pt").to(device)
    input_tensor = inputs['pixel_values']
    input_tensor.requires_grad_(True)
    
    outputs = model(input_tensor)
    
    if target_class is None:
        target_class = outputs.logits.argmax(dim=-1).item()
    
    score = outputs.logits[0, target_class]
    model.zero_grad()
    score.backward()
    
    gradients = input_tensor.grad.detach()
    saliency = torch.sum(torch.abs(gradients), dim=1)[0]
    
    return saliency.cpu().numpy()

def generate_disease_focused_map(model, processor, image, target_class=None, patch_size=48, stride=32):
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    max_size = 300
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_array = np.array(resized_image)
        h, w = new_h, new_w
        print(f"Resized image to {w}x{h} for faster processing")
    else:
        resized_image = image
    
    inputs = processor(images=resized_image, return_tensors="pt").to(device)
    with torch.no_grad():
        baseline_output = model(**inputs)
        if target_class is None:
            target_class = baseline_output.logits.argmax(dim=-1).item()
        baseline_probs = F.softmax(baseline_output.logits, dim=-1)[0]
        baseline_disease_score = baseline_probs[target_class].item()
    
    target_label = model.config.id2label[target_class].lower()
    is_disease_prediction = any(keyword in target_label for keyword in disease_keywords)
    
    disease_importance_map = np.zeros((h, w))
    
    effective_patch_size = min(patch_size, h//4, w//4)
    effective_stride = max(stride, effective_patch_size//2)
    
    total_patches = ((h - effective_patch_size) // effective_stride + 1) * ((w - effective_patch_size) // effective_stride + 1)
    print(f"Processing {total_patches} patches with size {effective_patch_size}x{effective_patch_size}")
    
    patch_count = 0
    if is_disease_prediction:
        print("Disease detected - mapping disease regions...")
        
        for y in range(0, h - effective_patch_size + 1, effective_stride):
            for x in range(0, w - effective_patch_size + 1, effective_stride):
                patch_count += 1
                if patch_count % 10 == 0:
                    print(f"Processed {patch_count}/{total_patches} patches")
                
                occluded_img = img_array.copy()
                patch_mean = img_array[y:y+effective_patch_size, x:x+effective_patch_size].mean(axis=(0,1))
                occluded_img[y:y+effective_patch_size, x:x+effective_patch_size] = patch_mean
                
                occluded_pil = Image.fromarray(occluded_img.astype(np.uint8))
                occluded_inputs = processor(images=occluded_pil, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    occluded_output = model(**occluded_inputs)
                    occluded_probs = F.softmax(occluded_output.logits, dim=-1)[0]
                    occluded_disease_score = occluded_probs[target_class].item()
                
                importance = max(0, baseline_disease_score - occluded_disease_score)
                
                disease_importance_map[y:y+effective_patch_size, x:x+effective_patch_size] = np.maximum(
                    disease_importance_map[y:y+effective_patch_size, x:x+effective_patch_size], 
                    importance
                )
    
    if max(image.size) > max_size:
        original_h, original_w = np.array(image).shape[:2]
        disease_importance_map = cv2.resize(disease_importance_map, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    
    return disease_importance_map

def detect_disease_regions_cv(image, disease_type="blight"):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    h, w = img_array.shape[:2]
    disease_mask = np.zeros((h, w), dtype=np.float32)
    
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    if any(keyword in disease_type.lower() for keyword in ['blight', 'spot', 'rust']):
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([25, 255, 150])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 50])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_mask = np.abs(laplacian) > np.percentile(np.abs(laplacian), 85)
        
        disease_regions = brown_mask + yellow_mask + dark_mask
        disease_regions = np.clip(disease_regions, 0, 255).astype(np.uint8)
        
        texture_weight = texture_mask.astype(np.float32) * 0.3
        disease_mask = disease_regions.astype(np.float32) / 255.0 + texture_weight
        
    elif 'mosaic' in disease_type.lower():
        b, g, r = cv2.split(img_array)
        
        kernel = np.ones((15, 15), np.float32) / 225
        b_var = cv2.filter2D(b.astype(np.float32), -1, kernel)
        g_var = cv2.filter2D(g.astype(np.float32), -1, kernel)
        r_var = cv2.filter2D(r.astype(np.float32), -1, kernel)
        
        color_variation = np.sqrt((b - b_var)**2 + (g - g_var)**2 + (r - r_var)**2)
        disease_mask = color_variation / color_variation.max()
        
    elif any(keyword in disease_type.lower() for keyword in ['powdery', 'downy']):
        light_mask = hsv[:,:,2] > 200
        not_green = (hsv[:,:,0] < 35) | (hsv[:,:,0] > 85)
        
        mildew_mask = light_mask & not_green
        disease_mask = mildew_mask.astype(np.float32)
        
    else:
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        normal_green = cv2.inRange(hsv, green_lower, green_upper)
        
        anomaly_mask = 255 - normal_green
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_anomaly = np.abs(laplacian) > np.percentile(np.abs(laplacian), 80)
        
        disease_mask = (anomaly_mask.astype(np.float32) / 255.0) * 0.7 + texture_anomaly.astype(np.float32) * 0.3
    
    disease_mask = cv2.medianBlur(disease_mask.astype(np.float32), 5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
    
    disease_mask = cv2.GaussianBlur(disease_mask, (15, 15), 0)
    
    if disease_mask.max() > 0:
        disease_mask = disease_mask / disease_mask.max()
    
    return disease_mask

def generate_hybrid_disease_map(model, processor, image, target_class=None):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        if target_class is None:
            target_class = outputs.logits.argmax(dim=-1).item()
    
    disease_name = model.config.id2label[target_class].lower()
    
    cv_disease_mask = detect_disease_regions_cv(image, disease_name)
    
    try:
        input_tensor = inputs['pixel_values']
        input_tensor.requires_grad_(True)
        
        outputs = model(input_tensor)
        score = outputs.logits[0, target_class]
        model.zero_grad()
        score.backward()
        
        gradients = input_tensor.grad.detach()
        ai_importance = torch.sum(torch.abs(gradients), dim=1)[0]
        ai_importance = ai_importance.cpu().numpy()
        
        img_h, img_w = np.array(image).shape[:2]
        if ai_importance.shape != (img_h, img_w):
            ai_importance = cv2.resize(ai_importance, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        
        if ai_importance.max() > 0:
            ai_importance = ai_importance / ai_importance.max()
        
        combined_mask = cv_disease_mask * 0.7 + ai_importance * 0.3
        
        agreement_boost = cv_disease_mask * ai_importance * 0.5
        combined_mask = combined_mask + agreement_boost
        
    except Exception as e:
        print(f"AI guidance failed, using CV only: {e}")
        combined_mask = cv_disease_mask
    
    return combined_mask

def create_heatmap_overlay(original_image, heatmap_data, alpha=0.5):
    if heatmap_data is None:
        return original_image
    
    img_array = np.array(original_image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    h, w = img_array.shape[:2]
    
    if heatmap_data.shape != (h, w):
        heatmap_data = cv2.resize(heatmap_data.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    
    heatmap_min, heatmap_max = heatmap_data.min(), heatmap_data.max()
    if heatmap_max > heatmap_min:
        heatmap_norm = (heatmap_data - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_data)
    
    heatmap_smooth = cv2.GaussianBlur(heatmap_norm, (21, 21), 0)
    
    colored_heatmap = cm.jet(heatmap_smooth)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(img_array.astype(np.uint8), 1-alpha, colored_heatmap, alpha, 0)
    
    return Image.fromarray(overlay)

def analyze_predictions(preds):
    top_label, top_conf = preds[0]
    has_disease = any(k in top_label.lower() for k in disease_keywords)
    has_healthy = any(k in top_label.lower() for k in healthy_keywords)
    
    if has_disease:
        status = "🔴 Disease Detected"
        recommendation = (
            "⚠️ Potential plant health issue identified.\n"
            "• Consult with an agricultural expert\n"
            "• Isolate the plant if possible\n"
            "• Monitor nearby plants for symptoms"
        )
    elif has_healthy:
        status = "🟢 Plant appears healthy"
        recommendation = "Continue regular care and monitor plant"
    else:
        status = "🟡 Classification uncertain"
        recommendation = "Manual inspection recommended"
    
    return status, recommendation

def predict(image: Image):
    try:
        print("Starting prediction...")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, 5)
        results = []
        for i in range(5):
            label = model.config.id2label[top_indices[0][i].item()]
            label = label.replace("_", " ")
            conf = top_probs[0][i].item()
            results.append((label, conf))
        
        target_class = top_indices[0][0].item()
        print(f"Top prediction: {results[0][0]} ({results[0][1]*100:.1f}%)")
        
        heatmap_data = None
        
        print("Trying hybrid computer vision + AI method...")
        try:
            heatmap_data = generate_hybrid_disease_map(model, processor, image, target_class)
            print("Hybrid method successful!")
        except Exception as e:
            print(f"Hybrid method failed: {e}")
        
        if heatmap_data is None:
            print("Trying pure computer vision disease detection...")
            try:
                disease_name = model.config.id2label[target_class].lower()
                heatmap_data = detect_disease_regions_cv(image, disease_name)
                print("Computer vision method successful!")
            except Exception as e:
                print(f"Computer vision method failed: {e}")
        
        if heatmap_data is None and min(image.size) < 250:
            print("Trying optimized disease-focused mapping...")
            try:
                heatmap_data = generate_disease_focused_map(model, processor, image, target_class)
                print("Disease-focused mapping successful!")
            except Exception as e:
                print(f"Disease-focused mapping failed: {e}")
        
        if heatmap_data is None:
            print("Trying simple gradients as fallback...")
            try:
                heatmap_data = generate_simple_gradients(model, processor, image, target_class)
                print("Simple gradients successful!")
            except Exception as e:
                print(f"Simple gradients failed: {e}")
        
        if heatmap_data is not None:
            heatmap_image = create_heatmap_overlay(image, heatmap_data)
        else:
            print("All methods failed, using original image")
            heatmap_image = image
        
        status, rec = analyze_predictions(results)
        
        output_text = f"🎯 Status: {status}\n"
        output_text += f"📊 Top Prediction: {results[0][0]} ({results[0][1]*100:.1f}%)\n"
        output_text += f"💡 Recommendation:\n{rec}\n\n"
        output_text += "📋 All Predictions:\n"
        for label, conf in results:
            output_text += f"- {label} ({conf*100:.1f}%)\n"
        
        return output_text, heatmap_image
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        
        try:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs, 5)
            results = []
            for i in range(5):
                label = model.config.id2label[top_indices[0][i].item()]
                label = label.replace("_", " ")
                conf = top_probs[0][i].item()
                results.append((label, conf))
            
            status, rec = analyze_predictions(results)
            
            output_text = f"⚠️ Heatmap generation failed: {str(e)}\n\n"
            output_text += f"🎯 Status: {status}\n"
            output_text += f"📊 Top Prediction: {results[0][0]} ({results[0][1]*100:.1f}%)\n"
            output_text += f"💡 Recommendation:\n{rec}\n\n"
            output_text += "📋 All Predictions:\n"
            for label, conf in results:
                output_text += f"- {label} ({conf*100:.1f}%)\n"
            
            return output_text, image
        except:
            return "Critical error occurred", image

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Analysis Results", lines=20),
        gr.Image(label="Attribution Heatmap", type="pil")
    ],
    title="🌱 Plant Disease Detector",
    description=(
        "Upload a plant image to get:\n"
        "• Disease classification with confidence scores\n"
        "• Attribution heatmap showing which areas influenced the AI's decision\n"
        "• Treatment recommendations\n\n"
        "**Heatmap Colors:**\n"
        "Highlighted areas have more impact on AI decision"
    ),
    theme=gr.themes.Soft(),
    examples=None
)

if __name__ == "__main__":
    iface.launch(share=True, debug=True)