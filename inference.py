import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import requests
import open_clip

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models.causalcrisis_v2 import CausalCrisisV2Model

CLASSES_TASK2 = [
    'affected_individuals', 
    'infrastructure_and_utility_damage', 
    'injured_or_dead_people', 
    'missing_or_found_people', 
    'not_humanitarian', 
    'other_relevant_information', 
    'rescue_volunteering_or_donation_effort', 
    'vehicle_damage'
]

# Hardcoded domain list based on Phase 3 output
DOMAINS = [
    'california_wildfires', 'hurricane_harvey', 'hurricane_irma', 
    'hurricane_maria', 'iraq_iran_earthquake', 'mexico_earthquake', 'srilanka_floods'
]

class CausalCrisisInferenceEngine:
    def __init__(self, device_str='auto'):
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
            
        print(f"[INFO] Initializing inference engine on {self.device}...")
        
        # 1. Load CLIP Model aligned with Training (ViT-B/32 LAION2B)
        print("[INFO] Loading CLIP Processor and Model (open_clip: ViT-B-32, laion2b_s34b_b79k)...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # 2. Load CausalCrisis V2 Backbone
        print("[INFO] Initializing CausalCrisisV2 Backbone...")
        self.model = CausalCrisisV2Model(
            img_dim=512, txt_dim=512,
            hidden_dim=256, causal_dim=256, spurious_dim=256,
            num_domains=len(DOMAINS), num_classes=len(CLASSES_TASK2),
            dropout=0.0 # eval mode
        ).to(self.device)
        
        self.model.eval()
        print("[INFO] Engine ready. Waiting for weights (Optional: you can load state_dict here).")

    def load_weights(self, weights_path):
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[INFO] Successfully loaded model weights from {weights_path}")
        else:
            print(f"[WARNING] Weights file not found: {weights_path}. Using randomly initialized backbone.")

    def _get_clip_embeddings(self, image_path, text):
        # Open image
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
            
        # Process inputs
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(image_input)
            txt_feat = self.clip_model.encode_text(text_input)
            
            # Unit L2 Normalization (Standard for Training)
            img_feat = F.normalize(img_feat, p=2, dim=-1)
            txt_feat = F.normalize(txt_feat, p=2, dim=-1)
            
        return img_feat, txt_feat

    def predict(self, image_path, text):
        img_feat, txt_feat = self._get_clip_embeddings(image_path, text)
        
        with torch.no_grad():
            out = self.model(img_feat, txt_feat)
            
            # Forward classification. Bỏ qua GNN, lấy logits gốc
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            
            # Get Top 3 Predictions
            top3_prob, top3_idx = torch.topk(probs, 3, dim=-1)
            
            results = []
            for i in range(3):
                idx = top3_idx[0, i].item()
                prob = top3_prob[0, i].item() * 100
                label = CLASSES_TASK2[idx]
                results.append((label, prob))
                
            return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Engine for Multimodal Disaster Classification")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the disaster image")
    parser.add_argument("--text", type=str, required=True, help="Text description / tweet content")
    parser.add_argument("--weights", type=str, default="", help="Path to saved CausalCrisis weights (.pth)")
    
    args = parser.parse_args()
    
    engine = CausalCrisisInferenceEngine()
    if args.weights:
        engine.load_weights(args.weights)
        
    print(f"\n[INFERENCE] Analyzing Input...")
    print(f"  IMAGE: {args.image}")
    print(f"  TEXT:  '{args.text}'")
    
    predictions = engine.predict(args.image, args.text)
    
    print(f"\n[RESULTS] Top 3 Categories:")
    for rank, (label, prob) in enumerate(predictions, 1):
        print(f"  #{rank}: {label} ({prob:.2f}%)")
