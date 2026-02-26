# 🔬 NGHIÊN CỨU CHI TIẾT CÁC KỸ THUẬT

## 📚 MỤC LỤC
1. [Text Processing](#1-text-processing)
2. [Image Processing](#2-image-processing)
3. [Attention Mechanisms](#3-attention-mechanisms)
4. [Graph Learning](#4-graph-learning)
5. [Social Context](#5-social-context)
6. [Feature Fusion](#6-feature-fusion)
7. [Optimization](#7-optimization)

---

# 1. TEXT PROCESSING

## 1.1 TF-IDF (Your Pipeline)

**Công thức:**
```
TF(t,d) = (Số lần term t xuất hiện trong d) / (Tổng số terms trong d)
IDF(t) = log(Tổng số documents / Số documents chứa term t)
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

**Ý nghĩa:**
- TF cao: Term quan trọng cho document đó
- IDF cao: Term mang tính phân biệt cao (hiếm trong corpus)
- TF-IDF cao: Vừa quan trọng, vừa có tính phân biệt

**Code:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=1000)
tfidf_matrix = vectorizer.fit_transform(tweets)
bigram_scores = tfidf_matrix.sum(axis=0).A1
top_indices = bigram_scores.argsort()[-10:][::-1]
```

---

## 1.2 BERT Encoding (CrisisSpot)

**Kiến trúc:**
```
Input: "A building is on fire"
   ↓
Tokenize: [CLS] A building is on fire [SEP]
   ↓
Token IDs: [101, 1037, 2311, 2003, 2006, 2543, 102]
   ↓
BERT (12 layers, 768 dims)
   ↓
Output: H_t ∈ ℝ^(128 × 768)
```

**Trong CrisisSpot:**
- Max sequence length: 128 tokens
- Embedding dimension: 768
- Output: Word-level features cho IDEA attention

---

## 1.3 BigBird (Your Pipeline)

**Problem:** Standard attention = O(n²)
- n = 4096 → 16.7M attention scores → Memory explosion!

**Solution: Sparse Attention**
```
BigBird = Random + Window + Global

1. RANDOM (r tokens): Captures long-range dependencies
2. WINDOW (w tokens): Captures local context  
3. GLOBAL (g tokens): [CLS], [SEP] attend to ALL
```

**Complexity:** O(n × (r + w + g)) = O(n)

**Visualization:**
```
Full Attention:          BigBird Sparse:
■ ■ ■ ■ ■ ■ ■ ■         ■ ■ ■ · · ■ · ■
■ ■ ■ ■ ■ ■ ■ ■         ■ ■ ■ ■ · · ■ ·
■ ■ ■ ■ ■ ■ ■ ■         ■ ■ ■ ■ ■ · · ·
■ ■ ■ ■ ■ ■ ■ ■         · ■ ■ ■ ■ ■ · ■
(64 scores)              (~24 scores, 62% reduction)
```

---

# 2. IMAGE PROCESSING

## 2.1 ResNet50 (CrisisSpot)

**Residual Connection:**
```
Standard: y = F(x)
Residual: y = F(x) + x  ← Skip connection giúp gradient flow
```

**Trong CrisisSpot:**
```
Image (224×224×3)
    ↓
ResNet50 (đến conv4_block6)
    ↓
Custom Conv (4×4) → (11×11×1024)
    ↓
Reshape → (121×1024)
    ↓
Padding → (128×1024)  ← Match với BERT output
```

---

## 2.2 CLIP (Your Pipeline + CrisisSpot)

**Core Idea: Contrastive Learning**
```
Text: "A dog playing in park"    Image: 🐕
         ↓                           ↓
    Text Encoder               Image Encoder
         ↓                           ↓
      [512d]         ←───→         [512d]
              Cosine Similarity
```

**Training:** Maximize sim(Tᵢ, Iᵢ), Minimize sim(Tᵢ, Iⱼ) for i≠j

**Application:**
```python
text_features = clip.encode_text(text)
image_features = clip.encode_image(images)
similarities = text_features @ image_features.T
top_images = similarities.argsort()[-k:]
```

---

## 2.3 BLIP-2 (Your Pipeline)

**Architecture: Q-Former Bridge**
```
Image → Frozen ViT-G → Q-Former → Frozen LLM (OPT-6.7B) → Caption
                         ↑
              32 Learnable Queries
              "Question" the image
```

**Key Insight:** Q-Former là "bridge" giữa vision và language
- Chỉ train Q-Former, freeze cả Image Encoder và LLM
- 4-bit quantization để fit Colab Free

---

# 3. ATTENTION MECHANISMS

## 3.1 Standard Self-Attention

**Formula:**
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Q = X × W_Q  (Query: "What am I looking for?")
K = X × W_K  (Key: "What do I contain?")
V = X × W_V  (Value: "What information do I have?")
```

**Problem:** Attention noise - many non-zero values → information diluted

---

## 3.2 IDEA Attention (CrisisSpot)

**5 Steps:**

### Step 1: Shared Embedding Space
```
H_Text = tanh(BatchNorm(H_t × W_t + b_t))  ∈ ℝ^(128 × 1024)
H_Vis  = tanh(BatchNorm(H_v × W_v + b_v))  ∈ ℝ^(128 × 1024)
```

### Step 2: Similarity Matrix
```
Ŝ_sim[i,k] = tanh(H_Text[i,k] + H_Vis[i,k])
S_sim = Ŝ_sim × W_sim  ∈ ℝ^(128 × 128)
```

### Step 3: Harmonious Attention (T=1.65, softer)
```
Att_HAM = softmax(S_sim / 1.65)
T_HAM = Att_HAM × H_t      ← Visual-enriched text
V_HAM = Att_HAM^T × H_v    ← Text-enriched image
```

### Step 4: Contrary Attention (T=0.75, sharper)
```
Inv_sim = 1 - S_sim        ← INVERTED similarity!
Att_CAM = softmax(Inv_sim / 0.75)
T_CAM = Att_CAM × H_t      ← Disagreement features
```

### Step 5: Fusion
```
Output = [T_HAM; V_HAM; T_CAM; V_CAM]
       = [Agreement, Agreement, Conflict, Conflict]
```

**Code:**
```python
class IDEAAttention(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, shared_dim=1024):
        self.T_HAM = 1.65  # Harmonious
        self.T_CAM = 0.75  # Contrary
        
    def forward(self, H_t, H_v):
        # Project to shared space
        H_Text = torch.tanh(self.text_bn(self.text_proj(H_t)))
        H_Vis = torch.tanh(self.image_bn(self.image_proj(H_v)))
        
        # Similarity matrix
        S_sim = self.sim_proj(torch.tanh(H_Text + H_Vis))
        
        # HAM (T=1.65)
        Att_HAM = F.softmax(S_sim / 1.65, dim=-1)
        T_HAM = Att_HAM @ H_t
        
        # CAM (T=0.75, inverted)
        Att_CAM = F.softmax((1 - S_sim) / 0.75, dim=-1)
        T_CAM = Att_CAM @ H_t
        
        return torch.cat([T_HAM, V_HAM, T_CAM, V_CAM], dim=-1)
```

---

## 3.3 Differential Attention (Diff Transformer)

**Core Idea: Noise Cancellation via Subtraction**
```
Analogy: Noise-canceling headphones 🎧
Signal + Noise₁ - λ(Signal + Noise₂) ≈ Signal (when noises similar)
```

**Formula:**
```
[Q₁, Q₂] = split(X × W_Q)
[K₁, K₂] = split(X × W_K)

A₁ = softmax(Q₁ × K₁^T / √d)
A₂ = softmax(Q₂ × K₂^T / √d)

DiffAttn(X) = (A₁ - λ × A₂) × V
```

**Lambda Initialization:**
```
λ_init = 0.8 - 0.6 × exp(-0.3 × (layer - 1))

Layer 1:  λ ≈ 0.20 → Less subtraction, capture more
Layer 12: λ ≈ 0.75 → More subtraction, focus on key info
```

**Visualization:**
```
A₁:                -  λ × A₂:               = Sparse Result:
0.3 0.1 0.2 0.4       0.2 0.1 0.3 0.4         ▓▓░░░░▓░
0.1 0.5 0.2 0.2       0.1 0.2 0.3 0.4         ░░████░░
                                               (Focused!)
```

**Code:**
```python
class DifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads, layer_idx):
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        
    def forward(self, x):
        Q1, Q2 = self.W_Q(x).chunk(2, dim=-1)
        K1, K2 = self.W_K(x).chunk(2, dim=-1)
        
        A1 = F.softmax(Q1 @ K1.T / scale, dim=-1)
        A2 = F.softmax(Q2 @ K2.T / scale, dim=-1)
        
        lambda_val = compute_lambda() + self.lambda_init
        A_diff = A1 - lambda_val * A2
        
        return A_diff @ V
```

---

# 4. GRAPH LEARNING

## 4.1 GraphSAGE (CrisisSpot)

**Problem:** Traditional GNN requires all nodes → Not scalable

**Solution: Sample & Aggregate**
```
For each layer k:
    For each node v:
        1. Sample neighbors: N(v) = sample(neighbors(v))
        2. Aggregate: h_N = AGGREGATE({h_u : u ∈ N(v)})
        3. Update: h_v^k = σ(W · [h_v^{k-1}; h_N])
```

**Trong CrisisSpot:**
```
Graph Construction:
- Nodes: Tweet embeddings (CLIP)
- Edges: cos_sim(tweet_i, tweet_j) > 0.75

Message Passing:
Tweet₁ ──┐
Tweet₂ ──┼──→ Aggregate ──→ Update Tweet₃
Tweet₄ ──┘

Similar tweets reinforce each other!
```

**Code:**
```python
from torch_geometric.nn import SAGEConv

def build_graph(embeddings, threshold=0.75):
    sim = embeddings @ embeddings.T
    edges = (sim > threshold).nonzero()
    return edges[edges[:,0] != edges[:,1]].T

class GraphSAGE(nn.Module):
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x
```

---

# 5. SOCIAL CONTEXT

## 5.1 VADER Sentiment (CrisisSpot)

**Output:** [positive, negative, neutral, compound] → 4 dims

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(text)
# {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.6}
```

## 5.2 EmoLex Emotions (CrisisSpot)

**11 Emotion Categories:**
anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust, valence

## 5.3 Crisis Lexicon (CrisisSpot)

**4,268 crisis-related terms curated from CrisisLex**

```python
crisis_keywords = ['earthquake', 'flood', 'fire', 'explosion', ...]
crisis_score = sum(1 for word in text if word in crisis_keywords)
```

## 5.4 User Engagement (CrisisSpot)

**5 Features:**
- likes, retweets, replies, followers, friends

## 5.5 User Credibility

**Historical informativeness score**

---

# 6. FEATURE FUSION

## 6.1 CrisisSpot MFN

```
MALN: [T_HAM; V_HAM; T_CAM; V_CAM] → Dense(512) → 128d
SHLN: 21-dim social features → Dense(32) → 8d  
GFLN: GraphSAGE output → Mean pool → 64d

F_joint = [F_MALN; F_SHLN; F_GFLN] (200d)
→ Dense(64) → Dense(2) → Softmax
```

## 6.2 Your Pipeline (Late Fusion)

```
TEXT: Tweets → TF-IDF → BigBird → Summary

IMAGE: Images → CLIP → (wait for summary) → Top 10
       Top 10 → BLIP-2 → Captions → CLIP → Top 4

Final: [Text Summary] + [4 Images]
```

**Pros:** Flexible, interpretable, robust
**Cons:** Misses early cross-modal interactions

---

# 7. OPTIMIZATION

## 7.1 Quantization (Your Pipeline)

```python
# 4-bit quantization for BLIP-2
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
# 6.7B → ~1.7GB
```

## 7.2 Temperature Scaling (CrisisSpot)

```
T > 1: Softer distribution (more exploration)
T < 1: Sharper distribution (more exploitation)

HAM: T=1.65 (explore agreements)
CAM: T=0.75 (focus on contradictions)
```

## 7.3 Lambda Scheduling (Diff Transformer)

```
Early layers: λ small → Capture more info
Later layers: λ large → More noise cancellation
```

---

# 📊 COMPARISON TABLE

| Technique | MEASF | CrisisSpot | Diff Transformer |
|-----------|-------|------------|------------------|
| Text Encoder | BigBird | BERT | - |
| Image Encoder | CLIP | ResNet50+CLIP | - |
| Attention | CLIP cosine | IDEA (HAM+CAM) | Differential (A₁-λA₂) |
| Temperature | None | 1.65/0.75 | None |
| Lambda | None | None | Learnable |
| Graph | None | GraphSAGE | None |
| Social | None | 21 features | None |
| Fusion | Late | Concat | - |
| Task | Summarization | Classification | Language Model |

---

# 💡 KEY TAKEAWAYS

1. **IDEA:** Contrary attention reveals multimodal conflicts
2. **Diff Transformer:** Subtraction cancels common-mode noise
3. **GraphSAGE:** Similar samples reinforce each other
4. **Temperature:** Controls exploration vs exploitation
5. **Lambda:** Adaptive noise cancellation per layer
6. **Social Context:** User credibility matters for crisis data
