# 🔬 CrisisSpot Deep Research Document

**Paper:** A Social Context-aware Graph-based Multimodal Attentive Learning Framework for Disaster Content Classification during Emergencies

**Authors:** Shahid et al.

**Source:** https://arxiv.org/abs/2410.08814

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [IDEA Attention Mechanism](#3-idea-attention-mechanism)
4. [Multimodal Graph Learning](#4-multimodal-graph-learning)
5. [Social Context Features](#5-social-context-features)
6. [Multimodal Fusion Network](#6-multimodal-fusion-network)
7. [Implementation Details](#7-implementation-details)
8. [Experimental Results](#8-experimental-results)
9. [Code Implementation](#9-code-implementation)
10. [Potential Improvements for MEASF](#10-potential-improvements-for-measf)

---

## 1. Overview

### 1.1 Problem Statement

CrisisSpot giải quyết bài toán **Disaster Content Classification**:
- **Input:** Tweet multimodal (text + image)
- **Output:** Classification labels
  - **Task 1 (Binary):** Informative / Non-informative
  - **Task 2 (Multi-class):** Humanitarian categories

### 1.2 Key Innovations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CRISISSPOT KEY INNOVATIONS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1️⃣ INVERTED DUAL EMBEDDED ATTENTION (IDEA)                                │
│     ├── Harmonious Attention (HAM): Tìm sự ĐỒNG THUẬN                       │
│     └── Contrary Attention (CAM): Tìm sự MÂU THUẪN                          │
│                                                                             │
│  2️⃣ MULTIMODAL GRAPH LEARNING                                              │
│     ├── CLIP-based multimodal encoding                                      │
│     ├── Cosine similarity graphs (threshold ≥ 0.75)                         │
│     └── GraphSAGE for knowledge propagation                                 │
│                                                                             │
│  3️⃣ SOCIAL CONTEXT FEATURES (21 dimensions)                                │
│     ├── Text Semantic Profiling (TSP): 15 dims                              │
│     │   ├── EmoQuotient (EmoLex): 11 emotions                               │
│     │   ├── SentiQuotient (VADER): 3 sentiments                             │
│     │   └── Crisis Informative Score (CIS): 1 dim                           │
│     └── Social Interaction Metrics (SIM): 6 dims                            │
│         ├── User Engagement Metrics (UEM): 5 dims                           │
│         └── User Informative Score (UIS): 1 dim                             │
│                                                                             │
│  4️⃣ MULTIMODAL FUSION NETWORK (MFN)                                        │
│     ├── MALN: Multimodal features → 128d                                    │
│     ├── SHLN: Social context → 8d                                           │
│     ├── GFLN: Graph features → variable                                     │
│     └── JFLN: Joint fusion → 64d → Prediction                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Architecture Diagram

```
                              ┌─────────────────────────────────────┐
                              │           INPUT TWEET               │
                              │    (Text + Image + Metadata)        │
                              └──────────────┬──────────────────────┘
                                             │
                   ┌─────────────────────────┼─────────────────────────┐
                   │                         │                         │
                   ▼                         ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
        │  TEXT ENCODER    │      │  IMAGE ENCODER   │      │  SOCIAL CONTEXT  │
        │     (BERT)       │      │   (ResNet50)     │      │    EXTRACTOR     │
        │   d_t = 768      │      │   d_v = 1024     │      │     21 dims      │
        └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
                 │                         │                         │
                 │    ┌────────────────────┴────────────────────┐    │
                 │    │                                         │    │
                 ▼    ▼                                         ▼    │
        ┌────────────────────────────────────────────────────────┐   │
        │             SHARED EMBEDDING SPACE (d_se = 1024)       │   │
        │    H_Text = tanh(BatchNorm(H_t × W_t + b_t))          │   │
        │    H_Vis = tanh(BatchNorm(H_v × W_v + b_v))           │   │
        └─────────────────────────┬──────────────────────────────┘   │
                                  │                                   │
                                  ▼                                   │
        ┌───────────────────────────────────────────────────────┐    │
        │                    IDEA ATTENTION                      │    │
        │  ┌─────────────────┐      ┌─────────────────┐         │    │
        │  │      HAM        │      │      CAM        │         │    │
        │  │  T=1.65 (soft)  │      │  T=0.75 (sharp) │         │    │
        │  │  ALIGNMENT      │      │  CONTRADICTION  │         │    │
        │  └────────┬────────┘      └────────┬────────┘         │    │
        │           │                        │                   │    │
        │           └───────────┬────────────┘                   │    │
        │                       ▼                                │    │
        │              Fused Attended Vector                     │    │
        │                  (FAV: 1792d)                          │    │
        └───────────────────────┬───────────────────────────────┘    │
                                │                                     │
     ┌──────────────────────────┼─────────────────────────────────────┤
     │                          │                                     │
     ▼                          ▼                                     ▼
┌──────────────┐     ┌──────────────────┐              ┌──────────────────┐
│ CLIP ENCODER │     │      MALN        │              │      SHLN        │
│  (Text+Img)  │     │  1792→1024→512   │              │    21→16→8       │
│    512d      │     │  →256→128        │              │                  │
└──────┬───────┘     └────────┬─────────┘              └────────┬─────────┘
       │                      │                                  │
       ▼                      │                                  │
┌──────────────┐              │                                  │
│   GRAPH      │              │                                  │
│  LEARNING    │              │                                  │
│ (GraphSAGE)  │              │                                  │
│   GFLN       │              │                                  │
└──────┬───────┘              │                                  │
       │                      │                                  │
       └──────────────────────┴──────────────────────────────────┘
                                       │
                                       ▼
                          ┌───────────────────────────┐
                          │    JOINT FUSION (JFLN)    │
                          │   Concat → 264→256→128→64 │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    PREDICTION LAYER       │
                          │  Binary: Sigmoid + BCE    │
                          │  Multi: Softmax + CCE     │
                          └───────────────────────────┘
```

---

## 2. Architecture Deep Dive

### 2.1 Feature Extraction

#### 2.1.1 Text Feature Extraction (BERT)

```python
# Configuration
max_sequence_length = 128
embedding_dimension = 768  # d_t

# Process
S_t = {T_1, T_2, ..., T_n}  # Token sequence
H_t = BERT_tokenizer(S_t)   # H_t ∈ ℝ^{d × d_t} = ℝ^{128 × 768}
```

**Key Details:**
- Sử dụng BERT-based tokenization
- Extract word-level features
- Output shape: `(batch, 128, 768)`

#### 2.1.2 Visual Feature Extraction (ResNet50)

```python
# Configuration
target_layer = 'conv4_block6_3_conv'  # Để match với text dimension
custom_conv_filters = 1024
kernel_size = (4, 4)

# Process Flow
Input Image → ResNet50 (frozen up to conv4_block6_3_conv)
            → F_emb (feature map)
            → Custom Conv (1024 filters, 4×4 kernel)
            → Reshape to (1, 121, 1024)
            → Padding to (1, 128, 1024)
            → H_v ∈ ℝ^{128 × 1024}
```

**Key Insight:** Paper chọn layer `conv4_block6_3_conv` thay vì layer cuối cùng để:
1. Capture complex representations
2. Align shape với text encoder (128 tokens)
3. Preserve hierarchical visual information

---

## 3. IDEA Attention Mechanism (Chi Tiết)

### 3.1 Shared Embedding Module

**Purpose:** Project cả text và image vào cùng một không gian embedding để có thể tính similarity.

```
Textual Modality Transformation:
────────────────────────────────
H_Text = tanh(BatchNorm(H_t × W_t + b_t))

Where:
- H_t ∈ ℝ^{d × d_t} = ℝ^{128 × 768}
- W_t ∈ ℝ^{d_t × d_se} = ℝ^{768 × 1024}
- b_t ∈ ℝ^{1 × d_se} = ℝ^{1 × 1024}
- H_Text ∈ ℝ^{d × d_se} = ℝ^{128 × 1024}

Visual Modality Transformation:
────────────────────────────────
H_Vis = tanh(BatchNorm(H_v × W_v + b_v))

Where:
- H_v ∈ ℝ^{d × d_v} = ℝ^{128 × 1024}
- W_v ∈ ℝ^{d_v × d_se} = ℝ^{1024 × 1024}
- b_v ∈ ℝ^{1 × d_se} = ℝ^{1 × 1024}
- H_Vis ∈ ℝ^{d × d_se} = ℝ^{128 × 1024}
```

### 3.2 Similarity Matrix Computation

```
Step 1: Compute intermediate similarity
─────────────────────────────────────────
Ŝ_sim[i,k] = tanh(H_Text[i,m] + H_Vis[i,m])

Step 2: Learnable transformation
─────────────────────────────────────────
S_sim[i,j] = Ŝ_sim × W_sim

Where:
- W_sim ∈ ℝ^{d_se × d} = ℝ^{1024 × 128}
- S_sim ∈ ℝ^{d × d} = ℝ^{128 × 128}

Interpretation:
- Row i: Similarity between i-th visual feature and ALL text features
- Column j: Similarity between j-th text feature and ALL visual features
```

### 3.3 Harmonious Attention Module (HAM)

**Purpose:** Identify và amplify areas where modalities ALIGN/AGREE.

```python
# Temperature parameter (CRITICAL!)
T_HAM = 1.65  # Higher = softer, more distributed attention

# Attention weights
Att_weights = softmax(S_sim / T_HAM, axis=0)  # Along dimension i

# Visual-enriched textual features
T_HAM = Att_weights @ H_t  # T_HAM ∈ ℝ^{d × d_t}

# Text-enriched visual features
V_HAM = Att_weights @ H_v  # V_HAM ∈ ℝ^{d × d_v}
```

**Temperature Effect:**
```
T = 1.65 (HAM - High Temperature):
─────────────────────────────────
[0.20, 0.18, 0.22, 0.15, 0.25, ...]  ← More uniform distribution
                                       → Captures broader alignment

T = 0.75 (CAM - Low Temperature):
─────────────────────────────────
[0.05, 0.02, 0.85, 0.03, 0.05, ...]  ← Peaked distribution
                                       → Focuses on specific discrepancies
```

### 3.4 Contrary Attention Module (CAM)

**Purpose:** Identify và amplify areas where modalities DISAGREE/CONTRADICT.

**Key Insight:** Multiply by `-1` to INVERT the embeddings, then apply attention!

```python
# Temperature parameter
T_CAM = 0.75  # Lower = sharper, more focused attention

# INVERT the similarity matrix!
Inverted_sim = -1 * S_sim  # Flip the direction in vector space

# Attention weights on inverted similarities
Att_weights_CAM = softmax(Inverted_sim / T_CAM, axis=0)

# Inverted visual-enriched textual features
T_CAM = Att_weights_CAM @ H_t

# Inverted text-enriched visual features  
V_CAM = Att_weights_CAM @ H_v
```

**Why Contrary Information Matters:**

```
Example 1: Disaster Detection
─────────────────────────────
Image: 🏚️ Collapsed building, smoke rising
Text:  "Everything is fine here, just a normal day"

HAM: Low attention (no alignment)
CAM: HIGH attention (detects contradiction!)
→ Flag as potentially UNRELIABLE content

Example 2: Misinformation Detection
───────────────────────────────────
Image: 📷 Old photo from different disaster
Text:  "Breaking: Major earthquake hits today"

HAM: Some alignment (both mention disaster)
CAM: HIGH attention (temporal inconsistency detected)
→ Flag for fact-checking
```

### 3.5 Attended Fusion Module

```python
# Combine HAM and CAM outputs

# Visual side fusion
V_fuse = (V_HAM ⊕ V_CAM) × W_vis + b_vis  # ⊕ = concatenation
# V_fuse ∈ ℝ^{d × d_v}

# Textual side fusion
T_fuse = (T_HAM ⊕ T_CAM) × W_text + b_text
# T_fuse ∈ ℝ^{d × d_t}

# Visual-fusion feature
V_fusion = (T_fuse ⊕ H_v) × W̃_v + b̃_v
# V_fusion ∈ ℝ^{d × d_v}

# Textual-fusion feature
T_fusion = (V_fuse ⊕ H_t) × W̃_t + b̃_t
# T_fusion ∈ ℝ^{d × d_t}

# Global Average Pooling
T_fusion_pooled = GlobalAvgPool(T_fusion)  # ∈ ℝ^{1 × d_t}
V_fusion_pooled = GlobalAvgPool(V_fusion)  # ∈ ℝ^{1 × d_v}

# Self-attention for final representation
e_i = (T_fusion × W_Q) × (T_fusion × W_K)^T / √d_z  # Attention score
α_i = softmax(e_i)  # Attention weight
T_final = α_i × (T_fusion × W_V)  # ∈ ℝ^{1 × d_t}

# Similarly for V_final
V_final ∈ ℝ^{1 × d_v}

# Fused Attended Vector (FAV)
FAV = T_final ⊕ V_final  # ∈ ℝ^{1 × (d_t + d_v)} = ℝ^{1 × 1792}
```

---

## 4. Multimodal Graph Learning

### 4.1 Multimodal Encoder (CLIP)

```python
# Use CLIP for shared multimodal embeddings
text_embedding = CLIP.encode_text(text)   # ∈ ℝ^{512}
image_embedding = CLIP.encode_image(img)  # ∈ ℝ^{512}

# Contrastive learning aligns these in shared semantic space
```

### 4.2 Graph Construction

```python
def build_similarity_graph(H, threshold=0.75):
    """
    Build adjacency matrix based on cosine similarity.
    
    Args:
        H: Feature matrix (n_samples × dim)
        threshold: Minimum similarity to create edge
    
    Returns:
        A: Adjacency matrix (n_samples × n_samples)
    """
    n = len(H)
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Cosine similarity
            S_ij = cosine_similarity(H[i], H[j])
            
            if S_ij >= threshold:
                A[i, j] = 1
                A[j, i] = 1  # Undirected edge
    
    return A

# Create separate graphs for text and images
A_text = build_similarity_graph(H_text, threshold=0.75)
A_image = build_similarity_graph(H_image, threshold=0.75)
```

### 4.3 GraphSAGE Knowledge Propagation

```python
def graphsage_propagation(H, A, K=3):
    """
    GraphSAGE message passing.
    
    Args:
        H: Node embeddings (n × d)
        A: Adjacency matrix (n × n)
        K: Number of layers/depth
    
    Returns:
        Updated node embeddings
    """
    h = H  # Initialize with input features
    
    for k in range(K):
        h_new = []
        for v in range(len(H)):
            # Get neighbors from adjacency matrix
            neighbors = np.where(A[v] == 1)[0]
            
            if len(neighbors) > 0:
                # AGGREGATE: Mean of neighbor embeddings
                h_N = np.mean([h[u] for u in neighbors], axis=0)
            else:
                h_N = np.zeros_like(h[v])
            
            # CONCAT and TRANSFORM
            h_concat = np.concatenate([h[v], h_N])
            h_v_new = relu(W[k] @ h_concat)  # W[k] is learnable
            
            # NORMALIZE
            h_v_new = h_v_new / np.linalg.norm(h_v_new)
            
            h_new.append(h_v_new)
        
        h = np.array(h_new)
    
    return h

# Apply to both text and image graphs
H_text_updated = graphsage_propagation(H_text, A_text, K=3)
H_image_updated = graphsage_propagation(H_image, A_image, K=3)
```

**Why GraphSAGE Matters:**

```
Before Graph Propagation:
─────────────────────────
Tweet A: "Building collapsed" [score: 0.7]
Tweet B: "Rescue underway"    [score: 0.6]
Tweet C: "Many trapped"       [score: 0.65]
(Similar tweets classified independently)

After Graph Propagation:
────────────────────────
Tweet A: [score: 0.85]  ┐
Tweet B: [score: 0.80]  ├── Similar tweets REINFORCE each other!
Tweet C: [score: 0.82]  ┘

KEY INSIGHT: Collective wisdom > Individual classification
```

---

## 5. Social Context Features (21 Dimensions)

### 5.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SOCIAL CONTEXT FEATURES (21 dims)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📝 TEXT SEMANTIC PROFILING (TSP) - 15 dimensions                           │
│  ════════════════════════════════════════════════                           │
│                                                                             │
│  1. SentiQuotient (VADER) - 3 dims                                          │
│     ├── positive_score                                                      │
│     ├── negative_score                                                      │
│     └── neutral_score                                                       │
│                                                                             │
│  2. EmoQuotient (EmoLex) - 11 dims                                          │
│     ├── anger, anticipation, disgust, fear                                  │
│     ├── joy, sadness, surprise, trust                                       │
│     └── positive, negative, compound                                        │
│                                                                             │
│  3. Crisis Informative Score (CIS) - 1 dim                                  │
│     └── Proportion of crisis-related terms in text                          │
│         (From lexicon of 4,268 crisis terms)                                │
│                                                                             │
│  👥 SOCIAL INTERACTION METRICS (SIM) - 6 dimensions                         │
│  ═════════════════════════════════════════════════                          │
│                                                                             │
│  4. User Engagement Metrics (UEM) - 5 dims                                  │
│     ├── favorites_count (likes)                                             │
│     ├── retweets_count                                                      │
│     ├── followers_count                                                     │
│     ├── friends_count                                                       │
│     └── statuses_count                                                      │
│                                                                             │
│  5. User Informative Score (UIS) - 1 dim                                    │
│     └── Historical ratio of informative tweets                              │
│         UIS = (N_informative - N_non_informative) / N_total                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 SentiQuotient Analysis (VADER)

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiquotient(text):
    """
    Extract sentiment scores using VADER.
    
    Returns:
        [positive, negative, neutral] - 3 dimensions
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    return [
        scores['pos'],  # Positive sentiment
        scores['neg'],  # Negative sentiment
        scores['neu']   # Neutral sentiment
    ]

# Example
text = "Devastating earthquake destroys buildings, rescue teams arriving"
# Output: [0.15, 0.42, 0.43]
```

### 5.3 EmoQuotient Analysis (EmoLex/NRC)

```python
from nrclex import NRCLex

def get_emoquotient(text):
    """
    Extract emotion scores using NRC Emotion Lexicon.
    
    Returns:
        11 emotion dimensions
    """
    emotion = NRCLex(text)
    
    emotions = [
        'anger', 'anticipation', 'disgust', 'fear',
        'joy', 'sadness', 'surprise', 'trust',
        'positive', 'negative'
    ]
    
    freq = emotion.raw_emotion_scores
    total_words = len(text.split())
    
    # Normalize by text length
    scores = [freq.get(e, 0) / total_words for e in emotions]
    
    # Add compound score
    scores.append(sum(scores[:8]) / 8)  # Average of 8 basic emotions
    
    return scores  # 11 dimensions

# Example
text = "Devastating earthquake destroys buildings"
# Output: [0.0, 0.1, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0, 0.1, 0.7, 0.19]
```

### 5.4 Crisis Informative Score (CIS)

```python
# Crisis lexicon with 4,268 terms
CRISIS_LEXICON = {
    'earthquake', 'flood', 'tsunami', 'hurricane',
    'rescue', 'evacuate', 'emergency', 'disaster',
    'victims', 'casualties', 'destroyed', 'collapsed',
    'relief', 'aid', 'donation', 'shelter',
    # ... 4,268 terms total
}

def get_crisis_informative_score(text, all_texts):
    """
    Calculate Crisis Informative Score for a tweet.
    
    Args:
        text: Single tweet
        all_texts: All tweets in dataset (for normalization)
    
    Returns:
        Normalized CIS score
    """
    words = set(text.lower().split())
    crisis_count = len(words.intersection(CRISIS_LEXICON))
    
    # Calculate for all texts to normalize
    all_counts = [
        len(set(t.lower().split()).intersection(CRISIS_LEXICON))
        for t in all_texts
    ]
    
    # Min-max normalization
    min_c, max_c = min(all_counts), max(all_counts)
    normalized = (crisis_count - min_c) / (max_c - min_c + 1e-8)
    
    return normalized

# Example
text = "Earthquake destroys buildings, rescue teams evacuate victims"
# CIS = 5/7 words match → normalized to dataset
```

### 5.5 User Engagement Metrics (UEM)

```python
def get_user_engagement_metrics(tweet_metadata):
    """
    Extract and normalize user engagement features.
    
    Returns:
        5 normalized dimensions
    """
    metrics = [
        tweet_metadata['favorite_count'],    # Likes
        tweet_metadata['retweet_count'],     # Retweets
        tweet_metadata['user']['followers_count'],
        tweet_metadata['user']['friends_count'],
        tweet_metadata['user']['statuses_count']
    ]
    
    # Min-max normalization across dataset
    normalized = min_max_normalize(metrics)
    
    return normalized

# Example output: [0.12, 0.08, 0.45, 0.23, 0.67]
```

### 5.6 User Informative Score (UIS)

```python
def get_user_informative_score(user_id, user_history):
    """
    Calculate user's historical informativeness.
    
    Args:
        user_id: Twitter user ID
        user_history: Dict mapping user_id → list of (tweet_id, is_informative)
    
    Returns:
        UIS score (normalized)
    """
    if user_id not in user_history:
        return 0.5  # Default for unknown users
    
    history = user_history[user_id]
    n_informative = sum(1 for _, is_info in history if is_info)
    n_non_informative = len(history) - n_informative
    n_total = len(history)
    
    uis = (n_informative - n_non_informative) / n_total
    
    # Normalize to [0, 1]
    uis_normalized = (uis + 1) / 2
    
    return uis_normalized

# Example: User posted 100 tweets, 70 informative, 30 non-informative
# UIS = (70 - 30) / 100 = 0.4 → normalized = 0.7
```

### 5.7 Social Holistic Vector Assembly

```python
def create_social_holistic_vector(text, tweet_metadata, user_history):
    """
    Combine all social context features into SHV.
    
    Returns:
        21-dimensional feature vector
    """
    # Text Semantic Profiling (15 dims)
    sentiquotient = get_sentiquotient(text)           # 3 dims
    emoquotient = get_emoquotient(text)               # 11 dims
    cis = get_crisis_informative_score(text)          # 1 dim
    
    # Social Interaction Metrics (6 dims)
    uem = get_user_engagement_metrics(tweet_metadata)  # 5 dims
    uis = get_user_informative_score(
        tweet_metadata['user']['id'],
        user_history
    )                                                  # 1 dim
    
    # Concatenate all
    shv = np.concatenate([
        sentiquotient,   # 3
        emoquotient,     # 11
        [cis],           # 1
        uem,             # 5
        [uis]            # 1
    ])                   # Total: 21 dims
    
    return shv
```

---

## 6. Multimodal Fusion Network (MFN)

### 6.1 MALN - Multimodal Adaptive Learning Network

```python
class MALN(nn.Module):
    """Process Fused Attended Vector from IDEA."""
    
    def __init__(self, input_dim=1792):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, fav):
        # fav: Fused Attended Vector from IDEA
        return self.layers(fav)  # Output: 128 dims
```

### 6.2 SHLN - Social Holistic Learning Network

```python
class SHLN(nn.Module):
    """Process Social Context Features."""
    
    def __init__(self, input_dim=21):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(21, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, shv):
        # shv: Social Holistic Vector (21 dims)
        return self.layers(shv)  # Output: 8 dims
```

### 6.3 GFLN - Graph Feature Learning Network

```python
class GFLN(nn.Module):
    """Process GraphSAGE outputs."""
    
    def __init__(self, input_dim=1024):  # 512 text + 512 image from CLIP
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, graph_features):
        return self.layers(graph_features)  # Output: 128 dims
```

### 6.4 JFLN - Joint Fusion Learning Network

```python
class JFLN(nn.Module):
    """Combine all modality outputs."""
    
    def __init__(self):
        super().__init__()
        # Input: MALN(128) + SHLN(8) + GFLN(128) = 264
        self.layers = nn.Sequential(
            nn.Linear(264, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, maln_out, shln_out, gfln_out):
        concat = torch.cat([maln_out, shln_out, gfln_out], dim=-1)
        return self.layers(concat)  # Output: 64 dims
```

### 6.5 Prediction Layer

```python
class PredictionLayer(nn.Module):
    """Final classification."""
    
    def __init__(self, task='binary'):
        super().__init__()
        self.task = task
        
        if task == 'binary':  # Informative Task
            self.classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:  # Humanitarian Task (multi-class)
            self.classifier = nn.Sequential(
                nn.Linear(64, 7),  # 7 humanitarian categories
                nn.Softmax(dim=-1)
            )
    
    def forward(self, x):
        return self.classifier(x)
    
    def compute_loss(self, pred, target):
        if self.task == 'binary':
            return F.binary_cross_entropy(pred, target)
        else:
            return F.cross_entropy(pred, target)
```

---

## 7. Implementation Details

### 7.1 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | 5e-5 | Adam optimizer |
| **Dropout** | 0.2 | All layers |
| **Batch Size** | 32 | Training |
| **Epochs** | 100 | Maximum |
| **Text Dim (d_t)** | 768 | BERT output |
| **Visual Dim (d_v)** | 1024 | ResNet output |
| **Shared Embed (d_se)** | 1024 | Projection space |
| **HAM Temperature** | 1.65 | Softer attention |
| **CAM Temperature** | 0.75 | Sharper attention |
| **Graph Threshold** | 0.75 | Edge creation |
| **GraphSAGE Layers** | K=2-3 | Propagation depth |

### 7.2 Hardware Requirements

```
- CPU: Intel Xeon Silver 4215R @ 3.20 GHz
- RAM: 256 GB (paper setup) / 16+ GB (practical)
- GPU: NVIDIA Tesla T4 (16 GB VRAM)
- Storage: ~50 GB for models + data
```

### 7.3 Training Pipeline

```python
# Pseudo-code for training loop
for epoch in range(100):
    for batch in dataloader:
        text, image, social_features, label = batch
        
        # 1. Extract base features
        H_t = bert_encoder(text)
        H_v = resnet_encoder(image)
        
        # 2. IDEA Attention
        H_Text = transform_text(H_t)
        H_Vis = transform_visual(H_v)
        S_sim = compute_similarity(H_Text, H_Vis)
        
        T_HAM, V_HAM = harmonious_attention(S_sim, H_t, H_v, T=1.65)
        T_CAM, V_CAM = contrary_attention(S_sim, H_t, H_v, T=0.75)
        FAV = fuse_attended(T_HAM, V_HAM, T_CAM, V_CAM)
        
        # 3. Graph Learning
        clip_text = clip_encoder.encode_text(text)
        clip_image = clip_encoder.encode_image(image)
        A_t = build_graph(clip_text)
        A_i = build_graph(clip_image)
        graph_features = graphsage(clip_text, clip_image, A_t, A_i)
        
        # 4. Fusion
        maln_out = MALN(FAV)
        shln_out = SHLN(social_features)
        gfln_out = GFLN(graph_features)
        joint = JFLN(maln_out, shln_out, gfln_out)
        
        # 5. Prediction
        pred = prediction_layer(joint)
        loss = compute_loss(pred, label)
        
        # 6. Backprop
        loss.backward()
        optimizer.step()
```

---

## 8. Experimental Results

### 8.1 Datasets

| Dataset | Samples | Task | Source |
|---------|---------|------|--------|
| **CrisisMMD** | ~18,000 | Binary + Multi-class | 7 Natural Disasters |
| **TSEqD** | 10,352 | Binary + Multi-class | Turkey-Syria Earthquake 2023 |

### 8.2 Performance Comparison

#### Binary Classification (Informative Task)

| Method | Acc | Prec | Rec | F1 |
|--------|-----|------|-----|-----|
| BERT (text-only) | 86.2 | 84.5 | 85.8 | 85.1 |
| ResNet (image-only) | 78.4 | 76.2 | 77.8 | 77.0 |
| VisualBERT | 88.7 | 87.3 | 88.0 | 87.6 |
| ViLBERT | 89.1 | 88.4 | 88.7 | 88.5 |
| **CrisisSpot** | **91.2** | **90.8** | **91.0** | **90.9** |

#### Multi-class Classification (Humanitarian Task)

| Method | Acc | Prec | Rec | F1 |
|--------|-----|------|-----|-----|
| BERT | 72.4 | 70.1 | 71.3 | 70.7 |
| VisualBERT | 75.8 | 73.5 | 74.2 | 73.8 |
| ViLBERT | 76.1 | 76.3 | 77.2 | 76.7 |
| DMCC | 79.7 | 79.4 | 78.5 | 78.9 |
| **CrisisSpot** | **81.7** | **82.1** | **80.7** | **81.1** |

### 8.3 Ablation Study

| Configuration | F1 (Informative) | F1 (Humanitarian) |
|---------------|------------------|-------------------|
| Full CrisisSpot | **90.9** | **81.1** |
| w/o CAM (only HAM) | 88.4 (-2.5) | 78.6 (-2.5) |
| w/o HAM (only CAM) | 87.1 (-3.8) | 77.3 (-3.8) |
| w/o Graph Learning | 88.7 (-2.2) | 79.2 (-1.9) |
| w/o Social Context | 89.3 (-1.6) | 79.8 (-1.3) |
| w/o IDEA + Graph | 86.5 (-4.4) | 76.4 (-4.7) |

**Key Insights:**
1. **IDEA (HAM+CAM) contributes most** to performance
2. **Contrary attention (CAM) is crucial** - removing it hurts more than removing HAM
3. **Graph learning adds ~2% F1** across tasks
4. **Social context adds ~1.5% F1** - modest but consistent

### 8.4 TSEqD Dataset Results (Turkey-Syria Earthquake 2023)

#### Informative Task

| Method | Acc | Prec | Rec | F1 |
|--------|-----|------|-----|-----|
| BERT | 74.47 | 75.12 | 85.07 | 79.79 |
| ResNet | 68.42 | 70.91 | 75.20 | 72.99 |
| VGG | 67.14 | 60.85 | 73.12 | 66.42 |
| SCBD | 74.73 | 76.08 | 85.97 | 80.72 |
| MANN | 75.23 | 75.65 | 87.21 | 81.02 |
| ViLBERT | 77.87 | 76.30 | 91.15 | 83.08 |
| VisualBERT | 77.01 | 75.70 | 90.25 | 82.33 |
| FixMatchLS | 78.87 | 78.87 | 87.20 | 82.28 |
| RoBERTaMFT | 75.74 | 73.89 | 85.37 | 79.18 |
| DMCC | 81.10 | 84.15 | 90.13 | 87.03 |
| **CrisisSpot** | **84.42** | **84.57** | **96.20** | **90.01** |

#### Humanitarian Task

| Method | Acc | Prec | Rec | F1 |
|--------|-----|------|-----|-----|
| BERT | 69.37 | 70.53 | 70.15 | 70.34 |
| ResNet | 60.24 | 62.45 | 59.82 | 61.11 |
| ViLBERT | 71.23 | 74.55 | 78.80 | 76.36 |
| FixMatchLS | 72.23 | 74.95 | 78.96 | 77.96 |
| RoBERTaMFT | 71.23 | 74.55 | 78.80 | 76.36 |
| DMCC | 78.10 | 77.15 | 78.80 | 77.96 |
| **CrisisSpot** | **80.55** | **81.87** | **82.20** | **82.03** |

### 8.5 Module Component Analysis

| Configuration | Acc | Prec | Rec | F1 |
|---------------|-----|------|-----|-----|
| **CrisisSpot (FEIM + MGLM + SCFM)** | **97.58** | **96.70** | **99.80** | **98.23** |
| w/o SCFM (Social Context) | 94.63 | 95.86 | 96.42 | 96.14 |
| w/o MGLM (Graph Learning) | 93.98 | 94.20 | 95.98 | 94.08 |
| w/o FEIM (Feature Extraction & Interaction) | 95.32 | 95.44 | 97.44 | 97.26 |

**Module Contributions:**
- **FEIM (IDEA Attention):** Core module for multimodal interaction
- **MGLM (Graph Learning):** Enriches features through knowledge propagation
- **SCFM (Social Context):** Adds user credibility and sentiment awareness

### 8.6 Attention Mechanism Analysis

| Configuration | Acc | Prec | Rec | F1 |
|---------------|-----|------|-----|-----|
| **CrisisSpot (with Attention)** | **97.58** | **96.70** | **99.80** | **98.23** |
| CrisisSpot (w/o Attention) | 95.86 | 94.32 | 98.72 | 96.47 |

**Improvement with IDEA:** +1.76% F1 score

### 8.7 Social Context Feature Analysis

| Configuration | Acc | Prec | Rec | F1 |
|---------------|-----|------|-----|-----|
| **CrisisSpot (CIS + UIS + Other)** | **97.58** | **96.70** | **99.80** | **98.23** |
| w/o UIS (User Informative Score) | 96.08 | 95.23 | 98.31 | 96.74 |
| w/o CIS (Crisis Informative Score) | 97.03 | 96.18 | 99.21 | 97.67 |
| w/o Other Features | 96.63 | 95.75 | 98.85 | 97.28 |

**Key Findings:**
1. **UIS is most impactful** (-1.49% F1) - User credibility matters!
2. **CIS adds ~0.56% F1** - Crisis vocabulary awareness helps
3. **Other features (Sentiment, Engagement)** add ~0.95% F1

### 8.8 Modality Combination Analysis

| Configuration | Acc | Prec | Rec | F1 |
|---------------|-----|------|-----|-----|
| **CrisisSpot (T+I+SCF)** | **97.58** | **96.70** | **99.80** | **98.23** |
| w/o Images | 85.13 | 81.04 | 93.50 | 86.80 |
| w/o Text | 83.68 | 81.86 | 89.15 | 85.35 |
| w/o SCF | 91.95 | 92.24 | 90.02 | 91.05 |

**Insight:** Removing ANY modality causes significant performance drop, confirming the importance of multimodal integration.

### 8.9 Inference Time Comparison

| Method | Total Time (s) | Per Sample (ms) |
|--------|---------------|-----------------|
| **CrisisSpot** | 4.23 | 16.45 |
| SCBD | 5.12 | 19.90 |
| FixMatchLS | 6.87 | 26.70 |
| RoBERTaMFT | 7.45 | 28.95 |
| ViLBERT | 8.23 | 31.98 |
| VisualBERT | 7.89 | 30.66 |

**Note:** CrisisSpot has **lower inference time** than most baselines despite having more components, thanks to efficient architecture design.

---

## 9. Code Implementation

### 9.1 Complete IDEA Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IDEA(nn.Module):
    """
    Inverted Dual Embedded Attention.
    
    Combines Harmonious Attention (alignment) and
    Contrary Attention (contradiction detection).
    """
    
    def __init__(self, d_t=768, d_v=1024, d_se=1024, d=128):
        super().__init__()
        
        self.d = d
        self.d_t = d_t
        self.d_v = d_v
        self.d_se = d_se
        
        # Shared embedding transformations
        self.W_t = nn.Linear(d_t, d_se)
        self.W_v = nn.Linear(d_v, d_se)
        self.bn_t = nn.BatchNorm1d(d)
        self.bn_v = nn.BatchNorm1d(d)
        
        # Similarity transformation
        self.W_sim = nn.Linear(d_se, d)
        
        # Fusion layers
        self.W_vis = nn.Linear(2 * d_v, d_v)
        self.W_text = nn.Linear(2 * d_t, d_t)
        self.W_v_fusion = nn.Linear(d_t + d_v, d_v)
        self.W_t_fusion = nn.Linear(d_t + d_v, d_t)
        
        # Self-attention for final features
        self.W_Q = nn.Linear(d_t, d_t)
        self.W_K = nn.Linear(d_t, d_t)
        self.W_V = nn.Linear(d_t, d_t)
        
        # Temperature parameters
        self.T_HAM = 1.65
        self.T_CAM = 0.75
    
    def shared_embedding(self, H_t, H_v):
        """Project modalities to shared space."""
        # Text transformation
        H_Text = self.W_t(H_t)  # (batch, d, d_se)
        H_Text = self.bn_t(H_Text.transpose(1, 2)).transpose(1, 2)
        H_Text = torch.tanh(H_Text)
        
        # Visual transformation
        H_Vis = self.W_v(H_v)  # (batch, d, d_se)
        H_Vis = self.bn_v(H_Vis.transpose(1, 2)).transpose(1, 2)
        H_Vis = torch.tanh(H_Vis)
        
        return H_Text, H_Vis
    
    def compute_similarity(self, H_Text, H_Vis):
        """Compute cross-modal similarity matrix."""
        S_hat = torch.tanh(H_Text + H_Vis)  # Element-wise sum
        S_sim = self.W_sim(S_hat)  # (batch, d, d)
        return S_sim
    
    def harmonious_attention(self, S_sim, H_t, H_v):
        """HAM: Focus on alignment."""
        # Soft attention with high temperature
        Att = F.softmax(S_sim / self.T_HAM, dim=1)
        
        # Attended features
        T_HAM = torch.bmm(Att, H_t)
        V_HAM = torch.bmm(Att, H_v)
        
        return T_HAM, V_HAM
    
    def contrary_attention(self, S_sim, H_t, H_v):
        """CAM: Focus on contradiction."""
        # Invert similarity and sharp attention
        Att = F.softmax(-S_sim / self.T_CAM, dim=1)
        
        # Attended features
        T_CAM = torch.bmm(Att, H_t)
        V_CAM = torch.bmm(Att, H_v)
        
        return T_CAM, V_CAM
    
    def fuse_attended(self, T_HAM, V_HAM, T_CAM, V_CAM, H_t, H_v):
        """Combine HAM and CAM outputs."""
        # Visual side fusion
        V_concat = torch.cat([V_HAM, V_CAM], dim=-1)
        V_fuse = self.W_vis(V_concat)
        
        # Text side fusion
        T_concat = torch.cat([T_HAM, T_CAM], dim=-1)
        T_fuse = self.W_text(T_concat)
        
        # Cross-modal fusion
        V_fusion = self.W_v_fusion(torch.cat([T_fuse, H_v], dim=-1))
        T_fusion = self.W_t_fusion(torch.cat([V_fuse, H_t], dim=-1))
        
        return T_fusion, V_fusion
    
    def self_attention_pooling(self, x):
        """Apply self-attention and pool."""
        # Global average pool
        x_pooled = x.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        
        # Self-attention
        Q = self.W_Q(x_pooled)
        K = self.W_K(x_pooled)
        V = self.W_V(x_pooled)
        
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.bmm(attn, V)
        
        return output.squeeze(1)
    
    def forward(self, H_t, H_v):
        """
        Forward pass.
        
        Args:
            H_t: Text features (batch, d, d_t)
            H_v: Visual features (batch, d, d_v)
        
        Returns:
            FAV: Fused Attended Vector (batch, d_t + d_v)
        """
        # 1. Shared embedding
        H_Text, H_Vis = self.shared_embedding(H_t, H_v)
        
        # 2. Similarity matrix
        S_sim = self.compute_similarity(H_Text, H_Vis)
        
        # 3. Dual attention
        T_HAM, V_HAM = self.harmonious_attention(S_sim, H_t, H_v)
        T_CAM, V_CAM = self.contrary_attention(S_sim, H_t, H_v)
        
        # 4. Fusion
        T_fusion, V_fusion = self.fuse_attended(
            T_HAM, V_HAM, T_CAM, V_CAM, H_t, H_v
        )
        
        # 5. Self-attention pooling
        T_final = self.self_attention_pooling(T_fusion)
        V_final = V_fusion.mean(dim=1)  # Simple pooling for visual
        
        # 6. Concatenate
        FAV = torch.cat([T_final, V_final], dim=-1)
        
        return FAV  # (batch, d_t + d_v)
```

### 9.2 Complete CrisisSpot Model

```python
class CrisisSpot(nn.Module):
    """Full CrisisSpot architecture."""
    
    def __init__(self, task='binary'):
        super().__init__()
        
        # Feature extractors (pretrained)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = resnet50(pretrained=True)
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        # Freeze base models
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Custom conv for ResNet
        self.custom_conv = nn.Conv2d(1024, 1024, kernel_size=4)
        
        # IDEA attention
        self.idea = IDEA(d_t=768, d_v=1024, d_se=1024, d=128)
        
        # GraphSAGE
        self.graphsage = GraphSAGE(input_dim=512, hidden_dim=512, output_dim=512)
        
        # Fusion networks
        self.maln = MALN(input_dim=1792)
        self.shln = SHLN(input_dim=21)
        self.gfln = GFLN(input_dim=1024)
        self.jfln = JFLN()
        
        # Classifier
        self.predictor = PredictionLayer(task=task)
    
    def extract_text_features(self, text_tokens):
        """Extract BERT features."""
        with torch.no_grad():
            output = self.bert(**text_tokens)
        return output.last_hidden_state  # (batch, seq_len, 768)
    
    def extract_visual_features(self, images):
        """Extract ResNet + custom conv features."""
        # Get intermediate ResNet features
        x = self.resnet_intermediate(images)
        
        # Custom conv
        x = self.custom_conv(x)
        
        # Reshape and pad
        batch = x.size(0)
        x = x.view(batch, -1, 1024)  # (batch, 121, 1024)
        
        # Pad to 128
        padding = torch.zeros(batch, 7, 1024, device=x.device)
        x = torch.cat([x, padding], dim=1)  # (batch, 128, 1024)
        
        return x
    
    def forward(self, text_tokens, images, social_features):
        # 1. Base feature extraction
        H_t = self.extract_text_features(text_tokens)
        H_v = self.extract_visual_features(images)
        
        # 2. IDEA attention
        FAV = self.idea(H_t, H_v)
        
        # 3. Graph learning (on batch)
        clip_text = self.clip.get_text_features(**text_tokens)
        clip_image = self.clip.get_image_features(images)
        graph_features = self.graphsage(clip_text, clip_image)
        
        # 4. Fusion
        maln_out = self.maln(FAV)
        shln_out = self.shln(social_features)
        gfln_out = self.gfln(graph_features)
        joint = self.jfln(maln_out, shln_out, gfln_out)
        
        # 5. Prediction
        output = self.predictor(joint)
        
        return output
```

---

## 10. Potential Improvements for MEASF

### 10.1 Direct Applicable Techniques

| Technique | From CrisisSpot | Application to MEASF | Complexity |
|-----------|-----------------|---------------------|------------|
| **IDEA Attention** | HAM + CAM | Image-Text alignment for better selection | 🔴 High |
| **Social Features** | 21-dim SHV | Add to ranking - prioritize credible sources | 🟡 Medium |
| **Temperature Softmax** | T=1.65/0.75 | Fine-tune CLIP similarity softmax | 🟢 Low |
| **GraphSAGE** | K=3 propagation | Similar tweet clustering | 🔴 High |
| **Crisis Lexicon** | 4,268 terms | Filter informative sentences for summarization | 🟢 Low |

### 10.2 Quick Wins (Low Effort, High Impact)

#### 1. Add Crisis Lexicon Scoring

```python
# Add to extractive summarization
CRISIS_TERMS = load_crisis_lexicon()  # 4,268 terms

def crisis_aware_tfidf_score(sentence):
    """Boost sentences with crisis terminology."""
    tfidf = compute_tfidf(sentence)
    crisis_boost = count_crisis_terms(sentence) * 0.1
    return tfidf + crisis_boost
```

#### 2. Temperature-Scaled CLIP Similarity

```python
# Current MEASF approach
similarity = cosine_similarity(text_emb, image_emb)

# CrisisSpot-inspired improvement
T = 1.65  # Softer distribution for image selection
scaled_similarity = similarity / T
attention_weights = softmax(scaled_similarity)
```

#### 3. User Credibility Weighting

```python
def get_tweet_weight(tweet, user_stats):
    """Weight tweets by user credibility."""
    uis = user_stats.get(tweet['user_id'], 0.5)
    base_score = tweet['relevance_score']
    return base_score * (1 + uis * 0.2)  # Boost credible users
```

### 10.3 Medium-Term Improvements

#### 1. Contrary Attention for Image Filtering

```python
class ContrastiveImageFilter:
    """
    Filter out images that CONTRADICT the text summary.
    
    Idea: If image content is opposite to summary sentiment,
    it might be misleading/wrong image.
    """
    
    def filter(self, images, summary):
        summary_emb = clip.encode_text(summary)
        
        valid_images = []
        for img in images:
            img_emb = clip.encode_image(img)
            
            # Harmonious score (alignment)
            ham_score = cosine_similarity(summary_emb, img_emb)
            
            # Contrary score (contradiction)
            cam_score = cosine_similarity(summary_emb, -img_emb)
            
            # Keep if harmonious > contrary
            if ham_score > cam_score:
                valid_images.append(img)
        
        return valid_images
```

#### 2. Sentiment-Aware Summarization

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_weighted_extraction(tweets, target_sentiment='negative'):
    """
    Weight tweets by alignment with crisis sentiment.
    Crisis content tends to be negative (damage, casualties).
    """
    analyzer = SentimentIntensityAnalyzer()
    
    weighted_tweets = []
    for tweet in tweets:
        scores = analyzer.polarity_scores(tweet)
        sentiment_weight = scores['neg'] if target_sentiment == 'negative' else scores['pos']
        
        # Combine with TF-IDF
        final_score = tweet['tfidf_score'] * (1 + sentiment_weight)
        weighted_tweets.append((tweet, final_score))
    
    return sorted(weighted_tweets, key=lambda x: -x[1])
```

### 10.4 Long-Term Research Directions

1. **Graph-based Tweet Clustering for Summarization**
   - Build similarity graph of tweets
   - Use GraphSAGE to propagate importance scores
   - Extract from each cluster (diversity)

2. **Dual-Stream Abstractive Model**
   - Stream 1: Aligned content (HAM-like)
   - Stream 2: Contradictions/exceptions (CAM-like)
   - Combines both for balanced summary

3. **User-Aware Generation**
   - Weight input by user credibility
   - Cite high-credibility sources in summary

---

## 📚 References

1. **CrisisSpot Paper:** https://arxiv.org/abs/2410.08814
2. **GitHub Repository:** https://github.com/Shahid-135/CrisisSpot
3. **CrisisMMD Dataset:** https://crisisnlp.qcri.org/
4. **VADER Sentiment:** https://github.com/cjhutto/vaderSentiment
5. **NRC EmoLex:** https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
6. **GraphSAGE:** https://arxiv.org/abs/1706.02216
7. **CLIP:** https://openai.com/research/clip

---

*Document created: 2026-02-09*
*Last updated: 2026-02-09*
