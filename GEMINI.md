# GEMINI.md — CausalCrisis V3 Project Rules

> Rules cụ thể cho project này. Global rules ở `~/.gemini/GEMINI.md`.
> Antigravity merge cả hai khi start conversation.

---

## 🚀 Skills Auto-Activation (MANDATORY)

Đầu mỗi conversation, đọc toàn bộ `SKILL.md` trong `.agent/skills/` và áp dụng skill tương ứng.

### Trigger Table — CausalCrisis V3

| User đề cập | Skill | Action |
|------------|-------|--------|
| bug, lỗi, crash, error, CUDA, fail, không chạy | **`debug`** | Đọc `.agent/skills/debug/SKILL.md` → workflow evidence-first |
| feature mới, implement, thêm, xây dựng, thiết kế | **`dev-lifecycle`** | Đọc `.agent/skills/dev-lifecycle/SKILL.md` → Phase 1 |
| hiểu code, document, giải thích, map module | **`capture-knowledge`** | Đọc `.agent/skills/capture-knowledge/SKILL.md` → analyse first |
| refactor, đơn giản, clean, phức tạp, technical debt | **`simplify-implementation`** | Đọc `.agent/skills/simplify-implementation/SKILL.md` → plan first |
| review docs, README, báo cáo, tài liệu | **`technical-writer`** | Đọc `.agent/skills/technical-writer/SKILL.md` → rate 4 dims |
| nhớ lại, lưu, convention, trước đây, đã làm gì | **`memory`** | Đọc `.agent/skills/memory/SKILL.md` → search trước khi hỏi |
| nghiên cứu tự động, experiment loop, autoresearch | **`autoresearch`** | Đọc `.agent/skills/autoresearch/SKILL.md` → two-loop architecture |
| CLIP, feature extraction, multimodal, visual | **`multimodal-clip`** | Đọc `.agent/skills/multimodal-clip/SKILL.md` → CLIP pipeline |
| viết paper, LaTeX, conference, submission | **`ml-paper-writing`** | Đọc `.agent/skills/ml-paper-writing/SKILL.md` → never hallucinate citations |
| brainstorm, ý tưởng, ideation, hướng mới | **`research-ideation`** | Đọc `.agent/skills/research-ideation/brainstorming/SKILL.md` → 10 frameworks |
| sáng tạo, novel, creative thinking, analogy | **`creative-thinking`** | Đọc `.agent/skills/research-ideation/creative-thinking/SKILL.md` → cognitive frameworks |
| evaluation, metrics, ablation, LODO, cross-val | **`evaluation`** | Đọc `.agent/skills/evaluation/SKILL.md` → metric + significance |
| dataset, dataloader, preprocessing, augmentation | **`data-processing`** | Đọc `.agent/skills/data-processing/SKILL.md` → pipeline design |

### Announce khi activate:
> *"Đang áp dụng skill: `<tên>` — [lý do 1 câu]"*

---

## 📌 Project Snapshot

| Field | Value |
|-------|-------|
| **Project** | CausalCrisis V3 |
| **Goal** | Multimodal Causal Classification > 90% F1 on CrisisMMD |
| **Backbone** | CLIP ViT-L/14 (frozen) → 768-dim features |
| **Novelty** | Per-modality causal disentanglement + Cross-modal C_vt + Backdoor Adjustment |
| **Stack** | Python · PyTorch · CLIP · AdamW · Focal Loss · Google Colab Pro (A100) |
| **Dataset** | CrisisMMD v2.0 |
| **Target Venue** | IEEE Access 2025 / AAAI 2026 |

---

## 📂 Documentation & Skills

```
.agent/skills/
├── autoresearch/         ← 🆕 Autonomous research orchestration (two-loop)
│   └── templates/        ← research-state.yaml, findings.md, research-log.md
├── multimodal-clip/      ← 🆕 CLIP feature extraction + caching
├── ml-paper-writing/     ← 🆕 Conference paper writing + citation verification
├── research-ideation/    ← 🆕 Brainstorming + creative thinking frameworks
│   ├── brainstorming/
│   └── creative-thinking/
├── evaluation/           ← 🆕 Metrics, ablation, LODO, statistical testing
├── data-processing/      ← 🆕 Dataset loading, DataLoader, class balance
├── capture-knowledge/    ← Knowledge documentation
├── debug/                ← Evidence-first debugging
├── dev-lifecycle/        ← SDLC workflow
├── memory/               ← AI DevKit memory
├── simplify-implementation/ ← Code simplification
└── technical-writer/     ← Documentation review
```

---

## ⚙️ Memory — Luôn dùng CLI

```bash
# Tìm kiếm trước khi hỏi lại
npx ai-devkit@latest memory search \
  --query "<topic>" \
  --scope "project:CausalCrisis"

# Lưu sau khi resolve
npx ai-devkit@latest memory store \
  --title "<title>" \
  --content "<solution>" \
  --tags "causalcrisis,clip,pytorch,causal" \
  --scope "project:CausalCrisis"
```

---

## ⚠️ Hard Rules

1. `debug` / `simplify-implementation`: **KHÔNG sửa code** trước khi user approve
2. `capture-knowledge`: **KHÔNG viết doc** trước khi phân tích xong
3. `memory`: **Search memory trước** khi hỏi câu lặp lại
4. `dev-lifecycle`: **Phase 1 trước** cho mọi feature mới
5. `ml-paper-writing`: **KHÔNG hallucinate citations** → verify trước khi cite
6. `autoresearch`: **Lock protocol trước run** → git commit protocol trước experiment
7. `evaluation`: **Ablation phải đầy đủ** → test mọi component individually

---

## 💻 Code Style

- Python PEP8 + type hints khi có thể
- Notebook cells: mỗi cell có header comment
- Tên biến: tiếng Anh, snake_case
- Comment giải thích được viết tiếng Việt

---

## 🔗 Xem thêm

- Global rules → `~/.gemini/GEMINI.md`
- Full rules → `CLAUDE.md`
- All skills → `.agent/skills/*/SKILL.md`
- Architecture → xem artifact `causalcrisis_v3_architecture.md`
- Skills source: [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs)
