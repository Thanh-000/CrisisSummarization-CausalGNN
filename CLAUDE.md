# AI DevKit — Project Rules for CrisisSummarization

## 🧠 Skills Auto-Activation Rules

**At the start of EVERY conversation**, the agent MUST:

1. **Read all SKILL.md files** from `.agent/skills/` directory before responding
2. **Silently determine** which skills are relevant based on user intent
3. **Auto-apply** matching skills without asking for permission — just announce: "Đang áp dụng skill: `<tên>`"

### 🎯 Skill Trigger Table — Auto-match user intent to skill

| Nếu user đề cập đến... | Skill tự động áp dụng |
|------------------------|----------------------|
| bug, lỗi, crash, error, không chạy, fail, exception, CUDA, memory leak, regression, tại sao X không hoạt động | **`debug`** |
| feature mới, thêm tính năng, implement, xây dựng, thiết kế hệ thống | **`dev-lifecycle`** |
| hiểu code, document, giải thích module, map function, architecture | **`capture-knowledge`** |
| refactor, đơn giản hóa, phức tạp quá, clean up, technical debt, khó đọc | **`simplify-implementation`** |
| review docs, README, báo cáo, hướng dẫn, tài liệu | **`technical-writer`** |
| nhớ lại, lưu lại, knowledge trước đó, đã làm gì, convention | **`memory`** |

### 📋 Rule bắt buộc khi áp dụng skill

**`debug` skill:**
- KHÔNG sửa code trước khi user approve kế hoạch fix
- Luôn hỏi: "Observed vs Expected behavior" trước
- Cung cấp ít nhất 3 hypothesis với test command cụ thể

**`dev-lifecycle` skill:**
- Luôn bắt đầu từ Phase 1 (requirements) với feature mới
- Tạo file docs tương ứng trong `docs/ai/{phase}/feature-{name}.md`
- Chạy `npx ai-devkit@latest lint` trước khi bắt đầu phase

**`capture-knowledge` skill:**
- Xác nhận entry point tồn tại trước khi phân tích
- Tạo diagram mermaid cho mọi flow phức tạp
- Lưu kết quả vào `docs/ai/implementation/knowledge-{name}.md`

**`simplify-implementation` skill:**
- KHÔNG sửa code cho đến khi user approve kế hoạch
- Đưa ra before/after snippet để so sánh
- Ưu tiên: readability > brevity

**`technical-writer` skill:**
- Rate 4 chiều: Clarity / Completeness / Actionability / Structure (1-5)
- Phân loại issue: High / Medium / Low priority
- Đề xuất fix text cụ thể, không vague

**`memory` skill:**
- Trước khi hỏi user câu lặp lại → search memory trước
- Sau khi resolve issue quan trọng → tự động lưu solution
- Dùng CLI nếu MCP không available: `npx ai-devkit@latest memory search/store`

---

## 📁 Project Context — CrisisSummarization

**Mục tiêu:** Multimodal Classification of Social Media Disaster Posts with GNN (IEEE Access 2025)

**Dataset chính:**
- CrisisMMD v2.0 (text + image, disaster tweets)
- Brazilian Protest Dataset (7Sept)

**Tech stack:**
- Python / PyTorch / CLIP embeddings
- GNN (Graph Neural Networks) với KNN graph construction
- Google Colab (GPU runtime)
- Notebook chính: `mm_class_experiment.ipynb`

**Thư mục quan trọng:**
- `docs/ai/` — Phase documentation (requirements, design, planning, testing...)
- `.agent/skills/` — AI skills (debug, dev-lifecycle, capture-knowledge, memory, technical-writer, simplify-implementation)
- `docs/latex_report/` — LaTeX báo cáo kỹ thuật
- `BaoCaoKyThuat.md` — Báo cáo kỹ thuật Markdown

---

## 📂 Documentation Structure

```
docs/ai/
├── requirements/     # Yêu cầu và mục tiêu
├── design/           # Kiến trúc hệ thống (dùng mermaid diagram)
├── planning/         # Task breakdown
├── implementation/   # Ghi chú implementation + knowledge docs
├── testing/          # Chiến lược test
├── deployment/       # Deploy / Colab setup
└── monitoring/       # Theo dõi experiment
```

---

## 💻 Code Style

- Python: PEP8, type hints khi có thể
- Comment tiếng Việt được phép cho explanation
- Notebook cells: mỗi cell có header comment rõ ràng
- Tên biến: tiếng Anh, snake_case

---

## 🔄 Development Workflow

1. Review `docs/ai/` trước khi implement bất kỳ feature nào
2. Với feature mới → kích hoạt `dev-lifecycle` skill
3. Với bug → kích hoạt `debug` skill (evidence-first, không sửa vội)
4. Sau khi resolve → lưu solution vào memory

---

## 🛠️ Key Commands

| Slash command | Tác dụng |
|--------------|---------|
| `/new-requirement` | Bắt đầu feature mới (Phase 1) |
| `/review-requirements` | Validate requirements doc |
| `/review-design` | Review design doc |
| `/execute-plan` | Thực thi planning tasks |
| `/update-planning` | Cập nhật sau khi xong task |
| `/check-implementation` | Verify code vs design |
| `/writing-test` | Viết tests (target 100% coverage) |
| `/code-review` | Review code trước khi push |
| `/capture-knowledge` | Document một module/function |
| `/debug` | Debug có cấu trúc |

---

## ⚙️ Memory Integration (Luôn dùng)

**Trước mỗi task** → Search memory:
```bash
npx ai-devkit@latest memory search --query "<topic liên quan>"
```

**Sau khi giải quyết vấn đề quan trọng** → Lưu vào memory:
```bash
npx ai-devkit@latest memory store \
  --title "<mô tả ngắn>" \
  --content "<giải pháp chi tiết>" \
  --tags "crisis,gnn,python" \
  --scope "project:CrisisSummarization"
```

---

## ⚡ Quick Reference — Skills trong project này

```
.agent/skills/
├── debug/                  → Dùng khi: bug, lỗi, crash
├── dev-lifecycle/          → Dùng khi: feature mới, SDLC
├── capture-knowledge/      → Dùng khi: hiểu/doc code
├── memory/                 → Dùng khi: lưu/tìm kiến thức
├── simplify-implementation/→ Dùng khi: refactor, clean code
└── technical-writer/       → Dùng khi: review docs
```
