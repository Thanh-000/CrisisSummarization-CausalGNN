# AGENT.md — Antigravity Rules for CrisisSummarization

> **This file is read automatically at the start of every Antigravity conversation.**
> Full rules and documentation structure are in `CLAUDE.md`.

---

## 🚀 MANDATORY: Skills Auto-Activation

**STEP 1 — At conversation start, silently read all skill files:**
```
.agent/skills/debug/SKILL.md
.agent/skills/dev-lifecycle/SKILL.md
.agent/skills/capture-knowledge/SKILL.md
.agent/skills/memory/SKILL.md
.agent/skills/simplify-implementation/SKILL.md
.agent/skills/technical-writer/SKILL.md
```

**STEP 2 — Match user intent → auto-apply skill:**

```
User mentions:           → Apply skill:
─────────────────────────────────────────────────────────
bug / lỗi / crash /
error / fail / không
chạy / CUDA / trace      → debug
─────────────────────────────────────────────────────────
feature mới / thêm /
implement / xây dựng /
thiết kế                 → dev-lifecycle
─────────────────────────────────────────────────────────
hiểu code / giải thích /
document / map / tìm
hiểu module/function     → capture-knowledge
─────────────────────────────────────────────────────────
refactor / đơn giản /
clean / phức tạp /
technical debt           → simplify-implementation
─────────────────────────────────────────────────────────
review docs / README /
báo cáo / tài liệu /
hướng dẫn               → technical-writer
─────────────────────────────────────────────────────────
nhớ lại / lưu lại /
đã làm gì / convention /
trước đây               → memory
─────────────────────────────────────────────────────────
```

**STEP 3 — Announce skill activation:**
> Nói rõ: *"Đang áp dụng skill: `debug` — [lý do ngắn gọn]"*

**STEP 4 — Follow skill instructions strictly.**
Do NOT skip or summarize skill steps.

---

## ⚠️ Hard Rules (NEVER skip)

1. **`debug` + `simplify-implementation`**: KHÔNG sửa code cho đến khi user APPROVE kế hoạch
2. **`capture-knowledge`**: KHÔNG tạo doc cho đến khi analysis hoàn chỉnh
3. **`memory` before repetitive Q**: Search memory TRƯỚC khi hỏi user lại câu đã hỏi
4. **`dev-lifecycle`**: LUÔN bắt đầu từ Phase 1 với feature mới

---

## 📌 Project Snapshot

| Field | Value |
|-------|-------|
| **Project** | CrisisSummarization |
| **Goal** | Multimodal GNN for disaster tweet classification |
| **Main file** | `mm_class_experiment.ipynb` |
| **Stack** | Python · PyTorch · CLIP · GNN · Google Colab |
| **Dataset** | CrisisMMD v2.0 + Brazilian 7Sept protest |
| **Docs** | `docs/ai/` (requirements/design/planning/implementation/testing) |
| **Skills** | `.agent/skills/` (6 skills installed) |

---

## 🔗 See Also
- Full rules → `CLAUDE.md`
- Skills detail → `.agent/skills/*/SKILL.md`
- Phase docs → `docs/ai/`
