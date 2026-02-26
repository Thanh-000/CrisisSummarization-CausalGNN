---
description: Cài đặt và cấu hình NotebookLM MCP Server cho Antigravity (TypeScript/npx version)
---

# Hướng dẫn cài đặt NotebookLM MCP Server

Workflow này sử dụng **PleasePrompto/notebooklm-mcp** (TypeScript) qua `npx`, không cần Python.

## Yêu cầu

- Node.js (đã có v24.13.0)
- npm/npx (đã có v11.6.2)

## Bước 1: Cấu hình MCP Server

Cập nhật file `C:\Users\Admin\.gemini\antigravity\mcp_config.json`, thêm entry sau vào `mcpServers`:

```json
"notebooklm-mcp-server": {
  "command": "npx.cmd",
  "args": [
    "-y",
    "notebooklm-mcp@latest"
  ]
}
```

> **Lưu ý**: Dùng `npx.cmd` thay vì `npx` để tránh lỗi PowerShell Execution Policy trên Windows.

## Bước 2: Reload Antigravity

> [!IMPORTANT]
> **YÊU CẦU RELOAD ANTIGRAVITY**
> 
> - Nhấn `Ctrl+Shift+P`
> - Gõ "Developer: Reload Window"
> - Nhấn Enter

## Bước 3: Xác thực với NotebookLM (lần đầu)

Sau khi reload, nói trong chat:
```
"Log me in to NotebookLM"
```

Hoặc sử dụng tool `setup_auth`. Trình duyệt Chrome sẽ mở ra để bạn đăng nhập Google.

## Bước 4: Xác nhận cài đặt thành công

Sử dụng tool `get_health` hoặc `list_notebooks` để kiểm tra server đã hoạt động.

---

## Thông tin thêm

- **Package**: `notebooklm-mcp` (npm, TypeScript)
- **Repo**: https://github.com/PleasePrompto/notebooklm-mcp
- **Config file**: `C:\Users\Admin\.gemini\antigravity\mcp_config.json`
- **Backup config**: `C:\Users\Admin\AppData\Roaming\Antigravity\User\mcp.json`
