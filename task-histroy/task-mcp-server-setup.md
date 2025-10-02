# 第五階段：MCP Server 實作 (FastMCP)

## 目標
將檢索邏輯封裝成一個符合模型內容協定 (MCP) 的伺服器，使其能被 AI Agent 或其他 MCP 客戶端標準化地呼叫。

## 任務清單

### 1. 框架選擇與安裝
- [x] 決定使用 `FastMCP` 框架。
- [x] 將 `fastmcp` 新增至 `requirements.txt` 並完成安裝。

### 2. 伺服器實作
- [x] 建立一個新的伺服器檔案 `fastmcp_server.py`。
- [x] 在伺服器中初始化 `RAGSystem`。

### 3. Tool 功能設計與重構
- [x] 將 `rag_logic.py` 中的文件檢索功能，封裝成一個名為 `retrieve_wiki_documents` 的 FastMCP `tool`。
- [x] `tool` 應支援 `question` 和 `k` 參數。
- [x] `tool` 的回傳值應為包含檢索結果的 JSON 字串。

### 4. 文件更新
- [x] 更新 `readme.md`，說明如何啟動和使用新的 FastMCP 伺服器。
- [x] 更新 `plan.md`，將第五階段的計畫同步為 FastMCP 架構。
- [x] 更新 `task.md` (本檔案) 以反映當前任務狀態。
- [x] 清理 `work-log.md` 中與 FastAPI 相關的舊紀錄。