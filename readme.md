# MediaWiki RAG 專案

> **Note:** 本專案主要由 Google Gemini Code Assist 開發與維護。

本專案使用 LangChain 實作一個 Retrieval-Augmented Generation (RAG) 系統，能以 MediaWiki 導出的 XML 檔案作為知識庫。預設情況下，它使用開源的 Sentence-Transformers 模型在**本地端**進行文本向量化，並可選擇性地設定使用 Google Gemini 大型語言模型進行最終的答案生成。

## 專案架構

```
.
├── data/                  # 存放 MediaWiki XML 檔案
├── chroma_db/             # 持久化的向量資料庫
├── .gitignore             # Git 忽略設定
├── main.py                # 主程式進入點
├── ingest.py              # 資料載入與處理腳本
├── tests/                 # 單元與整合測試
├── plan.md                # 開發計畫
├── readme.md              # 專案說明
└── requirements.txt       # Python 套件依賴
```

## 執行環境

- **作業系統**: Ubuntu 22.04.2 LTS
- **Python 版本**: 3.10.x
- **開發工具**: 本專案由 Google Gemini Code Assist 輔助開發

## 環境設定

1.  **建立並啟用虛擬環境**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **安裝依賴套件**

    ```bash
    pip install -r requirements.txt
    ```

3.  **設定環境變數 (可選)**

    僅在您需要使用線上 LLM (例如 Google Gemini) 進行答案生成，或希望將 Embedding 切換為線上 API 模型時，才需要設定此項。

    在專案根目錄建立一個 `.env` 檔案，並填入您的 Google API 金鑰：

    ```
    GOOGLE_API_KEY="your-google-api-key"
    ```

## 如何執行

### 1. 資料導入 (Ingestion)

此步驟會處理 `data/` 目錄中的 XML 檔案，將其轉換為向量並儲存在 `chroma_db` 中。

1.  將您的 MediaWiki XML 匯出檔案放入 `data/` 目錄下。
2.  執行導入腳本：

    ```bash
    python3 ingest.py
    ```

    **注意**：
    *   首次執行時，程式會自動從 Hugging Face Hub 下載所需的 Embedding 模型 (例如 `all-mpnet-base-v2`)，這可能需要幾分鐘時間。
    *   下載模型後，程式會處理 XML 檔案並建立 `chroma_db` 向量資料庫。此過程也可能需要一些時間，取決於您的資料量和硬體效能。
    *   如果 `chroma_db` 目錄已存在，腳本將會提示並終止，以避免覆蓋現有資料。如需重新導入，請先手動刪除 `chroma_db` 目錄。

### 2. 執行問答系統

資料導入完成後，您可以啟動主程式來進行問答。

```bash
python3 main.py
```

程式啟動後，會載入本地的向量資料庫和 LLM。您可以在命令列中輸入問題，程式會根據知識庫內容生成答案。

### 3. 執行 MCP 伺服器 (FastMCP)

本專案使用 FastMCP 提供一個符合模型內容協定 (Model Context Protocol) 的伺服器，讓 AI Agent 可以呼叫其提供的工具。

```bash
# 建議使用 fastmcp CLI 來啟動
fastmcp run fastmcp_server.py
```

伺服器預設會運行在 `http://127.0.0.1:8000`，並使用 SSE (Server-Sent Events) 協定進行通訊。

**客戶端如何呼叫**

任何支援 MCP 的客戶端都可以與此伺服器互動。以下是一個使用 `fastmcp.Client` 的 Python 客戶端範例：

```python
import asyncio
import json
from fastmcp import Client

async def main():
    # 連接到本地運行的 FastMCP 伺服器
    async with Client("http://127.0.0.1:8000") as client:
        print("成功連接到 MCP 伺服器")
        
        # 準備要呼叫的工具參數
        tool_params = {
            "question": "wmi是什麼",
            "k": 5
        }
        
        print(f"正在呼叫工具: retrieve_wiki_documents")
        
        # 呼叫伺服器上名為 'retrieve_wiki_documents' 的工具
        result = await client.call_tool("retrieve_wiki_documents", tool_params)
        
        # FastMCP 工具的回傳內容是字串，我們將其解析為 JSON
        retrieved_docs = json.loads(result.content[0].text)
        
        print("--- 工具回傳結果 ---")
        print(json.dumps(retrieved_docs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
```

## 測試

本專案包含一套測試，以確保資料處理邏輯的正確性。若要執行測試，請在專案根目錄下執行：

```bash
.venv/bin/python -m pytest
```

所有測試都應該會通過，以確保系統的穩定性。

## Embedding 模型設定

本專案預設使用免費的本地端模型進行 Embedding，但您也可以輕易地將其修改為使用線上的 API 服務。

### 使用本地端模型 (預設)

目前的 `ingest.py` 設定為使用 `HuggingFaceEmbeddings`。

```python
# in ingest.py
from langchain_community.embeddings import HuggingFaceEmbeddings

# ...

embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={'device': device}
)
```

- **優點**: 免費、資料不離開本地、一次下載後可離線運作。
- **缺點**: 需要在本地消耗 CPU/RAM 資源。

### 切換為線上 API 模型 (例如 Google)

如果您希望使用 Google 的線上 Embedding 服務，可以修改 `ingest.py`。

1.  **安裝對應套件**:
    ```bash
    pip install langchain-google-genai
    ```

2.  **修改程式碼**:
    在 `ingest.py` 中，註解掉 `HuggingFaceEmbeddings` 的部分，並換成 `GoogleGenerativeAIEmbeddings`。

    ```python
    # in ingest.py
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    # from langchain_community.embeddings import HuggingFaceEmbeddings # 註解掉

    # ...

    # --- Initialize Embeddings ---
    # embeddings = HuggingFaceEmbeddings(...) # 註解掉

    # 使用 Google API
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 請確保您已在 .env 檔案中設定 GOOGLE_API_KEY
    ```
- **優點**: 不消耗本地端運算資源。
- **缺點**: 需要網路連線、可能產生費用、資料需要傳送至外部 API。
