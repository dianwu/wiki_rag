## 2025-09-29

### 11. 更換 Embedding 模型：Google API -> 本地端 Sentence Transformers

- **動機**: `GoogleGenerativeAIEmbeddings` ("models/gemini-embedding-001") 為付費服務，且目前的 API Key 遇到 Quota 問題。為了降低成本並移除外部 API 依賴，決定更換為免費的本地端開源模型。
- **討論與決策**:
    - 探討了多個免費替代方案，包括 `all-MiniLM-L6-v2`, `multi-qa-mpnet-base-dot-v1`, 和 `paraphrase-multilingual-mpnet-base-v2`。
    - 使用者最終選擇了 `all-mpnet-base-v2`，因其在效能和品質之間取得了良好的平衡。
    - 確認了使用 `sentence-transformers` 函式庫的模型會在本地端執行，僅在首次使用時需要網路下載模型。
- **操作**:
    - **更新 `requirements.txt`**:
        - 移除了 `langchain-google-genai`。
        - 新增了 `sentence-transformers` 和 `langchain-community`。
    - **更新 `main.py`**:
        - 將 `GoogleGenerativeAIEmbeddings` 的 import 替換為 `langchain_community.embeddings.HuggingFaceEmbeddings`。
        - 修改 embedding 初始化邏輯，改為載入 `HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")`，並移除 Google API Key 相關的錯誤處理。
- **結論**: 專案現在使用 `all-mpnet-base-v2` 模型在本地進行 embedding，移除了對付費 Google API 的依賴，解決了 API Quota 問題。

# 工作日誌 (Work Log)

## 2025-09-28 (續)

### 8. 向量資料庫技術棧更換：FAISS -> Chroma

- **目標**: 根據專案需求，將計畫中的向量資料庫從 `FAISS` 更換為 `Chroma`。
- **操作**:
    - **更新 `plan.md`**: 將開發計畫中的 "第二階段：向量化與儲存" 和 "第三階段：檢索與生成" 部分的 `FAISS` 相關描述替換為 `Chroma`。
    - **更新 `task.md`**:
        - 將 "第二階段：向量化與儲存" 的任務標題和內容從 `FAISS` 更新為 `Chroma`。
        - 調整了任務步驟，以反映 `Chroma` 的 `from_documents` 和持久化 (`persist_directory`) 機制，移除了 `save_local` 和 `load_local` 的相關說明。
- **結論**: 專案的開發計畫和下一階段任務清單已完全更新，以反映使用 `Chroma` 作為向量資料庫的技術決策。

## 2025-09-28

### 1. 專案初始化

- **目標**：建立一個使用 LangChain 實作 RAG 的專案基礎架構。
- **操作**：
    - 建立 `data/` 目錄，用於存放 MediaWiki XML 資料。
    - 建立 `requirements.txt`，定義初始 Python 依賴 (當時以 OpenAI 為目標)。
    - 建立主程式入口 `main.py`。
    - 建立 `plan.md`，規劃了從資料處理到 RAG 鏈實現的完整開發步驟。
    - 建立 `readme.md`，提供專案說明與設定指南。
    - 設定 `.gitignore` 以忽略不必要的檔案。

### 2. 技術棧更換：OpenAI -> Google Gemini

- **目標**：根據使用者要求，將專案的 LLM 服務從 OpenAI 更換為 Google Gemini。
- **操作**：
    - **更新 `readme.md`**：修改說明文件，將套件名稱從 `langchain-openai` 改為 `langchain-google-genai`，並將環境變數從 `OPENAI_API_KEY` 改為 `GOOGLE_API_KEY`。
    - **更新 `requirements.txt`**：將依賴套件替換為 `langchain-google-genai`。
    - **更新 `plan.md`**：同步修改開發計畫中的模型名稱，例如 `GoogleGenerativeAIEmbeddings` 和 Gemini 模型。

### 3. 環境設定與依賴安裝

- **目標**：完成 `readme.md` 中要求的開發環境設定。
- **操作**：
    - 建立 `.env.example` 檔案，作為設定 `GOOGLE_API_KEY` 的範本。
    - 執行命令 `python3 -m venv .venv` 建立 Python 虛擬環境。
    - 啟用虛擬環境並執行 `pip install -r requirements.txt`，成功將所有依賴套件安裝至 `.venv` 中。

### 4. 細化第一階段開發任務

- **目標**：為方便後續開發者接手，將 `plan.md` 中的第一階段任務拆解為更具體的步驟。
- **操作**：
    - 根據 `plan.md` 的「第一階段：資料載入與處理」，建立了一份詳細的待辦清單。
    - 將此清單儲存為 `task.md` 檔案，其中包含從尋找檔案、解析 XML、清理內容到文件分割的具體子任務。

### 5. 完成第一階段：資料載入與處理

- **目標**：根據 `task.md` 的指引，完成 `main.py` 中資料處理的完整流程。
- **操作**：
    - **實作檔案搜尋**：在 `main.py` 中加入 `find_xml_file` 函式，自動尋找並選取 `data/` 目錄下的 XML 檔案。
    - **實作 XML 解析**：加入 `load_and_parse_xml` 函式，使用 `lxml.etree.iterparse` 高效解析 MediaWiki XML。
    - **實作內容清理**：加入 `clean_mediawiki_text` 函式，整合 `BeautifulSoup` 和正則表達式，移除 HTML 標籤和 Wiki 標記。
    - **實作文件分割**：加入 `split_documents` 函式，使用 `langchain` 的 `RecursiveCharacterTextSplitter` 將長文本切分為適合處理的區塊 (chunks)。

## 2025-09-28 (續)

### 6. 程式碼重構與單元測試

- **目標**：將 `main.py` 重構為物件導向的結構，並為其建立完整的單元測試。
- **操作**：
    - 將 `main.py` 中的函式重構為 `WikiDataProcessor` 類別。
    - 將 `print` 呼叫替換為 `logging` 模組。
    - 使用 `pathlib` 取代 `os.path`。
    - 更新 `tests/test_main.py` 以反映新的類別結構，並加入整合測試。
- **遇到的問題與解決方案**：
    1.  **問題**：在重構後，單元測試持續失敗，主要集中在 `clean_mediawiki_text`（正規表示式處理）和 `load_and_parse_xml`（`lxml` 的模擬）中。
    2.  **嘗試的解決方案 (皆失敗)**：
        - **正規表示式**：多次嘗試修正 wiki 連結的正規表示式 `r'\[\[([^\]|]+)\]\]'`，但始終在 `pytest` 環境中觸發 `re.error: unbalanced parenthesis` 或 `unterminated character set` 錯誤。這似乎是一個棘手的字串轉義問題，在工具的 `write_file` 呼叫中，字串的處理方式與預期不符。
        - **lxml 模擬**：嘗試使用 `mock.return_value` 和 `mock.side_effect` 等多種方式來模擬 `etree.iterparse`，但都無法讓測試正確地從記憶體中的假 XML 資料產生迭代器，導致測試失敗（回傳 0 個頁面或無限遞迴）。
        - **簡化策略**：嘗試將複雜的正規表示式拆分為多個簡單的表示式，並讓 XML 解析測試直接讀取暫存檔案而非模擬。此方法仍然觸發了相同的 `re.error`，顯示問題根源比想像的更深。

### 7. 驗證測試修復

- **目標**：確認使用者提供的 `tests/test_main.py` 已解決單元測試問題。
- **操作**：
    - 執行 `source .venv/bin/activate && pytest tests/test_main.py`。
    - 所有 5 個測試均成功通過。
- **結論**：測試問題已解決，專案不再被阻擋。

### 9. 實作向量化與儲存 (Chroma)

- **目標**: 根據 `task.md` 推進到第二階段，完成文本向量化與 Chroma 資料庫的建立。
- **操作**:
    - **環境設定**: 在 `main.py` 中加入 `dotenv`, `GoogleGenerativeAIEmbeddings`, 和 `Chroma` 的匯入。
    - **建立向量資料庫**: 實作 `create_vector_store` 函式，使用 `Chroma.from_documents` 來建立並持久化向量資料庫。
    - **整合儲存與載入邏輯**: 在 `main` 函式中，加入檢查 `chroma_db` 目錄是否存在的邏輯。如果存在則載入，否則執行資料處理並建立新的資料庫。
    - **更新依賴**: 將 `requirements.txt` 中的 `faiss-cpu` 替換為 `chroma` 和 `langchain-community`，並重新安裝依賴。

### 10. 處理 Google API Quota 問題

- **問題描述**: 執行 `main.py` 時，`GoogleGenerativeAIEmbeddings` 觸發 `ResourceExhausted: 429 You exceeded your current quota` 錯誤，即使在處理少量文件時也是如此。錯誤訊息顯示 `limit: 0`，暗示 API 金鑰的免費層級可能沒有 `embed_content` 的使用權限。
- **嘗試的解決方案**:
    1.  **限制資料量**: 修改 `main.py`，將處理的頁面數量限制為 10 頁。**結果**: 錯誤依舊存在，證明問題不在於請求量的大小，而在於權限本身。
    2.  **處理庫棄用警告**: `Chroma` 的使用方式已過時。將 `langchain-community` 和 `chromadb` 替換為 `langchain-chroma`，並更新 `main.py` 中的 import 語句。**結果**: 雖然代碼更現代化，但 API Quota 錯誤仍然存在。
- **結論**: 問題根源極可能在於 Google Cloud 專案的設定或 API 金鑰的權限，而非程式碼本身。已建議使用者檢查 Google Cloud Console 中的相關設定。
- **後續動作**: 已將 `main.py` 中限制資料量的臨時程式碼還原，以保持程式碼的整潔。專案目前因外部 API 權限問題而受阻。

### 目前狀態

- **已完成**：`main.py` 的向量化儲存與載入邏輯已根據 `task.md` 完成。
- **已完成**：專案依賴已更新至最新的 `langchain-chroma`。
- **受阻**: 專案的執行被 Google API 的 `ResourceExhausted` 錯誤阻擋。在使用者解決 API 金鑰權限問題之前，無法成功建立向量資料庫。
- **下一步**：等待使用者確認 Google Cloud API 權限問題已解決，然後重試執行 `main.py`。