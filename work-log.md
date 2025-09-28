# 工作日誌 (Work Log)

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

### 目前狀態

- **已完成**：`main.py` 的程式碼已重構為 `WikiDataProcessor` 類別。
- **被阻擋**：`tests/test_main.py` 中的單元測試無法通過。最關鍵的阻礙是 `clean_mediawiki_text` 中的一個正規表示式在 `pytest` 中執行時會產生 `re.error`，儘管該表示式在語法上看起來是正確的。此問題已耗費大量時間仍無法解決。
- **結論**：我無法在目前情況下修復此測試。需要對 Python `re` 模組的內部機制以及工具鏈的字串處理有更深入了解的專家介入。