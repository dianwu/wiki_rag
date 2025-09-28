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

### 目前狀態

專案基礎設定已全部完成。`task.md` 提供了第一階段開發的清晰路線圖，開發者可直接依此開始編寫程式碼。專案已準備好交接。

## 2025-09-28 (續)

### 5. 完成第一階段：資料載入與處理

- **目標**：根據 `task.md` 的指引，完成 `main.py` 中資料處理的完整流程。
- **操作**：
    - **實作檔案搜尋**：在 `main.py` 中加入 `find_xml_file` 函式，自動尋找並選取 `data/` 目錄下的 XML 檔案。
    - **實作 XML 解析**：加入 `load_and_parse_xml` 函式，使用 `lxml.etree.iterparse` 高效解析 MediaWiki XML。
    - **實作內容清理**：加入 `clean_mediawiki_text` 函式，整合 `BeautifulSoup` 和正則表達式，移除 HTML 標籤和 Wiki 標記。
    - **實作文件分割**：加入 `split_documents` 函式，使用 `langchain` 的 `RecursiveCharacterTextSplitter` 將長文本切分為適合處理的區塊 (chunks)。
- **遇到的問題與解決方案**：
    1.  **問題**：首次執行 `main.py` 時，出現 `ModuleNotFoundError: No module named 'lxml'`。
        - **分析**：雖然 `requirements.txt` 中已包含 `lxml`，但 `run_shell_command` 預設可能未使用專案的虛擬環境 (`.venv`)。
        - **解決方案**：明確使用虛擬環境中的 Python 解譯器來執行腳本，即改用命令 `.venv/bin/python3 main.py`，問題解決。
    2.  **問題**：腳本執行成功，但 XML 解析結果為 0 個頁面。
        - **分析**：懷疑是 XML 的命名空間 (namespace) 不符。程式碼中硬編碼為 `export-0.10`，但實際檔案可能不同。
        - **解決方案**：使用 `head` 命令查看 `.xml` 檔案的開頭，確認命名空間為 `export-0.11`。將程式碼中的命名空間更新後，成功解析出 1025 個頁面。
    3.  **問題**：腳本執行時 `BeautifulSoup` 產生 `XMLParsedAsHTMLWarning` 警告。
        - **分析**：這是因為 MediaWiki 的文本內容被當作 HTML 解析，而 `task.md` 中也指明使用 `html.parser`。雖然不影響結果，但為了輸出整潔，應當處理。
        - **解決方案**：根據警告提示，改用 `lxml` 作為 `BeautifulSoup` 的解析器，並加入程式碼以忽略此類警告，使執行輸出更乾淨。
- **目前狀態**：
    - `main.py` 現在具備完整的資料載入、解析、清理和分割功能，第一階段開發目標已全部達成。
    - `task.md` 中的所有項目都已完成並勾選。
