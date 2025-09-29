## 2025-09-29 (續)

### 16. 修復 `pytest` 測試失敗問題

- **動機**: 在將資料處理邏輯拆分至 `ingest.py` 並為其建立測試後，`pytest` 中出現了一個持續的測試失敗，需要解決以確保程式碼品質。
- **問題分析**:
    - 執行 `./.venv/bin/python -m pytest` 後，發現 `tests/test_ingest.py` 中的 `test_load_and_parse_xml` 測試失敗。
    - 錯誤為 `AssertionError: assert 'line break' in 'Some more text here. '`。
    - 經追查，問題發生在 `ingest.py` 的 `load_and_parse_xml` 函式中。原本的程式碼 `text_elem.text` 在解析 XML 時，只能獲取元素的直接文本，無法讀取像 `<br />` 這樣的子標籤及其後的文本內容。
- **操作**:
    - **修改 `ingest.py`**:
        - 將 `load_and_parse_xml` 函式中的文本提取邏輯從 `text_elem.text` 修改為 `"".join(text_elem.itertext())`。
        - 這個變更確保了 lxml 解析器能夠遞迴地提取出一個元素及其所有子孫節點中的全部文本內容。
    - **驗證修復**:
        - 再次執行 `./.venv/bin/python -m pytest`。
        - 所有 5 個測試（`5 passed`）全部成功通過。
- **結論**: XML 解析的錯誤已成功修復，確保了資料載入的完整性。專案的測試套件現在是穩定的。

### 15. 完成 RAG 鏈並整合 Ollama

- **動機**: 根據計畫完成第三和第四階段，建立一個完整、可互動的 RAG 系統。
- **操作**:
    - **技術棧選擇**: 根據使用者偏好，選擇使用 `Ollama` 在本地端運行大型語言模型（LLM），以達成系統的完全本地化。
    - **更新依賴**:
        - 安裝 `langchain-ollama` 套件以支援 Ollama。
        - 由於先前的 `pip freeze` 操作覆蓋了依賴，重新確認並安裝了被遺漏的 `beautifulsoup4` 套件。
    - **實作 RAG 鏈**:
        - 在 `main.py` 中，匯入 `Ollama`, `PromptTemplate`, `StrOutputParser`, 和 `RunnablePassthrough`。
        - 實作了完整的 RAG 鏈，包含：
            1.  從 ChromaDB 建立 `retriever`。
            2.  初始化 `Ollama` LLM，並將模型設定為 `wangshenzhi/gemma2-9b-chinese-chat`。
            3.  設計了一個詳細的繁體中文提示模板，指導模型如何根據上下文回答問題。
            4.  使用 LangChain Expression Language (LCEL) 將所有元件串聯起來。
    - **建立互動介面**:
        - 在 `main.py` 中加入了一個 `while True` 迴圈，讓使用者可以在程式執行後，於命令列中持續輸入問題並獲得即時的串流回覆。
        - 加入了 `exit` 和 `quit` 指令來正常關閉程式。
- **遇到的問題與解決方案**:
    1.  **問題**: 使用者回報 `ollama pull` 指令失敗，錯誤為 `file does not exist`。
    2.  **解決方案**: 經查證，發現原先的 `aiyah/Gemma-2B-Traditional-Chinese-Taiwan-v0.1` 是 Hugging Face ID。推薦並更換為 Ollama 模型庫中專為中文優化的 `wangshenzhi/gemma2-9b-chinese-chat` 模型，解決了此問題。
    3.  **問題**: 執行 `main.py` 時出現 `ModuleNotFoundError: No module named 'bs4'`。
    4.  **解決方案**: 分析後發現，雖然 `requirements.txt` 中包含 `beautifulsoup4`，但執行時使用的是系統預設的 Python，而非專案的虛擬環境。透過使用虛擬環境的完整路徑 (`./.venv/bin/python3 main.py`) 成功執行程式，解決了環境問題。
- **結論**: 專案的核心 RAG 功能已全部完成。系統現在能夠載入本地知識庫，並透過本地 LLM 提供問答服務。使用者可直接在終端機中與系統互動。

## 2025-09-29

### 14. 增強裝置設定的靈活性

- **動機**: 取代原先硬編碼的裝置設定 (`cuda`)，提供更靈活、可攜的設定方式。
- **操作**:
    - **更新 `main.py`**:
        - 實作了新的邏輯，在程式啟動時讀取名為 `EMBEDDING_DEVICE` 的環境變數。
        - 支援 `auto` (預設), `cpu`, `cuda`/`gpu` 三種模式。
        - 在 `auto` 模式下，程式會使用 `torch.cuda.is_available()` 自動偵測並選用最適合的裝置。
    - **更新 `requirements.txt`**:
        - 明確加入 `torch` 套件，以確保 CUDA 偵測功能正常運作。
    - **更新 `.env.example`**:
        - 新增 `EMBEDDING_DEVICE=auto` 變數，作為設定範例。
- **結論**: 應用程式現在可以自動適應不同的硬體環境，同時允許使用者透過環境變數強制指定運算裝置，大幅提升了可用性與便利性。

### 13. 效能標竿測試：GPU 向量化

- **硬體**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **模型**: `all-mpnet-base-v2`
- **執行結果**:
    - **輸入資料**: 1.2 GB 的 MediaWiki XML 檔案。
    - **處理時間**: 約 34 分鐘。
    - **輸出資料庫**: 產生的 ChromaDB 大小為 104 MB。
- **結論**: 在此硬體配置下，使用 GPU 進行本地端 embedding 的效能基準已建立。

### 12. 啟用 GPU 進行向量化

- **動機**: 為了加速 `sentence-transformers` 模型的向量化過程，需要將計算從 CPU 切換到 GPU。
- **操作**:
    - **更新 `main.py`**: 在 `HuggingFaceEmbeddings` 的初始化過程中，將 `model_kwargs` 的 `device` 參數從 `'cpu'` 修改為 `'cuda'`。
- **結論**: 程式現在被設定為使用 GPU 進行 embedding 計算，預期能大幅提升處理大量文件時的效能。此變更需要環境支援 CUDA。

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
