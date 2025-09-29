# 下階段任務：修復測試並推進至向量化

## 第一優先：修復資料處理的單元測試

- [x] **任務 1.1 (高優先級)**: **偵錯並修復 `tests/test_main.py`**
    - **問題描述**: `clean_mediawiki_text` 函式中的一個正規表示式在 `pytest` 環境下會導致 `re.error: unbalanced parenthesis`。這阻礙了所有後續的驗證工作。
    - **建議**: 
        1.  需要一位熟悉 Python `re` 模組內部機制和字串轉義的專家來審查 `main.py` 中的 `clean_mediawiki_text` 函式，特別是處理 wiki 連結的正規表示式。
        2.  在本地環境中隔離執行 `pytest`，並對 `sre_parse.py` 的錯誤進行更深入的偵錯。
        3.  在修復 `re.error` 後，重新審查 `test_load_and_parse_xml` 的模擬，確保它能正確運作。
    - **目標**: 讓 `.venv/bin/pytest` 中的所有測試都能成功通過。

## 第二階段：向量化與儲存 (Chroma) - 已完成

*注意：此階段的程式碼已完成，但因外部 Google API Quota 問題而受阻。*

- [x] **任務 2.1**: **環境與組態設定**
    - 在 `main.py` 中，匯入 `dotenv` 並呼叫 `load_dotenv()` 來載入環境變數。
    - 匯入 `GoogleGenerativeAIEmbeddings` 和 `Chroma`。

- [x] **任務 2.2**: **建立向量資料庫 (Chroma)**
    - 建立一個名為 `create_vector_store(documents, embeddings, persist_directory)` 的函式。
    - 在函式中，使用 `Chroma.from_documents()` 方法，傳入 `documents`、`embeddings` 模型和 `persist_directory`，來建立 Chroma 資料庫。
    - 函式返回建立的 `Chroma` 資料庫物件。

- [x] **任務 2.3**: **整合儲存與載入資料庫的邏輯**
    - 在 `main.py` 的主執行區塊中，定義持久化目錄變數 `PERSIST_DIR = "chroma_db"`。
    - 初始化 `GoogleGenerativeAIEmbeddings` 模型。
    - **如果資料庫不存在** (`os.path.exists(PERSIST_DIR)` 為 `False`)：
        - 執行資料處理流程以獲得 `documents`。
        - 呼叫 `create_vector_store(documents, embeddings, PERSIST_DIR)` 來建立新的向量資料庫。
    - **如果資料庫已存在**：
        - 使用 `Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)` 來載入本地資料庫。
    - 將載入或建立的向量資料庫物件指派給一個變數 (例如 `vector_db`) 以供後續階段使用，並印出成功訊息。

## 第三階段：檢索與生成 (RAG Chain)

- [x] **任務 3.1**: **建立檢索器 (Retriever)**
- [x] **任務 3.2**: **設計提示模板 (Prompt Template)**
- [x] **任務 3.3**: **建立 RAG 鏈**