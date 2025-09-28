# 第二階段任務：向量化與儲存

這是對 `plan.md` 中第二階段「向量化與儲存」的細化步驟。

### 1. 環境與組態設定

- [ ] **任務 1.1**: 確保 `.env` 檔案已建立且 `GOOGLE_API_KEY` 已正確設定。
- [ ] **任務 1.2**: 在 `main.py` 中，匯入 `dotenv` 並呼叫 `load_dotenv()` 來載入環境變數。
- [ ] **任務 1.3**: 在 `main.py` 中，匯入 `GoogleGenerativeAIEmbeddings` 和 `FAISS`。

### 2. 建立向量資料庫 (Vector Store)

- [ ] **任務 2.1**: 建立一個名為 `create_vector_store(documents, embeddings)` 的函式。
- [ ] **任務 2.2**: 在函式中，使用 `FAISS.from_documents()` 方法，傳入 `documents` 和 `embeddings` 模型，來建立 FAISS 索引。
- [ ] **任務 2.3**: 函式返回建立的 `FAISS` 索引物件。

### 3. 儲存與載入索引

- [ ] **任務 3.1**: 在 `main.py` 的主執行區塊中，定義索引路徑變數 `INDEX_PATH = "faiss_index"`。
- [ ] **任務 3.2**: 初始化 `GoogleGenerativeAIEmbeddings` 模型。
- [ ] **任務 3.3**: **如果索引不存在** (`os.path.exists(INDEX_PATH)` 為 `False`)：
    - 執行第一階段的完整資料處理流程以獲得 `documents`。
    - 呼叫 `create_vector_store(documents, embeddings)` 來建立新的向量資料庫。
    - 使用 `db.save_local(INDEX_PATH)` 將新建立的索引儲存到本地磁碟。
- [ ] **任務 3.4**: **如果索引已存在**：
    - 使用 `FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)` 來載入本地索引。
    - **注意**: `FAISS.load_local` 需要 `allow_dangerous_deserialization=True` 參數。
- [ ] **任務 3.5**: 將載入或建立的向量資料庫物件指派給一個變數 (例如 `vector_db`) 以供後續階段使用，並印出成功訊息。
