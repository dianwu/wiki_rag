# MediaWiki RAG 專案

本專案使用 LangChain 實作一個 Retrieval-Augmented Generation (RAG) 系統，能以 MediaWiki 導出的 XML 檔案作為知識庫。預設情況下，它使用開源的 Sentence-Transformers 模型在**本地端**進行文本向量化，並可選擇性地設定使用 Google Gemini 大型語言模型進行最終的答案生成。

## 專案架構

```
.
├── data/                  # 存放 MediaWiki XML 檔案
├── chroma_db/             # 持久化的向量資料庫
├── .gitignore             # Git 忽略設定
├── main.py                # 主程式進入點
├── plan.md                # 開發計畫
├── readme.md              # 專案說明
└── requirements.txt       # Python 套件依賴
```

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

1.  將您的 MediaWiki XML 匯出檔案放入 `data/` 目錄下。
2.  執行主程式：

    ```bash
    python3 main.py
    ```

    **注意**：
    *   首次執行時，程式會自動從 Hugging Face Hub 下載所需的 Embedding 模型 (例如 `all-mpnet-base-v2`)，這可能需要幾分鐘時間。
    *   下載模型後，程式會處理 `data/` 中的 XML 檔案，並建立一個名為 `chroma_db` 的本地向量資料庫。此過程也可能需要一些時間。
    *   後續執行將會直接載入已快取的模型和 `chroma_db` 資料庫。

## Embedding 模型設定

本專案預設使用免費的本地端模型進行 Embedding，但您也可以輕易地將其修改為使用線上的 API 服務。

### 使用本地端模型 (預設)

目前的 `main.py` 設定為使用 `HuggingFaceEmbeddings`。

```python
# in main.py
from langchain_community.embeddings import HuggingFaceEmbeddings

# ...

embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
```

- **優點**: 免費、資料不離開本地、一次下載後可離線運作。
- **缺點**: 需要在本地消耗 CPU/RAM 資源。

### 切換為線上 API 模型 (例如 Google)

如果您希望使用 Google 的線上 Embedding 服務，可以修改 `main.py`。

1.  **安裝對應套件**:
    ```bash
    pip install langchain-google-genai
    ```

2.  **修改程式碼**:
    在 `main.py` 中，註解掉 `HuggingFaceEmbeddings` 的部分，並換成 `GoogleGenerativeAIEmbeddings`。

    ```python
    # in main.py
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