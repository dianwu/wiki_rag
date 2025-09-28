# MediaWiki RAG 專案

本專案使用 LangChain 實作一個 Retrieval-Augmented Generation (RAG) 系統，能以 MediaWiki 導出的 XML 檔案作為知識庫，並透過 Google Gemini 大型語言模型回答相關問題。

## 專案架構

```
.
├── data/                  # 存放 MediaWiki XML 檔案
├── .gitignore             # Git 忽略設定
├── main.py                # 主程式進入點
├── plan.md                # 開發計畫
├── readme.md              # 專案說明
└── requirements.txt       # Python 套件依賴
```

## 環境設定

1.  **建立並啟用虛擬環境**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **安裝依賴套件**

    ```bash
    pip install -r requirements.txt
    ```

3.  **設定環境變數**

    在專案根目錄建立一個 `.env` 檔案，並填入您的 Google API 金鑰：

    ```
    GOOGLE_API_KEY="your-google-api-key"
    ```

## 如何執行

1.  將您的 MediaWiki XML 匯出檔案放入 `data/` 目錄下。
2.  執行主程式，程式會自動處理資料、建立索引並啟動問答介面：

    ```bash
    python main.py
    ```

    **注意**：首次執行時，程式會處理 `data/` 中的 XML 檔案，並建立一個名為 `faiss_index` 的本地向量資料庫。這可能需要一些時間。後續執行將會直接載入此索引。
