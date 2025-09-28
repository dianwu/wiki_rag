# RAG 專案開發計畫

本計畫旨在建立一個基於 LangChain 的 RAG (Retrieval-Augmented Generation) 系統，資料來源為 MediaWiki 導出的 XML 檔案。

## 第一階段：資料載入與處理

1.  **載入 XML**：從 `data/` 目錄讀取 MediaWiki XML 檔案。
2.  **解析 XML**：使用 `lxml` 或類似套件解析 XML，提取每個頁面的標題（title）和內容（text）。
3.  **內容清理**：從頁面內容中移除 MediaWiki 標記和 HTML 標籤，只保留純文字。`BeautifulSoup` 會是個好工具。
4.  **文件分割 (Chunking)**：將清理後的長文本分割成較小的、語意完整的段落，以便於後續的向量化處理。

## 第二階段：向量化與儲存

1.  **選擇 Embedding 模型**：使用 `langchain-google-genai` 的 `OpenAIEmbeddings` 將文本段落轉換為向量。
2.  **建立向量資料庫**：使用 `Chroma` 作為本地向量資料庫，它易於設置和使用。
3.  **儲存向量**：將所有文本段落的向量儲存到 Chroma 資料庫中，並將其持久化到本地磁碟，以便重複使用。

## 第三階段：檢索與生成 (RAG Chain)

1.  **建立檢索器 (Retriever)**：基於儲存的 Chroma 資料庫建立一個檢索器，該檢索器能根據使用者問題的向量，快速找到最相關的文本段落。
2.  **設計提示模板 (Prompt Template)**：設計一個提示，該提示包含三個部分：
    *   **上下文 (Context)**：從檢索器返回的相關文本段落。
    *   **問題 (Question)**：使用者的原始問題。
    *   **指令 (Instruction)**：指示大型語言模型 (LLM) 根據提供的上下文來回答問題。
3.  **建立 RAG 鏈**：使用 LangChain Expression Language (LCEL) 將檢索器、提示模板和大型語言模型（如 OpenAI 的 GPT）串聯起來，形成一個完整的 RAG 鏈。

## 第四階段：使用者介面與互動

1.  **建立主程式 (`main.py`)**：
    *   初始化 RAG 鏈。
    *   提供一個簡單的命令列介面 (CLI)，讓使用者可以輸入問題。
    *   接收問題，呼叫 RAG 鏈，並將生成的答案輸出到控制台。
2.  **（可選）Web 介面**：未來可考慮使用 Streamlit 或 FastAPI 建立一個簡單的 Web 應用，提供更友好的互動體驗。
