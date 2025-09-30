import logging
import warnings

from rag_logic import RAGSystem

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the RAG-based question answering pipeline.
    """
    try:
        rag_system = RAGSystem()
        logging.info("RAG system initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}")
        return

    # --- Interactive Query Loop ---
    print("\n--- QNAP Wiki RAG 系統已就緒 ---")
    print("模型: Google Gemini")
    print("現在您可以開始提問了 (輸入 'exit' 或 'quit' 來結束程式)。\n")

    while True:
        try:
            question = input("請輸入您的問題: ")
            if question.lower() in ["exit", "quit"]:
                print("正在關閉程式...")
                break
            if not question.strip():
                continue

            print("\n正在思考中...")

            # --- Verification Step: Retrieve and print context ---
            print("--- [驗證] 正在檢索相關資料... ---")
            retrieved_docs = rag_system.get_retrieved_docs(question)
            if not retrieved_docs:
                print("--- [驗證] 找不到相關資料。 ---")
            else:
                print(f"--- [驗證] 找到 {len(retrieved_docs)} 筆相關資料: ---")
                for i, doc in enumerate(retrieved_docs):
                    print(f"  [資料 {i+1}]")
                    print(f"    來源: {doc.metadata.get('source', 'N/A')}")
                    print(f"    內容總長度: {doc.page_content.__len__()}...")
                    print(f"    內容: {doc.page_content[:150]}...")
                print("------------------------------------")

            # Stream the response
            print("\n--- [回答] ---")
            full_response = ""
            for chunk in rag_system.get_answer_stream(question):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

        except KeyboardInterrupt:
            print("\n偵測到中斷指令，正在關閉程式...")
            break
        except Exception as e:
            logging.error(f"An error occurred during query processing: {e}")
            print(f"\n處理您的問題時發生錯誤，請稍後再試或換個問題。 錯誤訊息: {e}")
            continue


if __name__ == "__main__":
    main()