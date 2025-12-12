# AI / Human Text Detector (AI 文章偵測器)

這是一個基於 Streamlit 和 Transformers 的簡單 Web 應用程式，用於檢測文本是由人工智慧 (AI) 生成的還是由人類撰寫的。
部屬於: https://hw5-advanced-topic-2abq7jjte22wjxr2orgouc.streamlit.app/

## 功能特色

- **即時偵測**：使用者輸入文本後，立即顯示 AI 與人類撰寫的機率。
- **深度學習模型**：使用 Hugging Face 的 `roberta-base-openai-detector` 模型進行分析。
- **視覺化報告**：提供機率百分比、進度條以及長條圖分析。
- **簡潔 UI**：使用 Streamlit 打造的直觀介面。

## 專案結構

```
.
├── app.py              # 主程式 (Streamlit 應用)
├── requirements.txt    # 專案依賴套件清單
└── README.md           # 專案說明文件
```

## 安裝與執行

### 1. 安裝依賴套件

請確保您已安裝 Python (建議 3.8+)，然後執行以下指令安裝所需套件：

```bash
pip install -r requirements.txt
```

### 2. 執行應用程式

在終端機中執行以下指令啟動 Streamlit：

```bash
streamlit run app.py
```

啟動後，瀏覽器將自動開啟應用程式 (預設網址為 `http://localhost:8501`)。

## 部署至 Streamlit Cloud

本專案已準備好部署至 Streamlit Cloud。

1. 將專案上傳至 GitHub。
2. 登入 [Streamlit Cloud](https://streamlit.io/cloud)。
3. 選擇 "New app"。
4. 選擇您的 GitHub Repository、Branch (通常為 `main` 或 `master`) 以及主程式檔案 (`app.py`)。
5. 點擊 "Deploy"。

## 技術細節

- **Frontend**: Streamlit
- **Model**: `roberta-base-openai-detector` (GPT-2 Output Detector)
- **Backend**: PyTorch, Hugging Face Transformers

## 參考資料

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
