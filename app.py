import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt

# 1. 頁面基礎設定
st.set_page_config(
    page_title="AI/Human Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ AI vs Human 文章偵測器")
st.markdown("""
本工具使用 Transformer 模型來分析文本特徵。
請在下方輸入文章段落，系統將判斷其由 **人工智慧 (AI)** 生成的可能性。
""")

# 2. 載入模型 (關鍵：使用 @st.cache_resource 避免重複下載)
# 這裡選用 roberta-base-openai-detector，這是在 GPT-2 output 上微調過的經典模型
MODEL_NAME = "radar-ai/radar-roberta-base"

@st.cache_resource
def load_model():
    # 顯示載入中的提示
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

# 處理載入過程的 UI 提示
with st.spinner("正在啟動 AI 偵測引擎，初次載入需約 30 秒..."):
    try:
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"模型載入失敗，請檢查網路或記憶體狀態。\n錯誤訊息: {e}")
        st.stop()

# 3. 使用者介面
text_input = st.text_area(
    "輸入測試文本 (建議輸入英文，效果最佳)：", 
    height=200, 
    placeholder="Paste your text here to analyze..."
)

analyze_btn = st.button("🚀 開始偵測", use_container_width=True)

# 4. 偵測邏輯
if analyze_btn and text_input:
    if len(text_input.strip()) < 10:
        st.warning("⚠️ 文字太短，請至少輸入一個完整的句子。")
    else:
        try:
            # 將文字轉為模型可讀的格式
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            
            # 進行預測
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # 計算機率 (Softmax)
            probabilities = F.softmax(logits, dim=1).tolist()[0]
            
            # 該模型的標籤定義：Index 0 = Fake (AI), Index 1 = Real (Human)
            ai_prob = probabilities[0]
            human_prob = probabilities[1]
            
            # 5. 顯示結果
            st.divider()
            st.subheader("分析報告")

            # 判斷結果文字
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🤖 AI 生成機率", f"{ai_prob:.2%}")
            with col2:
                st.metric("🧑 人類撰寫機率", f"{human_prob:.2%}")

            # 進度條視覺化
            if ai_prob > human_prob:
                st.error(f"🚨 結論：這篇文章很高機率是 **AI 生成** 的。")
                st.progress(ai_prob, text="AI Probability")
            else:
                st.success(f"✅ 結論：這篇文章很高機率是 **人類撰寫** 的。")
                st.progress(human_prob, text="Human Probability")

            # 圖表視覺化 (加分項)
            st.write("---")
            st.caption("機率分佈圖表：")
            chart_data = pd.DataFrame({
                "來源": ["AI (Fake)", "Human (Real)"],
                "機率": [ai_prob, human_prob]
            })
            st.write("---")
            st.caption("機率分佈圖表：")
            
            # 準備資料
            chart_data = pd.DataFrame({
                "Source": ["AI (Fake)", "Human (Real)"],
                "Probability": [ai_prob, human_prob]
            })

            # 使用 Altair 來繪製，這樣可以精準指定顏色
            c = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Source', title='來源'),
                y=alt.Y('Probability', title='機率'),
                # 指定顏色：AI 用紅色 (#FF4B4B)，Human 用綠色 (#00CC96)
                color=alt.Color('Source', scale=alt.Scale(
                    domain=['AI (Fake)', 'Human (Real)'],
                    range=['#FF4B4B', '#00CC96']
                ), legend=None)
            )

            st.altair_chart(c, use_container_width=True)
            
        except Exception as e:
            st.error(f"偵測過程中發生錯誤：{e}")

# 頁尾
st.markdown("---")
st.caption("Model: `roberta-base-openai-detector` | Framework: `Streamlit`")
