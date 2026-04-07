import streamlit as st
from transformers import pipeline

# --- 1. SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Text Classifier Pro", page_icon="🏷️", layout="wide")

# CSS
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
    }
    .stTextArea textarea {
        background-color: #1c1f26;
        color: white;
        border-radius: 10px;
        border: 1px solid #2d3139;
    }
    .stTextInput input {
        background-color: #1c1f26;
        color: white;
        border-radius: 10px;
        border: 1px solid #2d3139;
    }
    h1, h2, h3 {
        color: #ffaa00 !important; /* ใช้สีส้มทองให้ตัดกับสีฟ้าของตัวนู้น */
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #ffaa00;
        color: black;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ffcc66;
        color: black;
    }
    /* แต่งส่วนแสดงผล Confidence */
    .stInfo {
        background-color: #1c1f26;
        border-left: 5px solid #ffaa00;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    # ใช้ BART สำหรับ Zero-shot classification
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🧠 NLP Engine")
    st.divider()
    st.write("📌 **How to use:**")
    st.caption("1. กรอกประโยคภาษาอังกฤษที่ต้องการ")
    st.caption("2. กำหนดหมวดหมู่ที่ต้องการแยก (Labels)")
    st.caption("3. กดวิเคราะห์เพื่อดูคะแนนความมั่นใจ")

# --- 4. MAIN CONTENT ---
st.title("🏷️ Smart Text Classifier Pro")
st.write("ระบบประมวลผลภาษาธรรมชาติเพื่อจัดหมวดหมู่ข้อความอัจฉริยะ")
st.write("---")

# แบ่ง Layout เป็น 2 ฝั่ง
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("⌨️ Input Data")
    text_input = st.text_area("กรอกข้อความที่ต้องการวิเคราะห์ (English Only):", 
                              "I am looking for a high-performance GPU for my machine learning projects.",
                              height=150)
    
    labels_input = st.text_input("กำหนดหมวดหมู่ (แยกด้วยเครื่องหมาย , )", 
                                 "Technology, Shopping, Gaming, Education")
    
    analyze_btn = st.button("วิเคราะห์ผลลัพธ์")

with col_result:
    st.subheader("Analysis Result")
    
    if analyze_btn:
        if text_input and labels_input:
            candidate_labels = [label.strip() for label in labels_input.split(",")]
            
            with st.spinner('AI กำลังประมวลผลความหมายเชิงลึก...'):
                res = classifier(text_input, candidate_labels)
                
            top_label = res['labels'][0]
            top_score = res['scores'][0]
            
            # แสดงผลลัพธ์แบบเด่นๆ
            st.markdown(f"### ผลลัพธ์ที่แม่นยำที่สุด: **{top_label}**")
            st.info(f"ระดับความมั่นใจ (Confidence Score): {top_score:.2%}")
            
            # กราฟแสดงคะแนนทั้งหมด
            st.divider()
            st.write("คะแนนเปรียบเทียบแต่ละหมวดหมู่:")
            chart_data = dict(zip(res['labels'], res['scores']))
            st.bar_chart(chart_data)
        else:
            st.warning("⚠️ กรุณากรอกข้อมูลให้ครบถ้วนในฝั่ง Input")
    else:
        st.write("รอการวิเคราะห์ข้อมูล...")
        st.image("https://cdn-icons-png.flaticon.com/512/2621/2621033.png", width=150)

# --- 5. FOOTER ---
st.divider()
st.caption("Natural Language Processing Implementation | for studying")