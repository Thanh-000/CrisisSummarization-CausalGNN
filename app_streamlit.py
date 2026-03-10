import streamlit as st
import torch
from PIL import Image
import os
import sys

# Thêm thư mục src vào path để load model
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference import CausalCrisisInferenceEngine, CLASSES_TASK2

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Causal GNN: Disaster Classification",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện đẹp hơn
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
    .confidence-high { color: #00cc66; font-weight: bold; }
    .confidence-medium { color: #ff9900; font-weight: bold; }
    .confidence-low { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Caching
# ---------------------------------------------------------
@st.cache_resource
def load_engine():
    """Load model engine 1 lần duy nhất, chia sẻ qua session"""
    engine = CausalCrisisInferenceEngine()
    # Nếu có file weights, có thể hardcode đường dẫn vào đây:
    # engine.load_weights("path_to_weights.pth")
    return engine

# Header
st.title("🌪️ CrisisSummarization: Causal GNN Framework")
st.markdown("""
Welcome to the **Multimodal Disaster Classification** interactive demo. 
This system utilizes a **Causal Graph Neural Network (GNN)**, fortified with **Spectral-Normalized Gradient Reversal**, to analyze social media posts (Text + Image) during crises and filter out spurious background noise (e.g., weather or irrelevant objects) to predict the genuine humanitarian category.
""")

# Load mô hình dưới nền tảng (Hiển thị spinner 1 lần)
with st.spinner("Loading CLIP Processor and Causal GNN Engine... (First time only)"):
    engine = load_engine()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("This dashboard runs the **Phase 4 Inference Pipeline**.")
    st.info("""
    **Architecture:**
    1. Modality Disentanglement
    2. Causal Disentanglement (Robust to OOD)
    3. Graph Message Passing
    """)
    st.markdown("---")
    st.markdown("### Supported Classes:")
    for c in CLASSES_TASK2:
        st.caption(f"- {c.replace('_', ' ').title()}")

# ---------------------------------------------------------
# Main UI Loop
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Multimodal Data")
    
    # Text input
    input_text = st.text_area(
        "📝 Text content (Tweet/Report):", 
        value="Flood waters are rising rapidly near the downtown bridge. We need immediate rescue boats!",
        height=100
    )
    
    # Image input
    uploaded_file = st.file_uploader("🖼️ Upload incident image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.caption("👈 Please upload an image to run the multimodal analysis.")

with col2:
    st.subheader("2. Causal Analysis & Prediction")
    
    if st.button("🚀 Analyze Disaster Event", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ Please upload an image first!")
        elif not input_text.strip():
            st.warning("⚠️ Please enter some text description!")
        else:
            with st.spinner("Extracting Causal Features and Running GNN..."):
                # Ghi tạm file ảnh ra disk để inference script dùng 
                # (Vì engine hiện tại nhận image_path, nhưng ta có thể override sửa xíu trong app)
                temp_img_path = "temp_inference_img.jpg"
                image.save(temp_img_path)
                
                # Gọi Model Predict
                try:
                    # predictions trả về danh sách [(label, prob), ...] đã xếp giảm dần
                    predictions = engine.predict(temp_img_path, input_text)
                    
                    st.success("Analysis Complete!")
                    st.markdown("### Top Predictions")
                    
                    # Hiển thị trực quan từng phán đoán
                    for i, (label, prob) in enumerate(predictions):
                        clean_label = label.replace('_', ' ').upper()
                        
                        # Bar chart tuỳ biến
                        st.markdown(f"**{i+1}. {clean_label}**")
                        
                        # Màu thanh tiến trình theo mức độ tự tin
                        if prob >= 75:
                            color_class = "confidence-high"
                        elif prob >= 40:
                            color_class = "confidence-medium"
                        else:
                            color_class = "confidence-low"
                            
                        st.markdown(f"Confidence: <span class='{color_class}'>{prob:.1f}%</span>", unsafe_allow_html=True)
                        st.progress(int(prob))
                        st.write("") # Spacer
                        
                    # Dọn rác
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                        
                except Exception as e:
                    st.error(f"An error occurred during inference: {str(e)}")
                    
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built for the CrisisSummarization Research Pipeline • Phase 4 Deployment</div>", unsafe_allow_html=True)
