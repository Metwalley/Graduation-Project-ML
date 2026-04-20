"""
Visual Chat UI using Streamlit
"""
import streamlit as st
import sys
from datetime import datetime

sys.path.insert(0, '.')

from rag_engine import RAGEngine

# Page config
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        direction: rtl;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    .status-box {
        background-color: #1a1a1a;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 15px;
        border-radius: 5px;
        font-size: 13px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_engine():
    """Load RAG engine once"""
    return RAGEngine()

# Sidebar - Status Panel
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # Initialize engine
    if st.session_state.engine is None:
        with st.spinner("⏳ Loading engine..."):
            st.session_state.engine = load_engine()
    
    # Status display
    status_html = f"""
    <div class="status-box">
    ╔═══════════════════════════════╗
    ║     CHATBOT STATUS            ║
    ╠═══════════════════════════════╣
    ║ Engine:    ✅ READY           ║
    ║ Model:     Llama 3.2 3B       ║
    ║ Backend:   Ollama             ║
    ║ Articles:  9 loaded           ║
    ║ RAG:       FAISS + Embeddings ║
    ║ Time:      {datetime.now().strftime("%H:%M:%S")}           ║
    ╚═══════════════════════════════╝
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### ⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "Llama 3.2")
        st.metric("Articles", "9")
    with col2:
        st.metric("Backend", "Ollama")
        st.metric("Token Limit", "∞")
    
    st.divider()
    
    st.markdown("### 💡 Try These")
    st.markdown("""
    **أسئلة مقترحة:**
    - كيف أساعد طفلي على التركيز؟
    - ما هي أعراض التوحد؟
    - تمارين لعسر القراءة
    - استراتيجيات للتعامل مع ADHD
    """)
    
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area - Chat
st.title("مساعد الدعم التعليمي لأولياء الأمور")
st.caption("Powered by Llama 3.2 3B + RAG")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 جاري التفكير..."):
            try:
                result = st.session_state.engine.process_question(prompt)
                response = result['answer']
                
                st.markdown(response)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ خطأ: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
