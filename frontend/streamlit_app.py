import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import base64
from io import StringIO

# Configure Streamlit page
st.set_page_config(
    page_title="نظام الاسترجاع والتوليد العربي",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for Arabic text support and modern styling
st.markdown("""
<style>
    .arabic-text {
        font-family: 'Arial', 'Tahoma', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .query-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .answer-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e6f3ff;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'system_status' not in st.session_state:
    st.session_state.system_status = {}

# Helper functions
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}

def display_system_status():
    """Display system status in sidebar."""
    status = make_api_request("/health")
    
    if "error" not in status:
        if status.get("system_initialized", False):
            st.sidebar.success("✅ النظام جاهز")
        else:
            st.sidebar.error("❌ النظام غير جاهز")
        
        st.sidebar.metric("حالة النظام", "نشط" if status.get("status") == "healthy" else "غير نشط")
    else:
        st.sidebar.error("❌ خطأ في الاتصال بالخادم")

def display_query_interface():
    """Display main query interface."""
    st.markdown('<div class="main-header"><h1>🔍 نظام الاسترجاع والتوليد العربي</h1><p>Arabic Retrieval-Augmented Generation System</p></div>', unsafe_allow_html=True)
    
    # Query input
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "اطرح سؤالك هنا:",
            height=100,
            placeholder="مثال: ما هو الذكاء الاصطناعي؟",
            help="اكتب سؤالك باللغة العربية"
        )
    
    with col2:
        st.markdown("### إعدادات البحث")
        top_k = st.slider("عدد الوثائق المسترجعة", 1, 10, 5)
        max_contexts = st.slider("الحد الأقصى للسياق", 1, 5, 3)
        similarity_threshold = st.slider("عتبة التشابه", 0.0, 1.0, 0.5, 0.1)
    
    # Query button
    if st.button("🔍 البحث", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("جاري معالجة السؤال..."):
                query_data = {
                    "question": question,
                    "top_k": top_k,
                    "max_contexts": max_contexts,
                    "similarity_threshold": similarity_threshold
                }
                
                result = make_api_request("/query", method="POST", data=query_data)
                
                if "error" not in result:
                    # Display answer
                    display_query_result(result)
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "question": question,
                        "answer": result.get("answer", ""),
                        "confidence": result.get("confidence", 0),
                        "processing_time": result.get("processing_time", 0),
                        "timestamp": time.time()
                    })
                else:
                    st.error(f"خطأ: {result['error']}")
        else:
            st.warning("يرجى إدخال سؤال")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_query_result(result: Dict):
    """Display query result."""
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    
    # Answer
    st.markdown("### 📝 الإجابة")
    st.markdown(f'<div class="arabic-text" style="font-size: 1.1em; line-height: 1.6;">{result.get("answer", "لا توجد إجابة")}</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = result.get("confidence", 0)
        st.metric("درجة الثقة", f"{confidence:.2%}")
    
    with col2:
        processing_time = result.get("processing_time", 0)
        st.metric("وقت المعالجة", f"{processing_time:.2f}s")
    
    with col3:
        success = result.get("success", False)
        st.metric("حالة الاستعلام", "نجح" if success else "فشل")
    
    # Retrieved documents
    retrieved_docs = result.get("retrieved_docs", [])
    if retrieved_docs:
        st.markdown("### 📚 الوثائق المسترجعة")
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3
            with st.expander(f"وثيقة {i+1} - درجة التشابه: {doc.get('similarity_score', 0):.3f}"):
                st.markdown(f'<div class="arabic-text">{doc.get("text", "")[:500]}...</div>', unsafe_allow_html=True)
                
                # Document metadata
                metadata = doc.get("metadata", {})
                if metadata:
                    st.json(metadata)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_document_management():
    """Display document management interface."""
    st.header("📄 إدارة الوثائق")
    
    tab1, tab2, tab3 = st.tabs(["إضافة وثيقة", "رفع ملف", "إدارة قاعدة المعرفة"])
    
    with tab1:
        st.subheader("إضافة وثيقة جديدة")
        
        document_text = st.text_area(
            "نص الوثيقة:",
            height=200,
            placeholder="أدخل نص الوثيقة هنا..."
        )
        
        # Metadata
        st.subheader("معلومات إضافية")
        col1, col2 = st.columns(2)
        
        with col1:
            doc_title = st.text_input("عنوان الوثيقة")
            doc_category = st.text_input("فئة الوثيقة")
        
        with col2:
            doc_author = st.text_input("المؤلف")
            doc_source = st.text_input("المصدر")
        
        if st.button("إضافة الوثيقة"):
            if document_text.strip():
                metadata = {
                    "title": doc_title,
                    "category": doc_category,
                    "author": doc_author,
                    "source": doc_source
                }
                
                data = {
                    "text": document_text,
                    "metadata": {k: v for k, v in metadata.items() if v}
                }
                
                with st.spinner("جاري إضافة الوثيقة..."):
                    result = make_api_request("/documents/add", method="POST", data=data)
                    
                    if "error" not in result:
                        st.success("تم إضافة الوثيقة بنجاح!")
                    else:
                        st.error(f"خطأ: {result['error']}")
            else:
                st.warning("يرجى إدخال نص الوثيقة")
    
    with tab2:
        st.subheader("رفع ملف")
        
        uploaded_file = st.file_uploader(
            "اختر ملف نصي:",
            type=['txt'],
            help="يدعم النظام الملفات النصية (.txt) فقط حالياً"
        )
        
        if uploaded_file is not None:
            if st.button("رفع ومعالجة الملف"):
                with st.spinner("جاري رفع ومعالجة الملف..."):
                    # For demo purposes, we'll show the file content
                    # In production, you'd upload to the API
                    content = uploaded_file.read().decode('utf-8')
                    st.text_area("محتوى الملف:", content, height=200)
                    st.success(f"تم رفع الملف {uploaded_file.name} بنجاح!")
    
    with tab3:
        st.subheader("إدارة قاعدة المعرفة")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("بناء قاعدة المعرفة", type="primary"):
                with st.spinner("جاري بناء قاعدة المعرفة..."):
                    data = {"force_rebuild": False}
                    result = make_api_request("/knowledge-base/build", method="POST", data=data)
                    
                    if "error" not in result:
                        st.success("تم بدء بناء قاعدة المعرفة!")
                    else:
                        st.error(f"خطأ: {result['error']}")
            
            if st.button("إعادة بناء قاعدة المعرفة"):
                with st.spinner("جاري إعادة بناء قاعدة المعرفة..."):
                    data = {"force_rebuild": True}
                    result = make_api_request("/knowledge-base/build", method="POST", data=data)
                    
                    if "error" not in result:
                        st.success("تم بدء إعادة بناء قاعدة المعرفة!")
                    else:
                        st.error(f"خطأ: {result['error']}")
        
        with col2:
            if st.button("تصدير قاعدة المعرفة"):
                st.info("سيتم تصدير قاعدة المعرفة...")
            
            if st.button("إعادة تعيين قاعدة المعرفة", type="secondary"):
                if st.checkbox("أؤكد رغبتي في حذف جميع البيانات"):
                    result = make_api_request("/knowledge-base/reset", method="POST")
                    
                    if "error" not in result:
                        st.success("تم إعادة تعيين قاعدة المعرفة!")
                    else:
                        st.error(f"خطأ: {result['error']}")

def display_evaluation_interface():
    """Display evaluation interface."""
    st.header("📊 تقييم النظام")
    
    st.markdown("""
    استخدم هذا القسم لتقييم أداء النظام باستخدام مجموعة من الأسئلة الاختبارية.
    """)
    
    # Test questions input
    st.subheader("الأسئلة الاختبارية")
    
    default_questions = [
        "ما هو الذكاء الاصطناعي؟",
        "كيف تعمل معالجة اللغات الطبيعية؟",
        "ما هي تطبيقات التعلم العميق؟"
    ]
    
    test_questions = st.text_area(
        "أدخل الأسئلة الاختبارية (سؤال في كل سطر):",
        value="\n".join(default_questions),
        height=150
    )
    
    generate_answers = st.checkbox("توليد إجابات مرجعية", value=True)
    
    if st.button("بدء التقييم", type="primary"):
        questions_list = [q.strip() for q in test_questions.split('\n') if q.strip()]
        
        if questions_list:
            with st.spinner("جاري تقييم النظام..."):
                data = {
                    "test_questions": questions_list,
                    "generate_answers": generate_answers
                }
                
                result = make_api_request("/evaluate", method="POST", data=data)
                
                if "error" not in result:
                    display_evaluation_results(result)
                else:
                    st.error(f"خطأ في التقييم: {result['error']}")
        else:
            st.warning("يرجى إدخال أسئلة اختبارية")

def display_evaluation_results(result: Dict):
    """Display evaluation results."""
    st.subheader("📈 نتائج التقييم")
    
    metrics = result.get("metrics", {})
    
    # Create metrics visualization
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("معدل النجاح", f"{metrics.get('success_rate', 0):.2%}")
    
    with col2:
        st.metric("دقة الإجابة", f"{metrics.get('answer_accuracy', 0):.2%}")
    
    with col3:
        st.metric("التشابه الدلالي", f"{metrics.get('semantic_similarity', 0):.3f}")
    
    with col4:
        st.metric("متوسط وقت الاستجابة", f"{metrics.get('average_response_time', 0):.2f}s")
    
    # Detailed metrics chart
    if metrics:
        fig = go.Figure()
        
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Success Rate', 'Answer Accuracy']
        metric_values = [
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('success_rate', 0),
            metrics.get('answer_accuracy', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        ))
        
        fig.update_layout(
            title="مقاييس الأداء",
            xaxis_title="المقياس",
            yaxis_title="القيمة",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_query_history():
    """Display query history."""
    st.header("📜 سجل الاستعلامات")
    
    if st.session_state.query_history:
        # Create DataFrame
        df = pd.DataFrame(st.session_state.query_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("إجمالي الاستعلامات", len(df))
        
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("متوسط الثقة", f"{avg_confidence:.2%}")
        
        with col3:
            avg_time = df['processing_time'].mean()
            st.metric("متوسط وقت المعالجة", f"{avg_time:.2f}s")
        
        # Display history table
        st.subheader("تاريخ الاستعلامات")
        
        for i, row in df.iterrows():
            with st.expander(f"استعلام {i+1}: {row['question'][:50]}..."):
                st.markdown(f"**السؤال:** {row['question']}")
                st.markdown(f"**الإجابة:** {row['answer']}")
                st.markdown(f"**درجة الثقة:** {row['confidence']:.2%}")
                st.markdown(f"**وقت المعالجة:** {row['processing_time']:.2f} ثانية")
                st.markdown(f"**التوقيت:** {row['timestamp']}")
        
        # Clear history button
        if st.button("مسح السجل"):
            st.session_state.query_history = []
            st.success("تم مسح السجل!")
            st.experimental_rerun()
    else:
        st.info("لا يوجد سجل استعلامات حتى الآن")

def display_system_configuration():
    """Display system configuration interface."""
    st.header("⚙️ إعدادات النظام")
    
    st.markdown("قم بتخصيص إعدادات النظام حسب احتياجاتك.")
    
    # Get available models
    models = make_api_request("/models")
    
    if "error" not in models:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("نماذج التضمين")
            embedding_model = st.selectbox(
                "اختر نموذج التضمين:",
                models.get("embedding_models", []),
                index=0
            )
            
            st.subheader("استراتيجية التقسيم")
            chunking_strategy = st.selectbox(
                "اختر استراتيجية التقسيم:",
                models.get("chunking_strategies", []),
                index=1
            )
        
        with col2:
            st.subheader("نماذج اللغة")
            llm_model = st.selectbox(
                "اختر نموذج اللغة:",
                models.get("llm_models", []),
                index=0
            )
            
            st.subheader("معاملات التقسيم")
            chunk_size = st.slider("حجم القطعة", 256, 1024, 512)
            chunk_overlap = st.slider("التداخل", 0, 200, 50)
        
        # Configuration form
        if st.button("تطبيق الإعدادات", type="primary"):
            config_data = {
                "embedding_model_name": embedding_model,
                "llm_model_name": llm_model,
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            
            with st.spinner("جاري تطبيق الإعدادات..."):
                result = make_api_request("/configure", method="POST", data=config_data)
                
                if "error" not in result:
                    st.success("تم تطبيق الإعدادات بنجاح!")
                else:
                    st.error(f"خطأ: {result['error']}")

# Main app
def main():
    """Main application."""
    
    # Sidebar
    st.sidebar.title("🔍 نظام RAG العربي")
    
    # System status
    display_system_status()
    
    # Navigation
    page = st.sidebar.selectbox(
        "اختر الصفحة:",
        [
            "🏠 الصفحة الرئيسية",
            "📄 إدارة الوثائق", 
            "📊 تقييم النظام",
            "📜 سجل الاستعلامات",
            "⚙️ الإعدادات"
        ]
    )
    
    # Display selected page
    if page == "🏠 الصفحة الرئيسية":
        display_query_interface()
    
    elif page == "📄 إدارة الوثائق":
        display_document_management()
    
    elif page == "📊 تقييم النظام":
        display_evaluation_interface()
    
    elif page == "📜 سجل الاستعلامات":
        display_query_history()
    
    elif page == "⚙️ الإعدادات":
        display_system_configuration()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**نظام الاسترجاع والتوليد العربي**")
    st.sidebar.markdown("Arabic RAG System v1.0")

if __name__ == "__main__":
    main()