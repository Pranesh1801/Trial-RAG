import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="Demo 8: Intelligent PDF Q&A with LLM", layout="wide")

st.title("🧠 Demo 8: Intelligent PDF Q&A with LLM")
st.markdown("**RAG with real language understanding - Upload PDF and ask anything!**")

# LLM Integration
def generate_with_llm(query, context, api_key=None, temp=0.7):
    """Generate intelligent answer using MLVoca FREE API (NO KEY NEEDED!)"""  
    try:
        import requests
        import json
        
        prompt = f"""Answer this question based on the context below.

Context: {context[:1500]}

Question: {query}

Answer:"""
        
        API_URL = "https://mlvoca.com/api/generate"
        
        response = requests.post(
            API_URL,
            json={"model": "tinyllama", "prompt": prompt, "options": {"temperature": temp}},
            timeout=30
        )
        
        if response.status_code == 200:
            answer = ""
            for line in response.text.strip().split('\n'):
                if line.strip():
                    answer += json.loads(line).get('response', '')
            tokens = len(prompt.split()) + len(answer.split())
            return answer.strip(), tokens
        else:
            return f"Error {response.status_code}: {response.text[:300]}", 0
            
    except Exception as e:
        return f"Error: {str(e)}", 0

# Chunking
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# PDF extraction
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.success("✅ Using FREE MLVoca API - No key needed!")
    
    st.divider()
    
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        if st.button("🔄 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Extracting text from PDF..."):
                text = extract_pdf_text(uploaded_file)
                st.session_state.chunks = chunk_text(text)
                st.session_state.pdf_processed = True
                st.success(f"✅ Extracted {len(st.session_state.chunks)} chunks!")
    
    st.divider()
    
    if st.session_state.pdf_processed:
        st.header("❓ Ask Questions")
        query = st.text_area("Your Question:", "What is this document about?", height=100)
        top_k = st.slider("Top-K Chunks:", 1, 5, 3)
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
        
        ask_btn = st.button("🧠 Ask with LLM", type="primary", use_container_width=True)
    else:
        ask_btn = False

if st.session_state.pdf_processed and ask_btn:
    chunks = st.session_state.chunks
    
    st.header(f"🔍 Query: {query}")
    
    # Progress
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Step 1: Embedding
    status.text("Step 1/3: Creating embeddings...")
    progress_bar.progress(33)
    start = time.time()
    vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
    doc_embeddings = vectorizer.fit_transform(chunks).toarray()
    query_embedding = vectorizer.transform([query]).toarray()[0]
    emb_time = time.time() - start
    
    # Step 2: Retrieval
    status.text("Step 2/3: Searching document...")
    progress_bar.progress(66)
    start = time.time()
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    search_time = time.time() - start
    
    # Step 3: LLM Generation
    status.text("Step 3/3: Generating intelligent answer...")
    progress_bar.progress(100)
    start = time.time()
    context = "\n\n".join([chunks[i] for i in top_indices])
    answer, tokens = generate_with_llm(query, context, temp=temperature)
    gen_time = time.time() - start
    
    progress_bar.empty()
    status.empty()
    
    total_time = emb_time + search_time + gen_time
    
    st.divider()
    
    # Display Answer
    st.subheader("🧠 Intelligent Answer")
    
    st.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 25px; border-radius: 10px; border-left: 5px solid #44ff44;">
        <p style="color: #ffffff; font-size: 16px; line-height: 1.8;">{answer}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Embedding", f"{emb_time*1000:.0f}ms")
    col2.metric("Retrieval", f"{search_time*1000:.0f}ms")
    col3.metric("LLM Generation", f"{gen_time:.2f}s")
    col4.metric("Total Time", f"{total_time:.2f}s")
    col5.metric("Tokens Used", tokens)
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Retrieved Context", "📊 Similarity Scores", "💡 How It Works"])
    
    with tab1:
        st.subheader("Retrieved Chunks Fed to LLM")
        
        for rank, idx in enumerate(top_indices, 1):
            with st.expander(f"📄 Chunk {rank} (Similarity: {similarities[idx]:.4f})", expanded=(rank==1)):
                st.text_area(
                    f"Content",
                    chunks[idx],
                    height=150,
                    key=f"chunk_{idx}",
                    label_visibility="collapsed"
                )
    
    with tab2:
        st.subheader("Relevance Scores")
        
        fig = go.Figure(data=[go.Bar(
            x=[f"Chunk {i+1}" for i in top_indices],
            y=[similarities[i] for i in top_indices],
            marker_color=['#44ff44' if similarities[i] > 0.3 else '#ffaa44' for i in top_indices],
            text=[f"{similarities[i]:.4f}" for i in top_indices],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Similarity Scores of Retrieved Chunks",
            xaxis_title="Chunk",
            yaxis_title="Similarity Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Analysis:**
        - Average Similarity: {np.mean([similarities[i] for i in top_indices]):.4f}
        - Max Similarity: {np.max([similarities[i] for i in top_indices]):.4f}
        - Min Similarity: {np.min([similarities[i] for i in top_indices]):.4f}
        """)
    
    with tab3:
        st.subheader("🧠 How This Works")
        
        st.markdown(f"""
        ### The Complete RAG Pipeline:
        
        1. **Document Processing**
           - PDF text extracted
           - Split into 500-word chunks
           - 100-word overlap for context
        
        2. **Query Embedding**
           - Your question converted to 384D vector
           - TF-IDF semantic representation
        
        3. **Semantic Search**
           - Compare query vector with all chunk vectors
           - Cosine similarity scoring
           - Retrieve top {top_k} most relevant chunks
        
        4. **LLM Generation**
           - Retrieved chunks sent as context
           - AI model understands and synthesizes
           - Generates human-like intelligent answer
           - Temperature {temperature} for creativity control
        
        ### Why This is Better:
        
        - Real Understanding: LLM comprehends the content
        - Synthesis: Combines information from multiple chunks
        - Natural Language: Explains in clear human terms
        - Context Aware: Answers based on actual document content
        - Grounded: Cites retrieved information reduces hallucination
        
        ### Cost Consideration:
        - Tokens used: {tokens}
        - 100% FREE - No API key required!
        - Uses MLVoca API
        """)

elif st.session_state.pdf_processed:
    st.info("👈 Enter your question and click 'Ask with LLM'!")
    
    st.markdown(f"""
    ## 📄 PDF Processed Successfully!
    
    **Chunks extracted**: {len(st.session_state.chunks)}
    
    ### 🎯 What Makes This Demo Special:
    
    1. **Real LLM Integration**
       - Uses FREE AI model
       - Actual language understanding
       - Intelligent synthesis
    
    2. **Smart Retrieval**
       - TF-IDF semantic search
       - Top-K most relevant chunks
       - Context-aware selection
    
    3. **Production-Ready**
       - Complete RAG pipeline
       - Performance metrics
       - Cost tracking
    
    ### 💡 Try These Questions:
    
    **General:**
    - "What is this document about?"
    - "Summarize the main points"
    
    **Specific:**
    - "What is a specific concept?"
    - "How does something work?"
    - "Who are the authors?"
    
    **Analytical:**
    - "What are the key contributions?"
    - "What problem does this solve?"
    - "Compare X and Y"
    
    The LLM will understand context and provide intelligent synthesized answers!
    """)

else:
    st.info("👈 Upload a PDF file to get started!")
    
    st.markdown("""
    ## 🧠 Intelligent PDF Q&A with LLM
    
    ### What This Demo Does:
    
    This is a **production-grade RAG system** that:
    
    1. **Extracts** text from any PDF
    2. **Chunks** document intelligently
    3. **Embeds** text for semantic search
    4. **Retrieves** most relevant sections
    5. **Generates** intelligent answers using AI
    
    ### Why Use an LLM?
    
    **Without LLM (Demo 7):**
    - Just returns raw text chunks
    - No understanding or synthesis
    - Cannot explain concepts
    - Dumps technical formulas
    
    **With LLM (This Demo):**
    - Understands the content
    - Synthesizes information
    - Explains in clear language
    - Answers intelligently
    
    ### Example Difference:
    
    **Question:** "What are embeddings?"
    
    **Without LLM:**
    > "max(0,xW1+b1)W2+b2 (2) While the linear transformations..."
    
    **With LLM:**
    > "Embeddings are vector representations of text that capture semantic meaning. 
    > In this paper they use learned embeddings to convert input tokens into 
    > continuous vectors of dimension 512 which allows the model to process 
    > and understand the relationships between words."
    
    ### 🚀 Ready to Try?
    
    1. Upload any PDF
    2. Ask questions naturally
    3. Get intelligent answers!
    4. 100% FREE - No API key needed!
    """)

st.divider()
st.caption("🧠 Intelligent PDF Q&A Demo - RAG with real language understanding (FREE)")
