import streamlit as st
import os
import tempfile

# --- 1. Smart SQLite Fix (Works on Windows & Cloud) ---
# This tries to use the cloud-fix. If it fails (on Windows), 
# it safely ignores it and uses your local SQLite.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- 2. LangChain Imports (Robust Fix) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- IMPORT FIX FOR STREAMLIT CLOUD ---
# This handles version mismatches on the cloud server automatically.
try:
    from langchain.chains import create_retrieval_chain
except ImportError:
    # Fallback for slightly different library versions
    from langchain.chains.retrieval import create_retrieval_chain

# --- 3. Page Configuration ---
st.set_page_config(
    page_title="AI Document Analyst",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stChatMessage { font-size: 1.05rem; }
    div[data-testid="stSidebar"] { background-color: #f7f9fc; }
    h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Document Analyst")
st.markdown("Upload a document and ask detailed questions. I analyze the content in real-time.")

# --- 4. Sidebar for File Upload ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    st.markdown("---")
    
    # Initialize session state for vector_db
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

# --- 5. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a PDF, and I can answer detailed questions about it."}
    ]

# --- 6. Process the PDF ---
if uploaded_file and st.session_state.vector_db is None:
    with st.spinner("🧠 Processing document... (Splitting & Embedding)"):
        try:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load & Split
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Embed (Using Free HuggingFace Model - Fast & Local)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create Vector Store
            st.session_state.vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            st.sidebar.success(f"✅ Indexed {len(splits)} chunks.")
            os.remove(tmp_path) # Cleanup

        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# --- 7. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. Chat Logic ---
if prompt := st.chat_input("Ask a specific question about the document..."):
    
    if st.session_state.vector_db is None:
        st.error("Please upload a PDF first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                
                # --- SECURE KEY HANDLING ---
                # We check st.secrets for the key. We DO NOT hardcode it here.
                api_key = st.secrets.get("OPENROUTER_API_KEY")
                
                if not api_key:
                    st.error("API Key missing! Please add 'OPENROUTER_API_KEY' to your secrets.toml file or Streamlit Cloud secrets.")
                    st.stop()
                
                # Initialize LLM
                llm = ChatOpenAI(
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name="openai/gpt-3.5-turbo", # Reliable & Fast
                    temperature=0.3
                )

                # Create Retriever
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

                # System Prompt
                system_prompt = (
                    "You are a highly intelligent AI analyst. "
                    "You have been provided with context from a document. "
                    "Answer the user's question based strictly on the context provided. "
                    "Provide a detailed, well-structured explanation. "
                    "If the answer is complex, break it down into bullet points. "
                    "If the context does not contain the answer, explicitly say so."
                    "\n\n"
                    "Context:\n{context}"
                )

                # Build Chain
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                
                combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                try:
                    # Get Answer
                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    
                    # Append assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Error generating response: {e}")