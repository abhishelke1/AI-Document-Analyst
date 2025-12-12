import streamlit as st
import os
import tempfile

# --- 1. Smart SQLite Fix (Works on Windows & Cloud) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- 2. LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# CHANGED: Import Hugging Face classes instead of OpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- IMPORT FIX FOR STREAMLIT CLOUD ---
try:
    from langchain.chains import create_retrieval_chain
except ImportError:
    from langchain.chains.retrieval import create_retrieval_chain

# --- 3. Page Configuration ---
st.set_page_config(
    page_title="AI Document Analyst (Hugging Face)",
    page_icon="🤗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { font-size: 1.05rem; }
    div[data-testid="stSidebar"] { background-color: #f7f9fc; }
    h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("🤗 AI Document Analyst")
st.markdown("Upload a document and ask detailed questions. Powered by **Llama-3 via Hugging Face**.")

# --- 4. Sidebar for File Upload ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    st.markdown("---")
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

# --- 5. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm running on Hugging Face servers. Upload a PDF to start!"}
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
            with st.spinner("Thinking (Hugging Face API)..."):
                
                # --- CHANGED: HUGGING FACE SETUP ---
                # Get the Hugging Face Token from Secrets
                hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
                
                if not hf_token:
                    st.error("Missing Hugging Face Token! Add 'HUGGINGFACEHUB_API_TOKEN' to your secrets.toml.")
                    st.stop()
                
                # 1. Define the Repo ID (Using Llama 3 8B Instruct - It is free and smart)
                repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

                # 2. Initialize the Endpoint
                llm = HuggingFaceEndpoint(
                    repo_id=repo_id,
                    max_new_tokens=512,
                    temperature=0.3,
                    huggingfacehub_api_token=hf_token,
                )

                # 3. Wrap in ChatHuggingFace for better prompt handling
                chat_model = ChatHuggingFace(llm=llm)

                # Create Retriever
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

                # System Prompt
                system_prompt = (
                    "You are a helpful AI assistant. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "\n\n"
                    "Context:\n{context}"
                )

                # Build Chain
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                
                combine_docs_chain = create_stuff_documents_chain(chat_model, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                try:
                    # Get Answer
                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    
                    # Append assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Note: The free Hugging Face API has rate limits. If you see a 503 error, wait a moment and try again.")