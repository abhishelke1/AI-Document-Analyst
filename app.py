# --- 1. SQLite Fix for Streamlit Cloud (MUST BE AT THE VERY TOP) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Standard Imports ---
import streamlit as st
import os
import tempfile

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Try importing from the specific module if the main one fails
try:
    from langchain.chains import create_retrieval_chain
except ImportError:
    from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 3. Page Configuration (Make it look professional) ---
st.set_page_config(
    page_title="AI Document Analyst",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stChatMessage { font-size: 1.1rem; }
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
    st.markdown("**Status:**")
    
    # Initialize session state for storing the vectorstore (so we don't reload it every time)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

# --- 5. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a PDF, and I can answer detailed questions about it."}
    ]

# --- 6. Process the PDF (Only runs once when file is uploaded) ---
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

# --- 8. Chat Logic (The Core) ---
# We check if user typed something AND if we have a document ready
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
                
                # Retrieve API Key from Secrets (or hardcoded fallback for safety)
                # Ideally use: api_key = st.secrets["OPENROUTER_API_KEY"]
                api_key = st.secrets.get("OPENROUTER_API_KEY", "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")
                
                # Initialize LLM (Using a better model for detailed replies)
                llm = ChatOpenAI(
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_name="openai/gpt-4o-mini", # Smart & Fast
                    temperature=0.3
                )

                # Create Retriever
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

                # Enhanced System Prompt for "Better Replies"
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

                # Get Answer
                response = retrieval_chain.invoke({"input": prompt})
                answer = response['answer']

                st.markdown(answer)
                
                # Append assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})