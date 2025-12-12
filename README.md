# 🤖 AI Document Analyst

An intelligent document analysis application that allows you to upload PDF documents and chat with them using AI. The application uses a RAG (Retrieval-Augmented Generation) pipeline to provide accurate, context-aware answers based on your document's content.

## ✨ Features

- **📄 PDF Upload**: Easily upload any PDF document for analysis
- **🔍 RAG Pipeline**: Uses LangChain and ChromaDB to intelligently index and retrieve relevant information
- **🤖 AI-Powered**: Utilizes OpenRouter's GPT-4o-mini for fast and intelligent responses
- **💬 Interactive Chat**: Clean and modern chat interface built with Streamlit
- **🚀 Local Embeddings**: Uses HuggingFace's sentence-transformers for fast, local text embeddings
- **💾 Vector Storage**: ChromaDB for efficient similarity search

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher** ([Download Python](https://www.python.org/downloads/))
- **pip** (Python package manager, comes with Python)
- **OpenRouter API Key** (for LLM access)

## 🛠️ Installation

Follow these steps to set up the project on your local machine:

### Step 1: Clone or Download the Repository

```bash
# If using Git
git clone <your-repository-url>
cd "AI Project"

# Or download and extract the ZIP file, then navigate to the folder
```

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps keep your project dependencies isolated:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required Python packages:

```bash
python -m pip install -r requirements.txt
```

This will install:
- `streamlit` - Web application framework
- `langchain` - LLM orchestration framework
- `langchain-community` - Community integrations
- `langchain-openai` - OpenAI/OpenRouter integration
- `langchain-huggingface` - HuggingFace embeddings
- `chromadb` - Vector database
- `pypdf` - PDF processing
- `sentence-transformers` - Text embedding models

### Step 4: Configure API Key

The application requires an OpenRouter API key to function. You have two options:

#### Option A: Using Secrets File (Recommended)

1. The `.streamlit/secrets.toml` file should already exist with the API key
2. If not, create the `.streamlit` directory and `secrets.toml` file:

```bash
# Windows
mkdir .streamlit
echo OPENROUTER_API_KEY = "your-api-key-here" > .streamlit\secrets.toml

# macOS/Linux
mkdir -p .streamlit
echo 'OPENROUTER_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
```

#### Option B: Get Your Own API Key

1. Visit [OpenRouter](https://openrouter.ai/) and sign up
2. Generate an API key from your dashboard
3. Replace the key in `.streamlit/secrets.toml`

## 🚀 How to Run

### Start the Application

Run the following command in your terminal (make sure you're in the project directory and your virtual environment is activated):

```bash
streamlit run app.py
```

**Alternative command** (if `streamlit` command is not found):

```bash
python -m streamlit run app.py
```

### Access the Application

- The application will automatically open in your default web browser
- If it doesn't open automatically, navigate to: `http://localhost:8501`
- You should see the "🤖 AI Document Analyst" interface

## 📖 How to Use

### Step 1: Upload a PDF Document

1. Look at the **left sidebar** titled "📂 Document Center"
2. Click the **"Browse files"** button
3. Select a PDF file from your computer
4. The application will automatically process the document

### Step 2: Wait for Processing

- You'll see a spinner with the message "🧠 Processing document... (Splitting & Embedding)"
- The sidebar will show status updates
- Once complete, you'll see "✅ Indexed X chunks" in the sidebar

### Step 3: Ask Questions

1. Type your question in the **chat input box** at the bottom of the screen
2. Press **Enter** or click the send button
3. The AI will analyze the document and provide an answer based on the content
4. Continue the conversation by asking follow-up questions

### Example Questions

- "What is the main topic of this document?"
- "Summarize the key points from section 3"
- "What does the document say about [specific topic]?"
- "Are there any recommendations mentioned?"

## 📁 Project Structure

```
AI Project/
├── .streamlit/
│   └── secrets.toml          # API key configuration
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── chroma_db/               # Vector database storage (auto-created)
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend Framework** | Streamlit |
| **LLM Orchestration** | LangChain |
| **Language Model** | OpenAI GPT-4o-mini (via OpenRouter) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local) |
| **Vector Store** | ChromaDB |
| **PDF Processing** | PyPDF |
| **Programming Language** | Python 3.10+ |

## ⚙️ Configuration

### Customizing the LLM

You can modify the LLM settings in `app.py` (lines 104-109):

```python
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4o-mini",  # Change model here
    temperature=0.3                     # Adjust creativity (0.0-1.0)
)
```

### Adjusting Chunk Size

Modify document chunking in `app.py` (line 63):

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for longer chunks
    chunk_overlap=200     # Overlap between chunks
)
```

### Changing Retrieval Settings

Adjust how many relevant chunks to retrieve in `app.py` (line 112):

```python
retriever = st.session_state.vector_db.as_retriever(
    search_kwargs={"k": 5}  # Number of chunks to retrieve
)
```

## 🐛 Troubleshooting

### Issue: "streamlit: command not found"

**Solution**: Use the full Python module path:
```bash
python -m streamlit run app.py
```

### Issue: "No module named 'streamlit'" or dependency errors

**Solution**: Ensure all dependencies are installed:
```bash
python -m pip install -r requirements.txt --upgrade
```

### Issue: Application starts but no response from AI

**Solution**: Check your API key:
1. Verify the key in `.streamlit/secrets.toml` is correct
2. Ensure you have credits in your OpenRouter account
3. Check the terminal for error messages

### Issue: PDF processing fails

**Solution**: 
1. Ensure the PDF is not corrupted
2. Try a different PDF file
3. Check that the PDF is not password-protected
4. Verify you have enough disk space for the vector database

### Issue: Slow processing

**Solution**: 
- First-time run downloads the embedding model (~80MB), which may take time
- Large PDFs take longer to process
- Subsequent runs will be faster as the model is cached

### Issue: Port 8501 already in use

**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```

## 🔒 Security Notes

- **API Keys**: Never commit your API keys to version control
- The `.streamlit/secrets.toml` file should be added to `.gitignore`
- For production deployment, use environment variables or secure secret management

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the **Troubleshooting** section above
2. Review the terminal output for error messages
3. Ensure all prerequisites are installed correctly
4. Verify your Python version: `python --version`

## 🎯 Future Enhancements

- [ ] Support for multiple document formats (DOCX, TXT, etc.)
- [ ] Document comparison features
- [ ] Export chat history
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Document summarization
- [ ] Citation tracking for answers

---

**Built with ❤️ using Python, Streamlit, and LangChain**
