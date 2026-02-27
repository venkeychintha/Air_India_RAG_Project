# ✈️ Air India Chat Assistant

A RAG-based (Retrieval-Augmented Generation) conversational assistant for answering questions about Air India, powered by AWS Bedrock, Amazon Titan Embeddings, and ChromaDB — served via a Streamlit web interface.

---

## 🧠 How It Works

1. Air India PDF documents are loaded and split into chunks.
2. Each chunk is embedded using **Amazon Titan Embed Text v2** via AWS Bedrock.
3. Embeddings are stored in a local **ChromaDB** vector store.
4. When a user asks a question, the top-3 most relevant chunks are retrieved.
5. The question + context are sent to **Amazon Nova Pro** (via Bedrock) to generate an answer.
6. The answer is displayed in the **Streamlit** chat UI.

---

## 🗂️ Project Structure

```
├── app.py          # Streamlit frontend
├── main.py         # Core RAG logic (embeddings, vector store, LLM call)
├── test.py         # Standalone test script for AWS Bedrock streaming
├── AirIndia/       # Directory containing source PDF documents
└── chroma_vectorestore/  # Persisted ChromaDB vector store (auto-generated)
```

---

## ⚙️ Prerequisites

- Python 3.9+
- An AWS account with access to **Amazon Bedrock**
- AWS credentials configured (via `~/.aws/credentials` or environment variables)
- Bedrock model access enabled for:
  - `amazon.titan-embed-text-v2:0`
  - `us.amazon.nova-pro-v1:0`

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/air-india-assistant.git
cd air-india-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
```
streamlit
langchain
langchain-community
langchain-chroma
boto3
tiktoken
pypdf
chromadb
```

### 3. Add your PDF documents

Place your Air India PDF files inside an `AirIndia/` directory in the project root:

```
AirIndia/
├── annual_report.pdf
├── policies.pdf
└── ...
```

### 4. Build the vector store (first-time only)

In `main.py`, **uncomment** the following blocks to load documents and populate ChromaDB:

```python
# Load and split documents
loader = PyPDFDirectoryLoader("AirIndia", glob="**/*.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Add to vector store
uuids = [str(uuid4()) for _ in range(len(texts))]
vector_store.add_documents(documents=texts, ids=uuids)
```

Run `main.py` once to populate the store:

```bash
python main.py
```

After this, **re-comment** those blocks to avoid re-indexing on every run.

---

## 🖥️ Running the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 🧪 Testing Bedrock Connectivity

Use `test.py` to verify your AWS Bedrock setup independently:

```bash
python test.py
```

This sends a sample creative writing prompt to Nova Pro using the streaming API and prints the response with time-to-first-token metrics.

---

## 🔐 AWS Configuration

Make sure your AWS credentials are set up with the appropriate permissions:

```bash
aws configure
```

Or set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-2
```

The required IAM permissions include:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`

---

## 📌 Notes

- The vector store is persisted locally in `./chroma_vectorestore/` and only needs to be built once.
- Token limits are handled automatically — inputs exceeding 8,000 tokens are truncated before embedding.
- The LLM is configured with deterministic settings (`temperature: 0`) for consistent answers.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.# Air_India_RAG_Project
