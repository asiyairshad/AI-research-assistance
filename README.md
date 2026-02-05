
🧠 Multimodal Contextual AI Research Assistant (RAG)

A production-oriented Multimodal Retrieval-Augmented Generation (RAG) system that allows users to upload research PDFs and ask grounded questions over both textual content and visual figures (diagrams, images).

This project focuses on correct system design, controlled multimodal reasoning, observability, and engineering trade-offs, rather than just building a demo chatbot.

✨ What This Project Does

📄 Accepts one or multiple research PDFs from users

🔍 Parses and indexes text content for semantic retrieval

🖼️ Extracts images/figures from PDFs for vision-based reasoning

🧠 Uses a decision layer to determine when image understanding is required

👁️ Invokes vision models only when necessary (to reduce latency & cost)

🛡️ Ensures answers are grounded strictly in retrieved context

📊 Provides full observability using LangSmith

⚡ Runs as an interactive web app using Streamlit

🏗️ High-Level Architecture
PDF Upload
   │
   ├── Text Parsing (Docling / PyMuPDF)
   │       └── Embeddings → ChromaDB (Vector Store)
   │
   ├── Image Extraction (PyMuPDF)
   │
User Query
   │
   ├── Text Retrieval (Dense Vector Search)
   │
   ├── Decision Model (Is vision required?)
   │
   ├── Text LLM (Fast path)
   │        OR
   │   Vision LLM (Quality path)
   │
Final Grounded Answer

PDF Upload
   │
   ├── Text Parsing (Docling / PyMuPDF)
   │       └── Embeddings → ChromaDB (Vector Store)
   │
   ├── Image Extraction (PyMuPDF)
   │
User Query
   │
   ├── Text Retrieval (Dense Vector Search)
   │
   ├── Decision Model (Is vision required?)
   │
   ├── Text LLM (Fast path)
   │        OR
   │   Vision LLM (Quality path)
   │
Final Grounded Answer


🧩 Key Design Decisions (Why This Matters)
1. Dense Semantic Retrieval

Uses vector embeddings for meaning-based search

Handles paraphrased and conceptual queries better than keyword search

2. Vision-Aware Decision Logic

Images are not always sent to the LLM

A lightweight decision step determines if visual reasoning is required

Prevents unnecessary latency and cost

3. Stateless RAG (No Memory for Now)

Each query is handled independently

Easier evaluation, lower hallucination risk

Industry-standard starting point for RAG systems

4. Observability Over Guesswork

LangSmith traces every step:

1.retrieval
2.prompts
3.decisions
4.latency
5.token usage

🛠️ Tech Stack
Layer	Technology
UI	Streamlit
PDF Parsing	Docling, PyMuPDF
Embeddings	OpenAI
Vector Store	ChromaDB
LLM (Text)	GPT-4o-mini
LLM (Vision)	GPT-4o
Observability	LangSmith
Environment	Python, uv
📂 Project Structure
.
├── app.py              # Streamlit entry point
├── rag.py              # Core RAG + multimodal reasoning
├── ingest.py           # PDF text & image extraction
├── embeddings.py       # Embedding generation
├── vector_store.py     # ChromaDB interface
├── model.py            # Data schemas
├── .env                # API keys (not committed)
└── README.md

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone <repo-url>
cd <project-folder>

2️⃣ Create environment & install dependencies
uv venv
uv add streamlit chromadb openai pillow torch transformers python-dotenv langsmith pymupdf

3️⃣ Add environment variables (.env)
OPENAI_API_KEY=sk-xxxxxx
LANGCHAIN_API_KEY=ls-xxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=multimodal-rag


⚠️ Never commit .env to GitHub.

▶️ Running the Application
uv run streamlit run app.py


Open in browser:

http://localhost:8501

🧪 How the System Behaves
Text-only question

“Explain multi-head attention”

➡️ Uses text retrieval + text LLM
➡️ No images involved

Vision-required question

“Explain the architecture shown in the figure”

➡️ System decides vision is required
➡️ Sends relevant image + context to vision LLM
➡️ Returns grounded explanation

📊 Observability & Debugging (LangSmith)

LangSmith enables:

Full prompt inspection

Retrieval trace analysis

Latency breakdown

Token usage monitoring

Comparison between fast vs quality mode

This allows the system to be measured, not guessed.

⏱️ Performance Design

The system supports two inference modes:

Mode	Goal
Fast	Lower latency, lower cost
Quality	Better reasoning, multimodal support

This exposes latency vs quality trade-offs explicitly, which is how real AI systems are engineered.

🧠 What This Project Does NOT Do (Yet)

❌ Conversational memory

❌ Long-term document persistence

❌ Hybrid retrieval (BM25 + dense)

❌ Automated Recall@K / Precision@K evaluation

These are intentional exclusions to keep the core system clean and verifiable.

🚀 Future Improvements (Planned Evolution)

🔹 Add retrieval evaluation (Recall@K, Precision@K)

🔹 Introduce hybrid retrieval (dense + sparse)

🔹 Image re-ranking for better figure selection

🔹 Optional conversational memory (session-based)

🔹 Persistent vector storage for multi-user deployments

🔹 Deployment on Google Cloud Run

👩‍💻 Author

Asiya Irshad
B.Tech Computer Science
Interested in Generative AI, Multimodal Systems, and Production RAG Architectures

📝 Final Note

This project is not a tutorial clone.

It demonstrates:

real RAG architecture

controlled multimodal reasoning

evaluation-ready design

professional observability practices

It is built with the mindset of:

“How would this work in a real AI team?”

