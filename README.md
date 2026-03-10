📘 Multimodal Research Assistant

A Hybrid Multimodal RAG System with Embedding & Answer Caching

🚀 Overview

This project is a Multimodal Research Assistant that allows users to:

1.Upload research PDFs

2. Ask natural language questions

3. Retrieve grounded answers using semantic search

4. Automatically retrieve and explain relevant figures

5. Switch between fast and high-quality reasoning modes

6. Benefit from embedding and answer caching for low latency

This system combines text retrieval + image retrieval + multimodal LLM reasoning into a single hybrid pipeline.

🧠 What Makes This Different?

This is NOT a basic PDF chatbot.

It implements:

✅ Hybrid Retrieval (Text + Image Embeddings)
✅ OpenAI embeddings for semantic text search
✅ CLIP embeddings for figure retrieval
✅ In-memory ChromaDB for vector similarity
✅ SQLite-based embedding caching
✅ SQLite-based full answer caching
✅ Vision-enabled GPT model for multimodal reasoning
✅ Strict context grounding to reduce hallucinations

🏗️ System Architecture
1️⃣ Document Ingestion
Extract text chunks from PDF
Extract images from PDF
Track page metadata

2️⃣ Dual Embedding Pipeline
Content Type	Embedding Model
Text	OpenAI Embeddings
Images	CLIP (ViT-B/32)
Text and image embeddings are stored in separate Chroma collections to prevent dimension conflicts.

3️⃣ Vector Storage
ChromaDB (in-memory)
text_vectors
image_vectors
Used only for similarity search
No disk persistence

4️⃣ Caching Layer (SQLite)
To reduce latency:
Text query embeddings cached
Image query embeddings cached
Full LLM answers cached
If a query is repeated:

✔ No embedding recomputation
✔ No vector search
✔ No LLM call
✔ Near-instant response

5️⃣ Query Flow
User submits query
Check answer cache
If exists → return immediately
Retrieve text embeddings (cached if available)
Perform semantic retrieval
If query contains visual keywords:
Retrieve image embeddings (cached)
Retrieve matching figure
Call appropriate model:
gpt-4o-mini (fast mode)
gpt-4o (quality mode)
Save answer to cache

🔄 Retrieval Strategy
Text retrieval is always performed.
Image retrieval is triggered only if query contains keywords like:
"figure"
"fig"
"diagram"
"visual"
"show"
This avoids unnecessary multimodal calls and reduces latency.

🛠️ Tech Stack
Python
Streamlit
OpenAI API
CLIP (Vision Transformer)
ChromaDB (in-memory)
SQLite (caching layer)
LangSmith (tracing)

Project structure
AI-research-assistance/
│
├── app.py              # Streamlit UI
├── rag.py              # Retrieval + LLM pipeline
├── embeddings.py       # Text & image embeddings
├── vector_store.py     # In-memory Chroma storage
├── cache_store.py      # SQLite caching layer
├── ingest.py           # PDF parsing
├── cache.db            # SQLite database
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/multimodal-research-assistant.git
cd multimodal-research-assistant

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Environment Variables
Create a .env file:

OPENAI_API_KEY=your_api_key_here
LANGCHAIN_PROJECT="your langchain api key for tracing"

▶️ Run the Application
streamlit run app.py


Open:
http://localhost:8501

🧪 Example Queries
“Summarize the paper”
“Explain masked attention”
“Explain Figure 1.1”
“Show the architecture diagram”

🎯 Design Decisions
Why In-Memory Chroma?
Vector search is fast in memory. No need for persistent vector storage for single-session usage.
Why SQLite Caching?
To avoid recomputation:
Reduces latency
Reduces API cost
Prevents redundant embedding calls
Prevents redundant LLM calls
Why Separate Text and Image Collections?
OpenAI and CLIP embeddings have different dimensions.
Keeping them separate prevents vector dimension conflicts.

🚧 Future Improvements
Add TTL-based cache expiration
Add RAG evaluation metrics
Improve image ranking strategy
Add multi-document cross-paper retrieval
Add semantic chunk ranking improvements

🧠 Skills Demonstrated
Multimodal RAG design
Hybrid embedding architecture
Vision-grounded LLM reasoning
Vector database design
Latency optimization via caching
Hallucination mitigation
Production-style pipeline structuring

👩‍💻 Author

Asiya Irshad
B.Tech CSE | AI & Generative AI Enthusiast

Interested in:
Multimodal AI
Retrieval-Augmented Generation
AI System Design
Generative AI Engineering

⭐ If You Found This Useful

Feel free to connect, fork, or contribute!
