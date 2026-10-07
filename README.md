# 🔬 AI Research Paper Assistant

> An AI-powered Retrieval-Augmented Generation (RAG) application for asking questions, summarizing, and comparing research papers using semantic search and Large Language Models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FAISS](https://img.shields.io/badge/Vector%20Database-FAISS-orange)
![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-green)
![Groq](https://img.shields.io/badge/LLM-Groq-purple)

---

## 📌 Overview

**AI Research Paper Assistant** is a Retrieval-Augmented Generation (RAG) based application designed to make research paper analysis faster and easier.

Instead of manually reading large research papers to find specific information, users can upload PDF research papers and interact with them using natural language.

The system:

- 📄 Processes uploaded research papers
- ✂️ Splits documents into smaller chunks
- 🧠 Converts text into vector embeddings
- 🔎 Performs semantic similarity search
- 📚 Retrieves the most relevant sections
- 🤖 Uses an LLM to generate answers based on retrieved context
- 📊 Provides source and retrieval information
- 📝 Supports research paper summarization
- ⚖️ Supports comparison of research papers
- 📈 Provides evaluation functionality

---

# 🎯 Problem Statement

Research papers often contain hundreds of pages of technical information, making it difficult and time-consuming to locate specific information.

Traditional keyword-based search may fail when the user's question uses different terminology from the document.

For example:

> "What method was used to improve model performance?"

The exact phrase may not exist in the paper, even though the relevant information is present.

The goal of this project is to build an intelligent assistant that understands the **semantic meaning** of the question and retrieves the most relevant information from research papers.

---

# 💡 Proposed Solution

The project uses **Retrieval-Augmented Generation (RAG)**.

Instead of asking an LLM to answer a question using only its pretrained knowledge, the system first retrieves relevant information from the uploaded research papers.

The retrieved context is then provided to the LLM to generate a grounded response.

### RAG Pipeline

```text
                    ┌─────────────────────┐
                    │   Research Paper    │
                    │        PDF          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PDF Processing    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Text Chunking      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Embeddings     │
                    │ all-MiniLM-L6-v2    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FAISS Vector DB   │
                    └──────────┬──────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Relevant Context    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       LLM           │
                    │       Groq          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Answer         │
                    └─────────────────────┘
🚀 Key Features
1. 📄 Research Paper Upload
Users can upload one or multiple PDF research papers through the Streamlit interface.
The uploaded papers are processed and added to the knowledge base.
2. 🧹 PDF Processing
The application extracts text from PDF documents using PDF processing utilities.
The extracted text is divided into smaller chunks so that relevant sections can be retrieved efficiently.
3. 🧠 Semantic Embeddings
The project uses:
all-MiniLM-L6-v2

from Sentence Transformers to convert text into numerical vector representations.
This allows the system to compare the semantic meaning of the user's question with document chunks.
4. 🔎 Semantic Search
FAISS is used as the vector database.
When the user asks a question:
1. The question is converted into an embedding.
2. FAISS searches the vector database.
3. The most relevant chunks are retrieved.
4. Retrieved chunks are provided to the LLM.
5. 🤖 RAG-Based Question Answering
The application uses a Large Language Model through Groq.
The LLM receives:
User Question
       +
Retrieved Context
       ↓
     LLM
       ↓
Grounded Answer

This helps the system generate answers based on the uploaded research papers rather than relying only on general model knowledge.
6. 📚 Source Retrieval
The application displays information about the retrieved sources, including:
- Source document
- Page number
- Similarity/retrieval score
- Retrieved text
This makes the generated response more transparent.
7. 📝 Research Paper Summarization
Users can summarize research papers and quickly understand their major concepts without reading the complete document.
8. ⚖️ Research Paper Comparison
The application provides functionality for comparing research papers based on their content.
This can help users understand differences between papers, approaches, methods, and findings.
9. 📊 Evaluation
The project includes an evaluation section for testing the performance of the RAG system using predefined questions.
🛠️ Tech Stack
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
Sentence Transformers	Text embeddings
all-MiniLM-L6-v2	Embedding model
FAISS	Vector similarity search
Groq	Large Language Model inference
PyMuPDF	PDF text extraction
LangChain	Prompting/text processing utilities
NumPy	Numerical operations
Pandas	Data processing
python-dotenv	Environment variable management


📂 Project Structure
AI-Research-Paper-Assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   └── uploads/
│
├── evaluation/
│   └── questions.json
│
├── vector_db/
│   └── faiss/
│       ├── faiss.index
│       └── metadata.json
│
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluator.py
│   ├── llm.py
│   ├── pdf_processor.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vectorstore.py
│
└── venv/

🔄 Application Workflow
Step 1 — Upload PDF
The user uploads one or more research papers.
Step 2 — Extract Text
Text is extracted from the PDF documents.
Step 3 — Chunk the Text
Large documents are divided into smaller text chunks.
Step 4 — Generate Embeddings
Each chunk is converted into a vector representation using:
all-MiniLM-L6-v2

Step 5 — Store Vectors
The generated vectors are stored in FAISS.
Step 6 — Ask a Question
The user enters a natural-language question.
Step 7 — Retrieve Relevant Chunks
The question is converted into an embedding and compared with stored document vectors.
Step 8 — Generate Answer
The retrieved context is passed to the LLM.
Step 9 — Display Answer and Sources
The application displays the generated answer along with relevant source information.
🧩 Core Components
app.py
The main Streamlit application.
Responsible for:
- UI
- PDF upload
- Knowledge base management
- User questions
- Summarization
- Comparison
- Evaluation
- Displaying retrieved sources
pdf_processor.py
Responsible for processing research paper PDFs and extracting text.
embeddings.py
Handles creation of vector embeddings using the Sentence Transformer model.
vectorstore.py
Handles the FAISS vector database.
Responsibilities include:
- Creating vector indexes
- Adding embeddings
- Saving indexes
- Loading indexes
- Searching vectors
retriever.py
Responsible for retrieving the most relevant document chunks for a given query.
The system uses similarity search to identify relevant information.
llm.py
Handles communication with the Large Language Model through Groq.
rag_pipeline.py
Connects the major components of the system:
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
Context
 ↓
LLM
 ↓
Answer

evaluator.py
Handles evaluation of the RAG system using predefined questions and evaluation data.
⚙️ Installation
1. Clone the Repository
git clone https://github.com/shashwatpratapsingh7717/RAG-BASED-RESEARCH-PAPER-ASSISTANT.git

Move into the project directory:
cd RAG-BASED-RESEARCH-PAPER-ASSISTANT

2. Create Virtual Environment
python -m venv venv

Windows
venv\Scripts\activate

Linux/macOS
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

🔐 Environment Variables
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key

⚠️ Never upload your .env file or API key to GitHub.
The .env file should be included in .gitignore.
▶️ Run the Application
After activating the virtual environment:
streamlit run app.py

The application will open in your browser.
Usually:
http://localhost:8501

📖 How to Use
1. Start the application
streamlit run app.py

2. Upload Research Papers
Use the Knowledge Base section to upload PDF files.
3. Build Knowledge Base
Click:
Build / Update Knowledge Base

The application will process the documents and create the vector index.
4. Ask Questions
Go to the Ask section and enter your question.
Example:
What methodology does this research paper use?

5. View Sources
The application displays the retrieved document sections and source information.
6. Summarize
Use the Summarize section to generate a concise summary.
7. Compare
Use the Compare section to analyze multiple research papers.
8. Evaluate
Use the Evaluation section to test the RAG pipeline.
📊 Example Questions
You can ask questions such as:
What is the main objective of this research paper?

What dataset was used in the study?

What methodology was proposed?

What are the main findings?

What limitations are mentioned by the authors?

What future work has been suggested?

🧠 Why RAG?
A normal LLM may not have access to the content of a newly uploaded research paper.
RAG solves this problem by retrieving relevant information from the user's documents before generating the response.
Traditional LLM
Question
   ↓
LLM
   ↓
Answer

RAG
Question
   ↓
Vector Search
   ↓
Relevant Document Chunks
   ↓
LLM
   ↓
Grounded Answer

This makes the system more suitable for document-based question answering.
🔍 Retrieval Process
The retrieval process works approximately as follows:
User Query
     ↓
Query Embedding
     ↓
FAISS Similarity Search
     ↓
Top-K Relevant Chunks
     ↓
Context Construction
     ↓
Prompt
     ↓
LLM
     ↓
Final Answer

The Top-K value controls how many relevant chunks are retrieved.
📈 Evaluation
The project contains an evaluation dataset:
evaluation/questions.json

The evaluation module can be used to test how effectively the RAG pipeline retrieves and answers predefined questions.
🔒 Security
Sensitive credentials should never be committed to GitHub.
The following files/directories should remain local:
.env
venv/
vector_db/
data/uploads/
__pycache__/

Use .gitignore to prevent accidental commits.
☁️ Deployment
The application can be deployed using Streamlit Community Cloud.
General deployment process:
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Connect Repository
       ↓
Select app.py
       ↓
Configure Secrets
       ↓
Deploy

Add the API key through Streamlit Cloud Secrets:
GROQ_API_KEY = "your_groq_api_key"

Do not hard-code the API key inside the source code.
🚧 Future Improvements
Possible future improvements include:
- 🔐 User authentication
- 📚 Support for additional document formats
- 🧠 Better embedding models
- 🔎 Hybrid keyword + semantic search
- 💬 Conversation memory
- 📑 Automatic citation generation
- 📊 Advanced RAG evaluation
- ⚡ Streaming LLM responses
- 🌐 Multi-user support
- 📈 Research paper analytics
- 🗂️ Document collections and folders
- 🎯 Reranking retrieved documents
🎓 Learning Outcomes
This project demonstrates practical knowledge of:
- Retrieval-Augmented Generation
- Large Language Models
- Natural Language Processing
- Semantic Search
- Vector Databases
- Text Embeddings
- PDF Processing
- Prompt Engineering
- Streamlit Application Development
- AI Application Deployment
- Git & GitHub
👨‍💻 Author
Shashwat Pratap Singh
B.Tech — Artificial Intelligence & Machine Learning
Profiles
- GitHub: https://github.com/shashwatpratapsingh7717
- LinkedIn: Add your LinkedIn profile
- LeetCode: Add your LeetCode profile
⭐ Project
If you find this project useful, consider giving the repository a ⭐ on GitHub.
📜 License
This project is developed for educational and academic purposes.
