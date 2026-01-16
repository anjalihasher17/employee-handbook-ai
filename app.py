from flask import Flask, request, jsonify, session,render_template
import os
import uuid
from werkzeug.utils import secure_filename
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Embedddings and Vector Store
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production-987654321')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

NOT_FOUND_MSG = "Information not available in the Employee Handbook."

# Global storage for document processors (in production, use Redis or database)
document_stores = {}

class LLMCallCounter(BaseCallbackHandler):
    """Simple callback to show when the LLM is actually invoked (vs cache hit)."""
    def __init__(self) -> None:
        self.calls = 0

    def on_llm_start(self, *args, **kwargs) -> None:
        self.calls += 1
        print(f"[LLM] call #{self.calls}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_handbook(path: str) -> List[Document]:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Use .pdf, .docx, or .txt")

    docs = loader.load()
    # Normalize metadata for traceability
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", os.path.basename(path))
    return docs


def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_or_load_vectorstore(chunks: List[Document], persist_dir: str, collection_name: str) -> Chroma:
    embeddings = OpenAIEmbeddings()
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # If empty collection, add documents
    try:
        existing = vs._collection.count()
    except Exception:
        existing = 0

    if existing == 0:
        print("[Index] Building vector index...")
        vs.add_documents(chunks)
        print("[Index] Done.")
    else:
        print(f"[Index] Using existing index ({existing} chunks).")

    return vs


def format_hits(docs: List[Document], max_chars: int = 1200) -> str:
    """Return a compact 'evidence pack' the agent can cite."""
    if not docs:
        return ""

    lines = []
    for i, d in enumerate(docs, 1):
        text = (d.page_content or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        src = d.metadata.get("source", "handbook")
        page = d.metadata.get("page", d.metadata.get("page_number"))
        loc = f"{src}" + (f", page {page}" if page is not None else "")
        lines.append(f"[{i}] ({loc}) {text}")

    return "\n".join(lines)


def process_document(filepath: str, session_id: str):
    """Process uploaded document and create chain for the session"""
    try:
        # Enable caching
        set_llm_cache(InMemoryCache())
        
        # Load and process document
        docs = load_handbook(filepath)
        chunks = chunk_docs(docs)
        
        persist_dir = f"./chroma_sessions/{session_id}"
        vs = build_or_load_vectorstore(chunks, persist_dir, "employee_handbook")
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        
        counter = LLMCallCounter()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            callbacks=[counter],
        )
        
        # Create a conversational retrieval QA chain
        prompt_template = """You are an internal policy assistant. Answer ONLY using the provided context from the Employee Handbook.

Rules:
- If no relevant information is found in the context, reply exactly: 'Information not available in the Employee Handbook.'
- If the question is not about employee policies/handbook scope, reply exactly: 'Information not available in the Employee Handbook.'
- Do not use outside knowledge. Do not guess.
- Keep answers clear and concise.
- When answering, cite the source information.
- Use the conversation history to understand follow-up questions and provide contextual responses.

Conversation History:
{chat_history}

Context: {context}

Human: {question}

Assistant:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Function to format retrieved documents
        def format_docs(docs):
            return "\n\n".join([f"Source: {doc.metadata.get('source', 'handbook')}\nContent: {doc.page_content}" for doc in docs])
        
        # Function to format chat history
        def format_chat_history(history):
            if not history:
                return "No previous conversation."
            formatted = []
            for entry in history[-6:]:  # Keep last 6 exchanges (3 Q&A pairs)
                formatted.append(f"Human: {entry['question']}")
                formatted.append(f"Assistant: {entry['answer']}")
            return "\n".join(formatted)
        
        # Create the chain using LCEL
        def create_conversational_chain(chat_history):
            return (
                {
                    "context": retriever | format_docs, 
                    "question": RunnablePassthrough(),
                    "chat_history": lambda _: format_chat_history(chat_history)
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        
        # Store in session data
        document_stores[session_id] = {
            'create_chain': create_conversational_chain,
            'filename': os.path.basename(filepath),
            'retriever': retriever,
            'chat_history': []  # Initialize empty conversation history
        }
        
        return True, "Document processed successfully!"
        
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # Process the document
        success, message = process_document(filepath, session_id)
        
        if success:
            return jsonify({
                'message': message,
                'filename': filename,
                'session_id': session_id
            })
        else:
            return jsonify({'error': message}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload .txt, .pdf, or .docx files.'}), 400


@app.route('/chat', methods=['POST'])
def chat():
    if 'session_id' not in session:
        return jsonify({'error': 'No session found. Please upload a document first.'}), 400
    
    session_id = session['session_id']
    if session_id not in document_stores:
        return jsonify({'error': 'No document processed. Please upload a document first.'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Please provide a question.'}), 400
    
    try:
        # Get session data
        session_data = document_stores[session_id]
        chat_history = session_data['chat_history']
        create_chain = session_data['create_chain']
        
        # Create chain with current chat history
        chain = create_chain(chat_history)
        
        # Get answer from the chain
        result = chain.invoke(question)
        
        # Extract content from the result
        if hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)
        
        # Add this Q&A to chat history
        chat_history.append({
            'question': question,
            'answer': answer,
            'timestamp': str(uuid.uuid4())[:8]  # Simple timestamp
        })
        
        # Keep only last 10 exchanges to prevent memory overflow
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Update stored chat history
        document_stores[session_id]['chat_history'] = chat_history
        
        return jsonify({
            'question': question,
            'answer': answer,
            'filename': document_stores[session_id]['filename'],
            'conversation_length': len(chat_history)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500


@app.route('/status')
def status():
    if 'session_id' not in session:
        return jsonify({'loaded': False})
    
    session_id = session['session_id']
    if session_id in document_stores:
        session_data = document_stores[session_id]
        return jsonify({
            'loaded': True,
            'filename': session_data['filename'],
            'conversation_length': len(session_data.get('chat_history', [])),
            'has_conversation': len(session_data.get('chat_history', [])) > 0
        })
    
    return jsonify({'loaded': False})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    if session_id in document_stores:
        document_stores[session_id]['chat_history'] = []
        return jsonify({'message': 'Conversation history cleared successfully'})
    
    return jsonify({'error': 'No document session found'}), 400


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('chroma_sessions', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)