# Employee Handbook AI Assistant

A Flask-based web application that allows users to upload employee handbook documents and query them using AI-powered natural language processing.

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **AI-Powered Q&A**: Ask questions about uploaded documents using OpenAI's language models
- **Vector Search**: Uses ChromaDB for efficient document similarity search
- **Session Management**: Maintains separate sessions for different document uploads
- **Web Interface**: Clean, responsive web interface for easy interaction

## Technologies Used

- **Flask**: Web framework
- **LangChain**: LLM orchestration and document processing
- **OpenAI**: Language model and embeddings
- **ChromaDB**: Vector database for document storage
- **Python**: Backend programming language

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Assignment
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install flask python-dotenv langchain langchain-community langchain-openai langchain-chroma langchain-text-splitters pypdf docx2txt chromadb openai
```

5. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload an employee handbook document (PDF, DOCX, or TXT)

4. Once processed, ask questions about the document content

## File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface template
├── uploads/              # Uploaded files storage
├── chroma_sessions/      # ChromaDB vector store sessions
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## Configuration

- **MAX_CONTENT_LENGTH**: 16MB (configurable in app.py)
- **ALLOWED_EXTENSIONS**: txt, pdf, docx
- **CHUNK_SIZE**: 900 characters with 150 character overlap

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.