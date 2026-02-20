# MCP-Ragging API

A FastAPI-based application for handling RAG (Retrieval Augmented Generation) operations via Model Context Protocol.

## Requirements

- Python 3.8+
- pip (Python package manager)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Access API Documentation

- **Interactive API docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative docs (ReDoc)**: http://localhost:8000/redoc

## Project Structure

```
MCP-Ragging/
├── main.py           # Main FastAPI application
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Development

For development with auto-reload:

```bash
uvicorn main:app --reload --port 8000
```

