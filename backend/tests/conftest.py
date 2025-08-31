import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary paths"""
    return type('Config', (), {
        'CHUNK_SIZE': 800,
        'CHUNK_OVERLAP': 100,
        'CHROMA_PATH': os.path.join(temp_dir, "test_chroma"),
        'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
        'MAX_RESULTS': 5,
        'ANTHROPIC_API_KEY': 'test-api-key-1234567890',
        'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
        'MAX_HISTORY': 2
    })()


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB components"""
    with patch('chromadb.PersistentClient') as mock_client_class:
        with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embedding_func:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            yield {
                'client': mock_client,
                'collection': mock_collection,
                'client_class': mock_client_class,
                'embedding_func': mock_embedding_func
            }


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic API client"""
    with patch('anthropic.Anthropic') as mock_anthropic_class:
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Default successful response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        yield {
            'client': mock_client,
            'client_class': mock_anthropic_class,
            'response': mock_response
        }


@pytest.fixture
def mock_vector_search_results():
    """Mock vector store search results"""
    return {
        'documents': [['Test document content 1'], ['Test document content 2']],
        'metadatas': [[
            {'course_title': 'Test Course', 'lesson_number': 1, 'lesson_title': 'Test Lesson 1'}
        ], [
            {'course_title': 'Test Course', 'lesson_number': 2, 'lesson_title': 'Test Lesson 2'}
        ]],
        'distances': [[0.1], [0.2]]
    }


@pytest.fixture
def sample_course_content():
    """Sample course content for testing document processing"""
    return """
Course Title: Test Python Course
Instructor: Test Instructor
Course Link: https://example.com/python-course

Lesson 1: Python Basics
This lesson covers Python fundamentals including variables, data types, and basic operations.
Variables in Python are dynamically typed and don't require declaration.

Lesson 2: Control Structures
This lesson covers if statements, loops, and conditional logic.
Python uses indentation to define code blocks instead of curly braces.

Lesson 3: Functions and Classes
This lesson introduces function definitions and object-oriented programming.
Functions are defined using the 'def' keyword and classes use the 'class' keyword.
    """


@pytest.fixture
def mock_rag_system(test_config, mock_chromadb, mock_anthropic):
    """Create a mock RAG system for testing"""
    with patch.object(RAGSystem, '__init__', return_value=None):
        rag_system = RAGSystem.__new__(RAGSystem)
        rag_system.vector_store = Mock()
        rag_system.ai_generator = Mock()
        rag_system.session_manager = Mock()
        rag_system.search_tool = Mock()
        rag_system.outline_tool = Mock()
        rag_system.tool_manager = Mock()
        
        # Mock successful query response
        rag_system.query = Mock(return_value=(
            "Test response",
            ["Test source 1", "Test source 2"], 
            ["https://example.com/lesson1", None]
        ))
        
        # Mock course analytics
        rag_system.get_course_analytics = Mock(return_value={
            "total_courses": 2,
            "course_titles": ["Test Course 1", "Test Course 2"]
        })
        
        yield rag_system


@pytest.fixture
def test_api_app():
    """Create a test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create app without static file mounting to avoid frontend dependency
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models (copied from main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        source_links: List[Optional[str]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    class ClearSessionRequest(BaseModel):
        session_id: str
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = (
        "Test response from API", 
        ["Test source"], 
        ["https://example.com"]
    )
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    mock_rag_system.session_manager.create_session.return_value = "test-session-123"
    mock_rag_system.session_manager.clear_session.return_value = None
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources, lesson_links = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                source_links=lesson_links,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/sessions/clear")
    async def clear_session(request: ClearSessionRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return {"message": "Session cleared successfully", "session_id": request.session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API - Test"}
    
    return app


@pytest.fixture
def api_client(test_api_app):
    """Create a test client for the API"""
    from fastapi.testclient import TestClient
    return TestClient(test_api_app)


@pytest.fixture(scope="session")
def real_docs_path():
    """Path to real docs directory if it exists"""
    docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
    if os.path.exists(docs_path):
        return docs_path
    return None


@pytest.fixture
def sample_test_file(temp_dir, sample_course_content):
    """Create a sample test file with course content"""
    test_file = os.path.join(temp_dir, "test_course.txt")
    with open(test_file, 'w') as f:
        f.write(sample_course_content)
    return test_file