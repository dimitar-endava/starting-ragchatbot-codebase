import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from ai_generator import AIGenerator
from models import Course, Lesson, CourseChunk


class TestSystemHealth:
    """System health diagnostic tests to identify real-world issues"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_anthropic_api_key_availability(self):
        """Test if ANTHROPIC_API_KEY is available in environment"""
        # This will test the actual environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            pytest.fail(
                "ANTHROPIC_API_KEY not found in environment. "
                "This is likely the root cause of 'query failed' errors. "
                "Please set ANTHROPIC_API_KEY in your .env file."
            )
        
        assert api_key != "", "ANTHROPIC_API_KEY is empty"
        assert len(api_key) > 10, "ANTHROPIC_API_KEY appears to be invalid (too short)"
        
    def test_config_loading(self):
        """Test that configuration loads correctly"""
        # Test config values
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100
        assert config.MAX_RESULTS == 5
        assert config.MAX_HISTORY == 2
        assert config.CHROMA_PATH == "./chroma_db"
        
        # Test API key from config
        if not config.ANTHROPIC_API_KEY:
            pytest.fail(
                "config.ANTHROPIC_API_KEY is empty. Check your .env file setup."
            )
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_vector_store_initialization(self, mock_embedding_func, mock_chroma_client):
        """Test VectorStore can be initialized without errors"""
        # Mock ChromaDB to avoid real database operations
        mock_client = Mock()
        mock_collection = Mock()
        mock_chroma_client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test initialization
        try:
            vector_store = VectorStore(
                chroma_path=os.path.join(self.temp_dir, "test_chroma"),
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )
            assert vector_store is not None
        except Exception as e:
            pytest.fail(f"VectorStore initialization failed: {e}")
    
    @patch('anthropic.Anthropic')
    def test_ai_generator_initialization(self, mock_anthropic):
        """Test AIGenerator can be initialized with API key"""
        # Use a test API key
        test_api_key = "test-key-1234567890"
        model = "claude-sonnet-4-20250514"
        
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        try:
            ai_generator = AIGenerator(test_api_key, model)
            assert ai_generator is not None
            assert ai_generator.model == model
            mock_anthropic.assert_called_once_with(api_key=test_api_key)
        except Exception as e:
            pytest.fail(f"AIGenerator initialization failed: {e}")
    
    def test_docs_directory_exists(self):
        """Test that the docs directory exists and contains course files"""
        docs_path = "../docs"
        full_docs_path = os.path.join(os.path.dirname(__file__), docs_path)
        
        if not os.path.exists(full_docs_path):
            pytest.fail(
                f"Docs directory not found at {full_docs_path}. "
                "This means no course data can be loaded, causing search failures."
            )
        
        # Check for course files
        course_files = [f for f in os.listdir(full_docs_path) 
                       if f.lower().endswith(('.txt', '.pdf', '.docx'))]
        
        if not course_files:
            pytest.fail(
                f"No course files found in {full_docs_path}. "
                "Expected .txt, .pdf, or .docx files for course content."
            )
        
        print(f"Found {len(course_files)} course files: {course_files}")
    
    def test_chroma_db_directory_accessibility(self):
        """Test ChromaDB directory can be created and accessed"""
        chroma_path = os.path.join(self.temp_dir, "test_chroma_access")
        
        try:
            # Try to create the directory
            os.makedirs(chroma_path, exist_ok=True)
            assert os.path.exists(chroma_path)
            assert os.access(chroma_path, os.W_OK)
            
            # Try to create a test file
            test_file = os.path.join(chroma_path, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            assert os.path.exists(test_file)
        except Exception as e:
            pytest.fail(f"ChromaDB directory access failed: {e}")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('anthropic.Anthropic')
    def test_rag_system_initialization(self, mock_anthropic, mock_embedding_func, mock_chroma_client):
        """Test RAGSystem can be initialized with all components"""
        # Mock dependencies
        mock_client = Mock()
        mock_collection = Mock()
        mock_chroma_client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        # Create a test config
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "test_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': 'test-key',
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            rag_system = RAGSystem(test_config)
            assert rag_system is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.session_manager is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None
            assert rag_system.outline_tool is not None
        except Exception as e:
            pytest.fail(f"RAGSystem initialization failed: {e}")
    
    def test_sentence_transformer_availability(self):
        """Test that sentence transformers can be imported and used"""
        try:
            from sentence_transformers import SentenceTransformer
            # Don't actually load the model in test, just verify import works
            assert SentenceTransformer is not None
        except ImportError as e:
            pytest.fail(
                f"SentenceTransformer import failed: {e}. "
                "This may indicate missing dependencies or model download issues."
            )
    
    def test_course_document_processing(self):
        """Test that course documents can be processed"""
        from document_processor import DocumentProcessor
        
        # Create a test document
        test_content = """
Course Title: Test Course
Instructor: Test Instructor
Course Link: http://example.com/course

Lesson 1: Introduction
This is the introduction lesson content.
It covers basic concepts and terminology.

Lesson 2: Advanced Topics  
This lesson covers more advanced material.
It builds upon the previous lesson.
        """
        
        test_file = os.path.join(self.temp_dir, "test_course.txt")
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        try:
            processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
            course, chunks = processor.process_course_document(test_file)
            
            assert course is not None
            assert course.title == "Test Course"
            assert course.instructor == "Test Instructor"
            assert len(course.lessons) > 0
            assert len(chunks) > 0
            
            # Verify chunks have required fields
            for chunk in chunks:
                assert chunk.content != ""
                assert chunk.course_title == "Test Course"
                assert chunk.chunk_index >= 0
                
        except Exception as e:
            pytest.fail(f"Document processing failed: {e}")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') 
    @patch('anthropic.Anthropic')
    def test_real_world_query_flow(self, mock_anthropic, mock_embedding_func, mock_chroma_client):
        """Test a realistic query flow to identify where it might be failing"""
        # Mock ChromaDB
        mock_client = Mock()
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_chroma_client.return_value = mock_client
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # Mock Anthropic
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        # Mock successful tool use response
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python basics"}
        mock_tool_content.id = "tool_123"
        mock_initial_response.content = [mock_tool_content]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Here's what I found about Python basics...")]
        
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response, 
            mock_final_response
        ]
        
        # Mock vector store search
        mock_content_collection.query.return_value = {
            'documents': [['Python basics content']],
            'metadatas': [[{'course_title': 'Python Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        # Mock lesson link retrieval
        mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': '[]'  # Empty lessons for simplicity
            }]
        }
        
        # Create test config
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "test_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2', 
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': 'test-key',
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            # Initialize system
            rag_system = RAGSystem(test_config)
            
            # Perform a query
            response, sources, lesson_links = rag_system.query("What are Python basics?")
            
            # Verify response
            assert response == "Here's what I found about Python basics..."
            assert isinstance(sources, list)
            assert isinstance(lesson_links, list)
            
            # Verify the flow worked
            assert mock_anthropic_client.messages.create.call_count == 2
            mock_content_collection.query.assert_called_once()
            
        except Exception as e:
            pytest.fail(f"Real-world query flow failed: {e}")
    
    def test_env_file_existence(self):
        """Test if .env file exists with required variables"""
        env_file_path = os.path.join(os.path.dirname(__file__), "../../.env")
        
        if not os.path.exists(env_file_path):
            # Check if .env.example exists
            example_path = os.path.join(os.path.dirname(__file__), "../../.env.example")
            if os.path.exists(example_path):
                pytest.fail(
                    f".env file not found at {env_file_path}. "
                    f"Copy .env.example to .env and set ANTHROPIC_API_KEY."
                )
            else:
                pytest.fail(
                    f".env file not found at {env_file_path}. "
                    f"Create .env file with ANTHROPIC_API_KEY=your_key_here"
                )
    
    def test_chromadb_import(self):
        """Test ChromaDB can be imported successfully"""
        try:
            import chromadb
            from chromadb.config import Settings
            assert chromadb is not None
            assert Settings is not None
        except ImportError as e:
            pytest.fail(f"ChromaDB import failed: {e}")
    
    def test_anthropic_import(self):
        """Test Anthropic SDK can be imported successfully"""
        try:
            import anthropic
            assert anthropic is not None
        except ImportError as e:
            pytest.fail(f"Anthropic import failed: {e}")


class TestSystemHealthActual:
    """Tests that run against the actual system setup (no mocks)"""
    
    def test_actual_chromadb_directory(self):
        """Test the actual ChromaDB directory configured in the system"""
        chroma_path = config.CHROMA_PATH
        
        try:
            # Check if parent directory exists
            parent_dir = os.path.dirname(os.path.abspath(chroma_path))
            if not os.path.exists(parent_dir):
                pytest.fail(f"Parent directory for ChromaDB doesn't exist: {parent_dir}")
            
            # Try to create ChromaDB path if it doesn't exist
            os.makedirs(chroma_path, exist_ok=True)
            
            # Check write permissions
            if not os.access(chroma_path, os.W_OK):
                pytest.fail(f"No write permissions for ChromaDB directory: {chroma_path}")
                
        except Exception as e:
            pytest.fail(f"ChromaDB directory check failed: {e}")
    
    def test_actual_docs_directory(self):
        """Test the actual docs directory used by the system"""
        # This is relative to the backend directory
        backend_dir = os.path.dirname(__file__) + "/.."
        docs_path = os.path.join(backend_dir, "../docs")
        full_docs_path = os.path.abspath(docs_path)
        
        if not os.path.exists(full_docs_path):
            pytest.fail(
                f"Actual docs directory not found: {full_docs_path}. "
                "No course documents can be loaded on startup."
            )
        
        # List actual files
        files = os.listdir(full_docs_path)
        course_files = [f for f in files if f.lower().endswith(('.txt', '.pdf', '.docx'))]
        
        if not course_files:
            pytest.fail(
                f"No course files in actual docs directory: {full_docs_path}. "
                f"Found files: {files}"
            )
        
        print(f"Actual course files found: {course_files}")
        
        # Check if files are readable
        for file in course_files:
            file_path = os.path.join(full_docs_path, file)
            if not os.access(file_path, os.R_OK):
                pytest.fail(f"Course file not readable: {file_path}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])