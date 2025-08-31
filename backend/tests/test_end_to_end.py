import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch
from pathlib import Path

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem
import asyncio
from app import app, rag_system as global_rag_system
from fastapi.testclient import TestClient


class TestEndToEndDiagnostic:
    """End-to-end tests to diagnose the actual system failures"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_real_rag_system_with_empty_database(self):
        """Test RAG system behavior when database is empty"""
        # Create a test RAG system with clean database
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "empty_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': config.ANTHROPIC_API_KEY,  # Use real API key
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            rag_system = RAGSystem(test_config)
            
            # Try to query without any data loaded
            response, sources, lesson_links = rag_system.query("What is Python?")
            
            print(f"Response with empty database: {response}")
            print(f"Sources: {sources}")
            print(f"Lesson links: {lesson_links}")
            
            # This should not fail completely - it should either:
            # 1. Return a general knowledge answer without tools
            # 2. Return a "no content found" message if tools are used
            assert response != "query failed"
            assert isinstance(response, str)
            assert len(response) > 0
            
        except Exception as e:
            pytest.fail(f"RAG system failed with empty database: {e}")
    
    def test_real_rag_system_initialization_with_real_docs(self):
        """Test RAG system with real document loading"""
        docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
        
        if not os.path.exists(docs_path):
            pytest.skip("Real docs directory not found")
        
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "real_docs_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': config.ANTHROPIC_API_KEY,  # Use real API key
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            rag_system = RAGSystem(test_config)
            
            # Load real documents
            courses_added, chunks_added = rag_system.add_course_folder(docs_path, clear_existing=True)
            
            print(f"Loaded {courses_added} courses with {chunks_added} chunks")
            
            if courses_added == 0:
                pytest.fail("No courses were loaded from real docs directory")
            
            # Test analytics
            analytics = rag_system.get_course_analytics()
            print(f"Analytics: {analytics}")
            
            assert analytics["total_courses"] > 0
            assert len(analytics["course_titles"]) > 0
            
        except Exception as e:
            pytest.fail(f"Real document loading failed: {e}")
    
    def test_real_api_query_with_loaded_data(self):
        """Test a real API query against loaded course data"""
        docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
        
        if not os.path.exists(docs_path):
            pytest.skip("Real docs directory not found")
        
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "api_test_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': config.ANTHROPIC_API_KEY,  # Use real API key
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            rag_system = RAGSystem(test_config)
            
            # Load documents
            courses_added, chunks_added = rag_system.add_course_folder(docs_path)
            print(f"Loaded {courses_added} courses with {chunks_added} chunks")
            
            if courses_added == 0:
                pytest.skip("No courses loaded - cannot test query")
            
            # Try a content-related query that should trigger search
            response, sources, lesson_links = rag_system.query("What topics are covered in the courses?")
            
            print(f"Query response: {response}")
            print(f"Sources: {sources}")
            print(f"Lesson links: {lesson_links}")
            
            # Validate response
            assert response != "query failed"
            assert isinstance(response, str)
            assert len(response) > 10  # Should be a meaningful response
            
            # If search was used, should have sources
            if sources:
                assert isinstance(sources, list)
                assert all(isinstance(source, str) for source in sources)
            
        except Exception as e:
            pytest.fail(f"Real API query failed: {e}")
    
    def test_vector_store_search_with_real_data(self):
        """Test vector store search functionality with real data"""
        docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
        
        if not os.path.exists(docs_path):
            pytest.skip("Real docs directory not found")
        
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "vector_search_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': config.ANTHROPIC_API_KEY,
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            rag_system = RAGSystem(test_config)
            
            # Load documents  
            courses_added, chunks_added = rag_system.add_course_folder(docs_path)
            
            if courses_added == 0:
                pytest.skip("No courses loaded - cannot test vector search")
            
            # Test direct vector store search
            search_results = rag_system.vector_store.search("course introduction")
            
            print(f"Vector search results error: {search_results.error}")
            print(f"Vector search results count: {len(search_results.documents)}")
            print(f"Vector search documents: {search_results.documents[:2]}")  # First 2 docs
            
            if search_results.error:
                pytest.fail(f"Vector store search failed: {search_results.error}")
            
            # Should find some results
            if search_results.is_empty():
                pytest.fail("Vector store search returned empty results for generic query")
            
            assert len(search_results.documents) > 0
            assert len(search_results.metadata) == len(search_results.documents)
            
        except Exception as e:
            pytest.fail(f"Vector store search with real data failed: {e}")
    
    def test_course_search_tool_with_real_data(self):
        """Test CourseSearchTool with real vector store data"""
        docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
        
        if not os.path.exists(docs_path):
            pytest.skip("Real docs directory not found")
        
        test_config = type('Config', (), {
            'CHUNK_SIZE': 800,
            'CHUNK_OVERLAP': 100,
            'CHROMA_PATH': os.path.join(self.temp_dir, "tool_test_chroma"),
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'MAX_RESULTS': 5,
            'ANTHROPIC_API_KEY': config.ANTHROPIC_API_KEY,
            'ANTHROPIC_MODEL': 'claude-sonnet-4-20250514',
            'MAX_HISTORY': 2
        })()
        
        try:
            rag_system = RAGSystem(test_config)
            
            # Load documents
            courses_added, chunks_added = rag_system.add_course_folder(docs_path)
            
            if courses_added == 0:
                pytest.skip("No courses loaded - cannot test search tool")
            
            # Test search tool directly
            search_result = rag_system.search_tool.execute("introduction")
            
            print(f"Search tool result: {search_result}")
            print(f"Last sources: {rag_system.search_tool.last_sources}")
            
            # Should return formatted results, not an error
            assert "ChromaDB connection failed" not in search_result
            assert "Search error:" not in search_result
            
            # Should either find content or say no content found
            if "No relevant content found" not in search_result:
                # Found content - should be formatted properly
                assert len(search_result) > 0
                # Should have sources tracked
                assert len(rag_system.search_tool.last_sources) > 0
            
        except Exception as e:
            pytest.fail(f"CourseSearchTool with real data failed: {e}")


class TestAPIEndpoint:
    """Test the actual FastAPI endpoint"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
    
    def test_api_endpoint_with_general_question(self):
        """Test API endpoint with a general knowledge question"""
        query_data = {
            "query": "What is 2+2?",
            "session_id": None
        }
        
        try:
            response = self.client.post("/api/query", json=query_data)
            
            print(f"API Response status: {response.status_code}")
            print(f"API Response body: {response.json()}")
            
            if response.status_code == 500:
                error_detail = response.json().get("detail", "Unknown error")
                pytest.fail(f"API endpoint failed with 500 error: {error_detail}")
            
            assert response.status_code == 200
            
            response_data = response.json()
            assert "answer" in response_data
            assert "sources" in response_data
            assert "source_links" in response_data
            assert "session_id" in response_data
            
            # Should get an answer, not "query failed"
            assert response_data["answer"] != "query failed"
            assert len(response_data["answer"]) > 0
            
        except Exception as e:
            pytest.fail(f"API endpoint test failed: {e}")
    
    def test_api_endpoint_with_course_question(self):
        """Test API endpoint with a course-specific question"""
        query_data = {
            "query": "What courses are available?",
            "session_id": None
        }
        
        try:
            response = self.client.post("/api/query", json=query_data)
            
            print(f"Course query status: {response.status_code}")
            
            if response.status_code == 500:
                error_detail = response.json().get("detail", "Unknown error")
                print(f"Course query error detail: {error_detail}")
                pytest.fail(f"API endpoint failed with course query: {error_detail}")
            
            assert response.status_code == 200
            
            response_data = response.json()
            print(f"Course query response: {response_data['answer']}")
            print(f"Course query sources: {response_data['sources']}")
            
            # Should get an answer, not "query failed"
            assert response_data["answer"] != "query failed"
            
        except Exception as e:
            pytest.fail(f"Course-specific API endpoint test failed: {e}")
    
    def test_course_stats_endpoint(self):
        """Test the course statistics endpoint"""
        try:
            response = self.client.get("/api/courses")
            
            print(f"Course stats status: {response.status_code}")
            
            if response.status_code == 500:
                error_detail = response.json().get("detail", "Unknown error")
                pytest.fail(f"Course stats endpoint failed: {error_detail}")
            
            assert response.status_code == 200
            
            response_data = response.json()
            print(f"Course stats: {response_data}")
            
            assert "total_courses" in response_data
            assert "course_titles" in response_data
            assert isinstance(response_data["total_courses"], int)
            assert isinstance(response_data["course_titles"], list)
            
        except Exception as e:
            pytest.fail(f"Course stats endpoint test failed: {e}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v", "-s"])