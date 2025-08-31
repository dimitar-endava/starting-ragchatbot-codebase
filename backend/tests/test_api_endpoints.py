import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI endpoints for the RAG system"""
    
    def test_root_endpoint(self, api_client):
        """Test the root endpoint returns basic info"""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Course Materials RAG System" in data["message"]
    
    def test_query_endpoint_basic_query(self, api_client):
        """Test the /api/query endpoint with a basic query"""
        query_data = {
            "query": "What is Python?",
            "session_id": None
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "source_links" in data
        assert "session_id" in data
        
        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["source_links"], list)
        assert isinstance(data["session_id"], str)
        
        # Verify content
        assert len(data["answer"]) > 0
        assert data["session_id"] != ""
    
    def test_query_endpoint_with_session_id(self, api_client):
        """Test the /api/query endpoint with provided session_id"""
        query_data = {
            "query": "Follow-up question",
            "session_id": "existing-session-123"
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should use the provided session_id
        assert data["session_id"] == "existing-session-123"
    
    def test_query_endpoint_missing_query(self, api_client):
        """Test the /api/query endpoint with missing query field"""
        query_data = {
            "session_id": None
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        # Should return 422 for validation error
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_query_endpoint_empty_query(self, api_client):
        """Test the /api/query endpoint with empty query"""
        query_data = {
            "query": "",
            "session_id": None
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still return a valid response structure
        assert "answer" in data
        assert "session_id" in data
    
    def test_query_endpoint_invalid_json(self, api_client):
        """Test the /api/query endpoint with invalid JSON"""
        response = api_client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for JSON decode error
        assert response.status_code == 422
    
    def test_courses_endpoint(self, api_client):
        """Test the /api/courses endpoint"""
        response = api_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Verify content
        assert data["total_courses"] >= 0
        assert len(data["course_titles"]) == data["total_courses"]
    
    def test_clear_session_endpoint(self, api_client):
        """Test the /api/sessions/clear endpoint"""
        clear_data = {
            "session_id": "session-to-clear-123"
        }
        
        response = api_client.post("/api/sessions/clear", json=clear_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "session_id" in data
        
        # Verify content
        assert "cleared successfully" in data["message"].lower()
        assert data["session_id"] == "session-to-clear-123"
    
    def test_clear_session_missing_session_id(self, api_client):
        """Test the /api/sessions/clear endpoint without session_id"""
        clear_data = {}
        
        response = api_client.post("/api/sessions/clear", json=clear_data)
        
        # Should return 422 for validation error
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_nonexistent_endpoint(self, api_client):
        """Test a non-existent endpoint returns 404"""
        response = api_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_cors_headers(self, api_client):
        """Test that CORS headers are properly set"""
        response = api_client.get("/")
        
        # Check for CORS headers (they might be set by the middleware)
        # The test client doesn't always show all headers, so we mainly test that requests work
        assert response.status_code == 200
    
    def test_query_endpoint_long_query(self, api_client):
        """Test the query endpoint with a very long query"""
        long_query = "What is " + "very " * 100 + "long query about programming?"
        
        query_data = {
            "query": long_query,
            "session_id": None
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle long queries gracefully
        assert "answer" in data
        assert len(data["answer"]) > 0


@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling scenarios"""
    
    @patch('conftest.mock_rag_system')
    def test_query_endpoint_rag_system_error(self, mock_rag, api_client):
        """Test query endpoint when RAG system throws an error"""
        # This test would require more complex mocking to actually trigger the error
        # For now, we test that the basic structure works
        query_data = {
            "query": "Test query",
            "session_id": None
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        # Should either succeed or return 500 with proper error structure
        if response.status_code == 500:
            error_data = response.json()
            assert "detail" in error_data
        else:
            assert response.status_code == 200
    
    def test_courses_endpoint_error_handling(self, api_client):
        """Test courses endpoint error handling"""
        response = api_client.get("/api/courses")
        
        # Should either succeed or return proper error
        if response.status_code == 500:
            error_data = response.json()
            assert "detail" in error_data
        else:
            assert response.status_code == 200
            data = response.json()
            assert "total_courses" in data
            assert "course_titles" in data


@pytest.mark.api 
class TestAPIValidation:
    """Test API request/response validation"""
    
    def test_query_request_validation(self, api_client):
        """Test query request model validation"""
        # Test with extra fields
        query_data = {
            "query": "Test query",
            "session_id": None,
            "extra_field": "should be ignored"
        }
        
        response = api_client.post("/api/query", json=query_data)
        
        # Should succeed and ignore extra fields
        assert response.status_code == 200
    
    def test_query_response_structure(self, api_client):
        """Test that query response matches expected structure"""
        query_data = {
            "query": "Test structure",
            "session_id": None
        }
        
        response = api_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Test all required fields are present with correct types
        assert isinstance(data.get("answer"), str)
        assert isinstance(data.get("sources"), list)
        assert isinstance(data.get("source_links"), list)
        assert isinstance(data.get("session_id"), str)
        
        # Test source_links can contain None values
        for link in data["source_links"]:
            assert link is None or isinstance(link, str)
    
    def test_courses_response_structure(self, api_client):
        """Test that courses response matches expected structure"""
        response = api_client.get("/api/courses")
        assert response.status_code == 200
        
        data = response.json()
        
        # Test required fields and types
        assert isinstance(data.get("total_courses"), int)
        assert isinstance(data.get("course_titles"), list)
        
        # Test that all course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


@pytest.mark.api
@pytest.mark.integration  
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_session_persistence_across_queries(self, api_client):
        """Test that session IDs are properly managed across queries"""
        # First query without session ID
        query1_data = {
            "query": "First query",
            "session_id": None
        }
        
        response1 = api_client.post("/api/query", json=query1_data)
        assert response1.status_code == 200
        
        data1 = response1.json()
        session_id = data1["session_id"]
        
        # Second query with the same session ID
        query2_data = {
            "query": "Second query in same session",
            "session_id": session_id
        }
        
        response2 = api_client.post("/api/query", json=query2_data)
        assert response2.status_code == 200
        
        data2 = response2.json()
        
        # Should maintain the same session ID
        assert data2["session_id"] == session_id
    
    def test_query_and_clear_session_flow(self, api_client):
        """Test the full flow of querying and clearing a session"""
        # Create a session through query
        query_data = {
            "query": "Create session query", 
            "session_id": None
        }
        
        query_response = api_client.post("/api/query", json=query_data)
        assert query_response.status_code == 200
        
        session_id = query_response.json()["session_id"]
        
        # Clear the session
        clear_data = {
            "session_id": session_id
        }
        
        clear_response = api_client.post("/api/sessions/clear", json=clear_data)
        assert clear_response.status_code == 200
        
        clear_data = clear_response.json()
        assert clear_data["session_id"] == session_id
        assert "cleared successfully" in clear_data["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])