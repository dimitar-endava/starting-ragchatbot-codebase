import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import tempfile
import shutil

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, CourseChunk, Lesson


class TestSearchResults:
    """Test suite for SearchResults class"""
    
    def test_from_chroma_success(self):
        """Test creating SearchResults from ChromaDB results"""
        # Arrange
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course_title': 'Course A'}, {'course_title': 'Course B'}]],
            'distances': [[0.1, 0.2]]
        }
        
        # Act
        results = SearchResults.from_chroma(chroma_results)
        
        # Assert
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course_title': 'Course A'}, {'course_title': 'Course B'}]
        assert results.distances == [0.1, 0.2]
        assert results.lesson_links == []
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        # Arrange
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        # Act
        results = SearchResults.from_chroma(chroma_results)
        
        # Assert
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        # Act
        results = SearchResults.empty("Database connection failed")
        
        # Assert
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.lesson_links == []
        assert results.error == "Database connection failed"
        assert results.is_empty()
    
    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty()
    
    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=['doc1'],
            metadata=[{'title': 'test'}],
            distances=[0.1]
        )
        assert not results.is_empty()


class TestVectorStore:
    """Test suite for VectorStore class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, "test_chroma")
        self.embedding_model = "all-MiniLM-L6-v2"
        self.max_results = 5
        
        # Mock ChromaDB to avoid real database operations
        self.mock_client = Mock()
        self.mock_catalog_collection = Mock()
        self.mock_content_collection = Mock()
        
        with patch('chromadb.PersistentClient') as mock_persistent_client:
            mock_persistent_client.return_value = self.mock_client
            self.mock_client.get_or_create_collection.side_effect = [
                self.mock_catalog_collection,  # First call for course_catalog
                self.mock_content_collection   # Second call for course_content
            ]
            
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                self.vector_store = VectorStore(
                    chroma_path=self.chroma_path,
                    embedding_model=self.embedding_model,
                    max_results=self.max_results
                )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_success(self):
        """Test successful VectorStore initialization"""
        assert self.vector_store.max_results == self.max_results
        assert self.vector_store.client == self.mock_client
        assert self.vector_store.course_catalog == self.mock_catalog_collection
        assert self.vector_store.course_content == self.mock_content_collection
        
        # Verify collections were created with correct names
        assert self.mock_client.get_or_create_collection.call_count == 2
        calls = self.mock_client.get_or_create_collection.call_args_list
        assert calls[0][1]['name'] == 'course_catalog'
        assert calls[1][1]['name'] == 'course_content'
    
    def test_search_success(self):
        """Test successful search operation"""
        # Arrange
        query = "Python basics"
        mock_chroma_results = {
            'documents': [['Python is a programming language']],
            'metadatas': [[{'course_title': 'Python Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        self.mock_content_collection.query.return_value = mock_chroma_results
        
        # Mock lesson link retrieval
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps([{
                    'lesson_number': 1,
                    'lesson_title': 'Introduction',
                    'lesson_link': 'http://example.com/lesson1'
                }])
            }]
        }
        
        # Act
        results = self.vector_store.search(query)
        
        # Assert
        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) == 1
        assert results.documents[0] == 'Python is a programming language'
        assert results.metadata[0]['course_title'] == 'Python Course'
        assert results.lesson_links[0] == 'http://example.com/lesson1'
        
        # Verify search was called correctly
        self.mock_content_collection.query.assert_called_once_with(
            query_texts=[query],
            n_results=self.max_results,
            where=None
        )
    
    def test_search_with_course_name_filter(self):
        """Test search with course name filtering"""
        # Arrange
        query = "advanced concepts"
        course_name = "Advanced Python"
        resolved_course_title = "Advanced Python Programming"
        
        # Mock course name resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Advanced Python Programming']],
            'metadatas': [[{'title': resolved_course_title}]]
        }
        
        # Mock content search
        mock_chroma_results = {
            'documents': [['Advanced concepts explained']],
            'metadatas': [[{'course_title': resolved_course_title, 'lesson_number': 2}]],
            'distances': [[0.15]]
        }
        self.mock_content_collection.query.return_value = mock_chroma_results
        
        # Mock lesson link retrieval
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps([{
                    'lesson_number': 2,
                    'lesson_title': 'Advanced Concepts',
                    'lesson_link': 'http://example.com/lesson2'
                }])
            }]
        }
        
        # Act
        results = self.vector_store.search(query, course_name=course_name)
        
        # Assert
        assert not results.is_empty()
        assert results.documents[0] == 'Advanced concepts explained'
        
        # Verify course name resolution was called
        self.mock_catalog_collection.query.assert_called_with(
            query_texts=[course_name],
            n_results=1
        )
        
        # Verify content search used the resolved course title
        self.mock_content_collection.query.assert_called_once()
        call_args = self.mock_content_collection.query.call_args[1]
        assert call_args['where'] == {'course_title': resolved_course_title}
    
    def test_search_with_lesson_number_filter(self):
        """Test search with lesson number filtering"""
        # Arrange
        query = "lesson content"
        lesson_number = 3
        
        mock_chroma_results = {
            'documents': [['Lesson 3 content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        self.mock_content_collection.query.return_value = mock_chroma_results
        
        # Act
        results = self.vector_store.search(query, lesson_number=lesson_number)
        
        # Assert
        assert not results.is_empty()
        
        # Verify search used lesson number filter
        call_args = self.mock_content_collection.query.call_args[1]
        assert call_args['where'] == {'lesson_number': lesson_number}
    
    def test_search_with_both_filters(self):
        """Test search with both course name and lesson number filters"""
        # Arrange
        query = "specific content"
        course_name = "Python"
        lesson_number = 1
        resolved_course_title = "Python Programming"
        
        # Mock course name resolution
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Python Programming']],
            'metadatas': [[{'title': resolved_course_title}]]
        }
        
        # Mock content search
        mock_chroma_results = {
            'documents': [['Specific Python content']],
            'metadatas': [[{'course_title': resolved_course_title, 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        self.mock_content_collection.query.return_value = mock_chroma_results
        
        # Act
        results = self.vector_store.search(query, course_name=course_name, lesson_number=lesson_number)
        
        # Assert
        assert not results.is_empty()
        
        # Verify search used both filters
        call_args = self.mock_content_collection.query.call_args[1]
        expected_filter = {
            "$and": [
                {"course_title": resolved_course_title},
                {"lesson_number": lesson_number}
            ]
        }
        assert call_args['where'] == expected_filter
    
    def test_search_course_not_found(self):
        """Test search when course name cannot be resolved"""
        # Arrange
        query = "content"
        course_name = "Nonexistent Course"
        
        # Mock course name resolution failure
        self.mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        # Act
        results = self.vector_store.search(query, course_name=course_name)
        
        # Assert
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()
    
    def test_search_empty_results(self):
        """Test search returning empty results"""
        # Arrange
        query = "nonexistent content"
        
        mock_chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        self.mock_content_collection.query.return_value = mock_chroma_results
        
        # Act
        results = self.vector_store.search(query)
        
        # Assert
        assert results.is_empty()
        assert results.error is None
    
    def test_search_exception_handling(self):
        """Test search handles exceptions gracefully"""
        # Arrange
        query = "test query"
        self.mock_content_collection.query.side_effect = Exception("ChromaDB connection failed")
        
        # Act
        results = self.vector_store.search(query)
        
        # Assert
        assert results.error == "Search error: ChromaDB connection failed"
        assert results.is_empty()
    
    def test_add_course_metadata(self):
        """Test adding course metadata to the catalog"""
        # Arrange
        lessons = [
            Lesson(lesson_number=1, title="Introduction", lesson_link="http://example.com/1"),
            Lesson(lesson_number=2, title="Basics", lesson_link="http://example.com/2")
        ]
        course = Course(
            title="Test Course",
            course_link="http://example.com/course",
            instructor="Test Instructor",
            lessons=lessons
        )
        
        # Act
        self.vector_store.add_course_metadata(course)
        
        # Assert
        self.mock_catalog_collection.add.assert_called_once()
        call_args = self.mock_catalog_collection.add.call_args[1]
        
        assert call_args['documents'] == ["Test Course"]
        assert call_args['ids'] == ["Test Course"]
        
        metadata = call_args['metadatas'][0]
        assert metadata['title'] == "Test Course"
        assert metadata['instructor'] == "Test Instructor"
        assert metadata['course_link'] == "http://example.com/course"
        assert metadata['lesson_count'] == 2
        
        # Verify lessons JSON
        lessons_data = json.loads(metadata['lessons_json'])
        assert len(lessons_data) == 2
        assert lessons_data[0]['lesson_number'] == 1
        assert lessons_data[0]['lesson_title'] == "Introduction"
        assert lessons_data[0]['lesson_link'] == "http://example.com/1"
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        # Arrange
        chunks = [
            CourseChunk(
                content="First chunk content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Second chunk content", 
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        # Act
        self.vector_store.add_course_content(chunks)
        
        # Assert
        self.mock_content_collection.add.assert_called_once()
        call_args = self.mock_content_collection.add.call_args[1]
        
        assert call_args['documents'] == ["First chunk content", "Second chunk content"]
        assert call_args['ids'] == ["Test_Course_0", "Test_Course_1"]
        
        metadatas = call_args['metadatas']
        assert len(metadatas) == 2
        assert metadatas[0]['course_title'] == "Test Course"
        assert metadatas[0]['lesson_number'] == 1
        assert metadatas[0]['chunk_index'] == 0
    
    def test_add_course_content_empty(self):
        """Test adding empty course content list"""
        # Act
        self.vector_store.add_course_content([])
        
        # Assert
        self.mock_content_collection.add.assert_not_called()
    
    def test_clear_all_data(self):
        """Test clearing all data from collections"""
        # Reset the mock call count for this test
        self.mock_client.get_or_create_collection.reset_mock()
        
        # Set up new mock collections for recreation
        new_catalog_collection = Mock()
        new_content_collection = Mock()
        self.mock_client.get_or_create_collection.side_effect = [
            new_catalog_collection,
            new_content_collection
        ]
        
        # Act
        self.vector_store.clear_all_data()
        
        # Assert
        assert self.mock_client.delete_collection.call_count == 2
        calls = self.mock_client.delete_collection.call_args_list
        assert calls[0][0][0] == "course_catalog"
        assert calls[1][0][0] == "course_content"
        
        # Verify collections were recreated
        assert self.mock_client.get_or_create_collection.call_count == 2  # 2 recreated
        
        # Verify the new collections were assigned
        assert self.vector_store.course_catalog == new_catalog_collection
        assert self.vector_store.course_content == new_content_collection
    
    def test_get_existing_course_titles(self):
        """Test retrieving existing course titles"""
        # Arrange
        self.mock_catalog_collection.get.return_value = {
            'ids': ['Course A', 'Course B', 'Course C']
        }
        
        # Act
        titles = self.vector_store.get_existing_course_titles()
        
        # Assert
        assert titles == ['Course A', 'Course B', 'Course C']
        self.mock_catalog_collection.get.assert_called_once()
    
    def test_get_existing_course_titles_empty(self):
        """Test retrieving course titles from empty catalog"""
        # Arrange
        self.mock_catalog_collection.get.return_value = {'ids': []}
        
        # Act
        titles = self.vector_store.get_existing_course_titles()
        
        # Assert
        assert titles == []
    
    def test_get_course_count(self):
        """Test getting course count"""
        # Arrange
        self.mock_catalog_collection.get.return_value = {
            'ids': ['Course A', 'Course B']
        }
        
        # Act
        count = self.vector_store.get_course_count()
        
        # Assert
        assert count == 2
    
    def test_get_lesson_link_success(self):
        """Test retrieving lesson link successfully"""
        # Arrange
        course_title = "Test Course"
        lesson_number = 2
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps([
                    {'lesson_number': 1, 'lesson_link': 'http://example.com/1'},
                    {'lesson_number': 2, 'lesson_link': 'http://example.com/2'},
                    {'lesson_number': 3, 'lesson_link': 'http://example.com/3'}
                ])
            }]
        }
        
        # Act
        link = self.vector_store.get_lesson_link(course_title, lesson_number)
        
        # Assert
        assert link == 'http://example.com/2'
        self.mock_catalog_collection.get.assert_called_once_with(ids=[course_title])
    
    def test_get_lesson_link_not_found(self):
        """Test retrieving lesson link when lesson doesn't exist"""
        # Arrange
        course_title = "Test Course"
        lesson_number = 99
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps([
                    {'lesson_number': 1, 'lesson_link': 'http://example.com/1'}
                ])
            }]
        }
        
        # Act
        link = self.vector_store.get_lesson_link(course_title, lesson_number)
        
        # Assert
        assert link is None
    
    def test_build_filter_no_filters(self):
        """Test filter building with no parameters"""
        # Act
        filter_dict = self.vector_store._build_filter(None, None)
        
        # Assert
        assert filter_dict is None
    
    def test_build_filter_course_only(self):
        """Test filter building with course title only"""
        # Act
        filter_dict = self.vector_store._build_filter("Test Course", None)
        
        # Assert
        assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self):
        """Test filter building with lesson number only"""
        # Act
        filter_dict = self.vector_store._build_filter(None, 3)
        
        # Assert
        assert filter_dict == {"lesson_number": 3}
    
    def test_build_filter_both(self):
        """Test filter building with both parameters"""
        # Act
        filter_dict = self.vector_store._build_filter("Test Course", 2)
        
        # Assert
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
        assert filter_dict == expected


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with more realistic scenarios"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, "integration_chroma")
        
        # Use mocks but allow more realistic interactions
        self.mock_client = Mock()
        self.mock_catalog_collection = Mock()
        self.mock_content_collection = Mock()
        
        with patch('chromadb.PersistentClient') as mock_persistent_client:
            mock_persistent_client.return_value = self.mock_client
            self.mock_client.get_or_create_collection.side_effect = [
                self.mock_catalog_collection,
                self.mock_content_collection
            ]
            
            with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                self.vector_store = VectorStore(
                    chroma_path=self.chroma_path,
                    embedding_model="all-MiniLM-L6-v2",
                    max_results=5
                )
    
    def teardown_method(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_realistic_course_workflow(self):
        """Test a realistic workflow of adding and searching course content"""
        # Step 1: Add course metadata
        course = Course(
            title="Python for Beginners",
            course_link="http://example.com/python",
            instructor="Jane Doe",
            lessons=[
                Lesson(lesson_number=1, title="Variables and Types", lesson_link="http://example.com/python/1"),
                Lesson(lesson_number=2, title="Control Flow", lesson_link="http://example.com/python/2")
            ]
        )
        self.vector_store.add_course_metadata(course)
        
        # Step 2: Add course content
        chunks = [
            CourseChunk(
                content="Python variables store data values. Python has no command for declaring a variable.",
                course_title="Python for Beginners",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Python supports the usual logical conditions from mathematics.",
                course_title="Python for Beginners", 
                lesson_number=2,
                chunk_index=1
            )
        ]
        self.vector_store.add_course_content(chunks)
        
        # Step 3: Mock search response
        self.mock_catalog_collection.query.return_value = {
            'documents': [['Python for Beginners']],
            'metadatas': [[{'title': 'Python for Beginners'}]]
        }
        
        self.mock_content_collection.query.return_value = {
            'documents': [['Python variables store data values. Python has no command for declaring a variable.']],
            'metadatas': [[{'course_title': 'Python for Beginners', 'lesson_number': 1, 'chunk_index': 0}]],
            'distances': [[0.1]]
        }
        
        self.mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps([
                    {'lesson_number': 1, 'lesson_title': 'Variables and Types', 'lesson_link': 'http://example.com/python/1'}
                ])
            }]
        }
        
        # Step 4: Search for content
        results = self.vector_store.search("variables", course_name="Python")
        
        # Assert
        assert not results.is_empty()
        assert "Python variables store data values" in results.documents[0]
        assert results.metadata[0]['lesson_number'] == 1
        assert results.lesson_links[0] == 'http://example.com/python/1'
        
        # Verify all the expected calls were made
        self.mock_catalog_collection.add.assert_called_once()
        self.mock_content_collection.add.assert_called_once()
        self.mock_catalog_collection.query.assert_called_once()
        self.mock_content_collection.query.assert_called_once()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])