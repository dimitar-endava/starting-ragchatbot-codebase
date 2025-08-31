import os
import sys
from unittest.mock import MagicMock, Mock

import pytest

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute method"""

    def setup_method(self):
        """Set up test fixtures"""
        # Mock vector store
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search(self):
        """Test CourseSearchTool with successful search results"""
        # Arrange
        mock_results = SearchResults(
            documents=["This is course content about Python basics"],
            metadata=[
                {
                    "course_title": "Python Programming Course",
                    "lesson_number": 1,
                    "chunk_index": 0,
                }
            ],
            distances=[0.1],
            lesson_links=["http://example.com/lesson1"],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("Python basics")

        # Assert
        assert isinstance(result, str)
        assert "Python Programming Course" in result
        assert "This is course content about Python basics" in result
        assert "[Python Programming Course - Lesson 1]" in result

        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics", course_name=None, lesson_number=None
        )

        # Check that sources were tracked
        assert self.search_tool.last_sources == ["Python Programming Course - Lesson 1"]
        assert self.search_tool.last_source_links == ["http://example.com/lesson1"]

    def test_execute_with_course_name_filter(self):
        """Test CourseSearchTool with course name filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Advanced Python concepts"],
            metadata=[{"course_title": "Advanced Python Course", "lesson_number": 2}],
            distances=[0.2],
            lesson_links=["http://example.com/lesson2"],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("concepts", course_name="Advanced Python")

        # Assert
        assert "Advanced Python Course" in result
        assert "Advanced Python concepts" in result

        # Verify search parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="concepts", course_name="Advanced Python", lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self):
        """Test CourseSearchTool with lesson number filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Python Course", "lesson_number": 3}],
            distances=[0.15],
            lesson_links=["http://example.com/lesson3"],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("content", lesson_number=3)

        # Assert
        assert "Python Course - Lesson 3" in result
        assert "Lesson 3 content" in result

        # Verify search parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="content", course_name=None, lesson_number=3
        )

    def test_execute_with_both_filters(self):
        """Test CourseSearchTool with both course name and lesson number filters"""
        # Arrange
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Python Course", "lesson_number": 1}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(
            "content", course_name="Python", lesson_number=1
        )

        # Assert
        assert "Python Course - Lesson 1" in result

        # Verify search parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="content", course_name="Python", lesson_number=1
        )

    def test_execute_empty_results(self):
        """Test CourseSearchTool when no results are found"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("nonexistent topic")

        # Assert
        assert result == "No relevant content found."

        # Check that sources were cleared
        assert self.search_tool.last_sources == []
        assert self.search_tool.last_source_links == []

    def test_execute_empty_results_with_filters(self):
        """Test empty results with filters shows filter information"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(
            "topic", course_name="Nonexistent Course", lesson_number=5
        )

        # Assert
        assert (
            "No relevant content found in course 'Nonexistent Course' in lesson 5."
            in result
        )

    def test_execute_search_error(self):
        """Test CourseSearchTool when vector store returns an error"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="ChromaDB connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("any query")

        # Assert
        assert result == "ChromaDB connection failed"

    def test_multiple_search_results(self):
        """Test formatting of multiple search results"""
        # Arrange
        mock_results = SearchResults(
            documents=["First result about Python", "Second result about programming"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            lesson_links=["http://example.com/a1", "http://example.com/b2"],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("Python programming")

        # Assert
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First result about Python" in result
        assert "Second result about programming" in result

        # Check sources tracking
        expected_sources = ["Course A - Lesson 1", "Course B - Lesson 2"]
        expected_links = ["http://example.com/a1", "http://example.com/b2"]
        assert self.search_tool.last_sources == expected_sources
        assert self.search_tool.last_source_links == expected_links

    def test_metadata_missing_lesson_number(self):
        """Test handling when lesson_number is missing from metadata"""
        # Arrange
        mock_results = SearchResults(
            documents=["Course overview content"],
            metadata=[
                {
                    "course_title": "Overview Course",
                    # lesson_number is missing
                }
            ],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute("overview")

        # Assert
        assert "[Overview Course]" in result  # No lesson number in header
        assert "Course overview content" in result

        # Check sources (no lesson number)
        assert self.search_tool.last_sources == ["Overview Course"]

    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted for Anthropic API"""
        # Act
        definition = self.search_tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Check required fields
        assert definition["input_schema"]["required"] == ["query"]


class TestToolManager:
    """Test suite for ToolManager"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_vector_store = Mock()

    def test_register_tool(self):
        """Test tool registration"""
        # Arrange
        search_tool = CourseSearchTool(self.mock_vector_store)

        # Act
        self.tool_manager.register_tool(search_tool)

        # Assert
        assert "search_course_content" in self.tool_manager.tools
        assert self.tool_manager.tools["search_course_content"] == search_tool

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        # Arrange
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)

        # Act
        definitions = self.tool_manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_success(self):
        """Test successful tool execution"""
        # Arrange
        search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(search_tool)

        # Mock successful search
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool_manager.execute_tool("search_course_content", query="test")

        # Assert
        assert "Test Course" in result
        assert "Test content" in result

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        # Act
        result = self.tool_manager.execute_tool("nonexistent_tool", query="test")

        # Assert
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self):
        """Test retrieving sources from last search"""
        # Arrange
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = ["Test Course - Lesson 1"]
        self.tool_manager.register_tool(search_tool)

        # Act
        sources = self.tool_manager.get_last_sources()

        # Assert
        assert sources == ["Test Course - Lesson 1"]

    def test_get_last_source_links(self):
        """Test retrieving source links from last search"""
        # Arrange
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_source_links = ["http://example.com/lesson1"]
        self.tool_manager.register_tool(search_tool)

        # Act
        links = self.tool_manager.get_last_source_links()

        # Assert
        assert links == ["http://example.com/lesson1"]

    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        # Arrange
        search_tool = CourseSearchTool(self.mock_vector_store)
        search_tool.last_sources = ["Test Course"]
        search_tool.last_source_links = ["http://example.com"]
        self.tool_manager.register_tool(search_tool)

        # Act
        self.tool_manager.reset_sources()

        # Assert
        assert search_tool.last_sources == []
        assert search_tool.last_source_links == []


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
