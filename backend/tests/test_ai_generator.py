import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"

        # Mock anthropic client to avoid real API calls
        with patch("anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)

    def test_init(self):
        """Test AIGenerator initialization"""
        # The mock client should be set
        assert self.ai_generator.client == self.mock_client
        assert self.ai_generator.model == self.model

        # Check base parameters
        expected_params = {"model": self.model, "temperature": 0, "max_tokens": 800}
        assert self.ai_generator.base_params == expected_params

    def test_generate_response_without_tools_success(self):
        """Test successful response generation without tools"""
        # Arrange
        query = "What is Python?"
        expected_response = "Python is a programming language."

        # Mock successful API response
        mock_response = Mock()
        mock_response.content = [Mock(text=expected_response)]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = self.ai_generator.generate_response(query)

        # Assert
        assert result == expected_response

        # Verify API call parameters
        call_args = self.mock_client.messages.create.call_args[1]
        assert call_args["model"] == self.model
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": query}]
        assert "system" in call_args
        assert "tools" not in call_args

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        # Arrange
        query = "What about Django?"
        history = "User: What is Python?\nAssistant: Python is a programming language."
        expected_response = "Django is a Python web framework."

        # Mock successful API response
        mock_response = Mock()
        mock_response.content = [Mock(text=expected_response)]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = self.ai_generator.generate_response(
            query, conversation_history=history
        )

        # Assert
        assert result == expected_response

        # Verify system content includes history
        call_args = self.mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert history in system_content
        assert self.ai_generator.SYSTEM_PROMPT in system_content

    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but not used"""
        # Arrange
        query = "What is 2+2?"
        expected_response = "2+2 equals 4."
        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Mock successful API response without tool use
        mock_response = Mock()
        mock_response.content = [Mock(text=expected_response)]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = self.ai_generator.generate_response(query, tools=tools)

        # Assert
        assert result == expected_response

        # Verify tools were included in API call
        call_args = self.mock_client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}

    def test_generate_response_with_tool_use_success(self):
        """Test successful tool execution flow"""
        # Arrange
        query = "Search for Python basics"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Found Python basics content"

        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python basics"}
        mock_tool_content.id = "tool_123"
        mock_initial_response.content = [mock_tool_content]

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Here's what I found about Python basics...")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Here's what I found about Python basics..."

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify two API calls were made (initial + follow-up)
        assert self.mock_client.messages.create.call_count == 2

    def test_generate_response_tool_execution_error(self):
        """Test handling when tool execution returns an error"""
        # Arrange
        query = "Search for nonexistent content"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Mock tool manager with error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "No relevant content found."

        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "nonexistent"}
        mock_tool_content.id = "tool_456"
        mock_initial_response.content = [mock_tool_content]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="I couldn't find any relevant content.")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "I couldn't find any relevant content."

        # Verify tool was executed despite error
        mock_tool_manager.execute_tool.assert_called_once()

    def test_generate_response_multiple_tool_calls(self):
        """Test handling multiple tool calls in one response"""
        # Arrange
        query = "Search multiple things"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        # Mock initial response with multiple tool uses
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"

        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "search_course_content"
        mock_tool1.input = {"query": "topic1"}
        mock_tool1.id = "tool1"

        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "search_course_content"
        mock_tool2.input = {"query": "topic2"}
        mock_tool2.id = "tool2"

        mock_initial_response.content = [mock_tool1, mock_tool2]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Combined results from multiple searches")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Combined results from multiple searches"

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_generate_response_api_error(self):
        """Test handling of Anthropic API errors"""
        # Arrange
        query = "Test query"

        # Mock API error
        api_error = Exception("API key invalid")
        self.mock_client.messages.create.side_effect = api_error

        # Act
        result = self.ai_generator.generate_response(query)

        # Assert - should return error message instead of raising
        assert "I encountered an error: API key invalid" in result

    def test_handle_tool_execution_message_building(self):
        """Test that tool execution builds messages correctly"""
        # Arrange
        query = "Test query"
        tools = [{"name": "test_tool", "description": "Test tool"}]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Mock initial response
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "test_tool"
        mock_tool_content.input = {"param": "value"}
        mock_tool_content.id = "tool_123"
        mock_initial_response.content = [mock_tool_content]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Final answer"

        # Check the second API call (final response) has correct message structure
        final_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = final_call_args["messages"]

        # Should have: initial user message, assistant response with tool use, user message with tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check tool result format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Tool result"

    def test_system_prompt_content(self):
        """Test that system prompt contains expected instructions"""
        system_prompt = AIGenerator.SYSTEM_PROMPT

        # Check for key instructions
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "Course-specific content questions" in system_prompt
        assert "Sequential tool usage" in system_prompt
        assert "When to use multiple tools" in system_prompt
        assert "Brief, Concise and focused" in system_prompt

    def test_sequential_tool_calling_two_rounds(self):
        """Test successful sequential tool calling with 2 rounds"""
        # Arrange
        query = "Search for lesson 4 of MCP course, then find another course with similar content"
        tools = [
            {"name": "get_course_outline", "description": "Get outline"},
            {"name": "search_course_content", "description": "Search content"},
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "MCP Introduction - Lesson 4: Server Implementation",  # First tool call
            "Found Python Web Dev course with server implementation content",  # Second tool call
        ]

        # Mock responses for 2 rounds
        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool_content1 = Mock()
        mock_tool_content1.type = "tool_use"
        mock_tool_content1.name = "get_course_outline"
        mock_tool_content1.input = {"course_name": "MCP"}
        mock_tool_content1.id = "tool1"
        mock_round1_response.content = [mock_tool_content1]

        # Round 2: Tool use response
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_tool_content2 = Mock()
        mock_tool_content2.type = "tool_use"
        mock_tool_content2.name = "search_course_content"
        mock_tool_content2.input = {"query": "server implementation"}
        mock_tool_content2.id = "tool2"
        mock_round2_response.content = [mock_tool_content2]

        # Final response after 2 rounds
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on the research, lesson 4 covers server implementation...")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert (
            result == "Based on the research, lesson 4 covers server implementation..."
        )
        assert mock_tool_manager.execute_tool.call_count == 2
        assert self.mock_client.messages.create.call_count == 3  # 2 rounds + final

    def test_sequential_tool_calling_early_termination(self):
        """Test sequential tool calling that terminates after first round (no more tools)"""
        # Arrange
        query = "What is Python?"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python programming content"

        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python"}
        mock_tool_content.id = "tool1"
        mock_round1_response.content = [mock_tool_content]

        # Round 2: Direct response (no tools)
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "end_turn"
        mock_round2_response.content = [
            Mock(text="Python is a programming language...")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Python is a programming language..."
        assert mock_tool_manager.execute_tool.call_count == 1
        assert self.mock_client.messages.create.call_count == 2  # 2 API calls total

    def test_sequential_tool_calling_max_rounds_reached(self):
        """Test that sequential calling stops after max rounds"""
        # Arrange
        query = "Complex multi-step query"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Some search result"

        # Both rounds return tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_id"
        mock_tool_response.content = [mock_tool_content]

        # Final response after max rounds
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer after max rounds")]

        self.mock_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2
            mock_final_response,  # Final call
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager, max_rounds=2
        )

        # Assert
        assert result == "Final answer after max rounds"
        assert mock_tool_manager.execute_tool.call_count == 2  # Max 2 rounds
        assert self.mock_client.messages.create.call_count == 3  # 2 rounds + final

    def test_sequential_tool_calling_tool_failure(self):
        """Test handling of tool execution failure in sequential rounds"""
        # Arrange
        query = "Search query"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool1"
        mock_round1_response.content = [mock_tool_content]

        # Final response after tool failure
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="I encountered some issues but here's what I can tell you...")
        ]

        self.mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert "I encountered some issues" in result
        assert mock_tool_manager.execute_tool.call_count == 1
        assert self.mock_client.messages.create.call_count == 2  # Round 1 + final

    def test_sequential_message_building(self):
        """Test that message history is built correctly across rounds"""
        # Arrange
        query = "Test query"
        tools = [{"name": "search_course_content", "description": "Search content"}]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"

        # Round 1 tool response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool1"
        mock_round1_response.content = [mock_tool_content]

        # Round 2 direct response
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "end_turn"
        mock_round2_response.content = [Mock(text="Final answer")]

        self.mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Final answer"

        # Check message building in round 2
        round2_call_args = self.mock_client.messages.create.call_args_list[1][1]
        messages = round2_call_args["messages"]

        # Should have: initial user, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"


class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator with more realistic scenarios"""

    def setup_method(self):
        """Set up test fixtures for integration tests"""
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"

        with patch("anthropic.Anthropic") as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)

    def test_realistic_course_query_with_tool_use(self):
        """Test a realistic course-related query that should trigger tool use"""
        # Arrange
        query = "What does the MCP course say about server implementation?"
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course name filter",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[MCP Introduction - Lesson 2]\nServer implementation involves creating handlers for tools and resources..."

        # Mock tool use response
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {
            "query": "server implementation",
            "course_name": "MCP",
        }
        mock_tool_content.id = "search_123"
        mock_initial_response.content = [mock_tool_content]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="According to the MCP course, server implementation involves creating handlers for tools and resources..."
            )
        ]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Act
        result = self.ai_generator.generate_response(
            query, tools=tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert "server implementation involves creating handlers" in result

        # Verify tool was called with appropriate parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="server implementation", course_name="MCP"
        )

    def test_general_knowledge_query_no_tool_use(self):
        """Test that general knowledge queries don't trigger tools"""
        # Arrange
        query = "What is the capital of France?"
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Mock direct response (no tool use)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="The capital of France is Paris.")]
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = self.ai_generator.generate_response(query, tools=tools)

        # Assert
        assert result == "The capital of France is Paris."

        # Verify only one API call was made (no tool execution)
        assert self.mock_client.messages.create.call_count == 1


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
