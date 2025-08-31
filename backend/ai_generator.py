from typing import Any, Dict, List, Optional, Tuple

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials  
- **get_course_outline**: Use for questions about course structure, lesson lists, or course overviews
- **Sequential tool usage**: You may use tools multiple times if needed for complex queries requiring information from different sources
- **When to use multiple tools**: For comparisons, multi-part questions, or when initial results suggest additional searches would help
- **When to stop**: When you have sufficient information to provide a complete answer, or after gathering information from multiple relevant sources
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use search_course_content first, then answer
- **Course outline/structure questions**: Use get_course_outline first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

When responding to outline queries, include:
- Course title
- Course link (if available)  
- Complete lesson list with numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with support for sequential tool usage.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages with user query
        messages = [{"role": "user", "content": query}]

        # Sequential tool execution rounds
        for round_num in range(max_rounds):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            try:
                # Get response from Claude
                response = self.client.messages.create(**api_params)

                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use" and tool_manager:
                    # Execute tools and continue conversation
                    messages, continue_rounds = self._handle_tool_execution_round(
                        response, messages, tool_manager
                    )

                    # If tool execution failed or no continuation needed, stop
                    if not continue_rounds:
                        # Try to get final response without tools
                        final_params = {
                            **self.base_params,
                            "messages": messages.copy(),
                            "system": system_content,
                        }
                        try:
                            final_response = self.client.messages.create(**final_params)
                            return final_response.content[0].text
                        except:
                            return "I encountered an issue processing your request."

                else:
                    # No tool use or no tool manager - return direct response
                    return response.content[0].text

            except Exception as e:
                # API error - return error message
                return f"I encountered an error: {str(e)}"

        # If we've completed max rounds, make final response without tools
        final_params = {
            **self.base_params,
            "messages": messages.copy(),
            "system": system_content,
        }

        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except:
            return "I've completed my research but encountered an issue generating the final response."

    def _handle_tool_execution_round(
        self, response, messages: List[Dict], tool_manager
    ) -> Tuple[List[Dict], bool]:
        """
        Handle tool execution for one round and update message history.

        Args:
            response: The response containing tool use requests
            messages: Current conversation messages
            tool_manager: Manager to execute tools

        Returns:
            tuple: (updated_messages, should_continue_rounds)
        """
        try:
            # Add AI's tool use response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result,
                            }
                        )
                    except Exception as e:
                        # Handle individual tool failures
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Tool execution failed: {str(e)}",
                            }
                        )

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                return messages, True  # Continue with next round
            else:
                return messages, False  # No tools executed, stop

        except Exception as e:
            # Tool execution failed completely
            return messages, False

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
