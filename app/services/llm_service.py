import json
import re
import traceback
from typing import Any, Dict, List, Optional

import openai
from loguru import logger

from app.models.base import (
    ActionableOutput,
    AnalysisResult,
    ConversationContext,
    Message,
    ToolCall,
    ToolResult,
)


class LLMService:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    async def analyze_message(self, context: ConversationContext) -> AnalysisResult:
        """Analyze the current message in context of conversation history."""
        prompt = self._create_analysis_prompt(context)
        system_prompt = """You are an expert sales conversation analyzer.
        Provide a JSON response with the following structure:
        {
            "intent": "string (e.g., inquiry, objection, buying signal)",
            "entities": ["list of key entities mentioned"],
            "sentiment": float (0.0 to 1.0),
            "key_points": ["list of main points to address"]
        }"""

        try:
            analysis = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            # Clean up the response - remove markdown code block if present
            analysis = re.sub(r"```json\s*|\s*```", "", analysis["content"]).strip()
            return AnalysisResult.model_validate_json(analysis)
        except Exception as e:
            logger.error(f"Error in message analysis: {str(e)}")
            raise

    async def decide_tool_usage(
        self,
        context: ConversationContext,
        analysis: AnalysisResult,
        tools: List[Dict[str, Any]],
    ) -> List[ToolCall]:
        """Decide which tools to use based on the analysis and conversation context."""
        system_prompt = """You are an expert sales assistant that decides which tools to use to gather information.
        IMPORTANT RULES:
        1. NEVER make assumptions about parameter values
        2. If a required parameter is missing or ambiguous, return a clarification request
        3. Only call tools when you have ALL required parameters with clear values
        4. For optional parameters, only include them if their value is explicitly clear

        When analyzing the conversation:
        - If prospect_id is required but not provided in context, request it
        - If query is ambiguous, ask for clarification
        - If fields are mentioned but unclear, ask which specific fields are needed
        - If entities are mentioned but unclear, ask which specific entities to filter by

        If no clarification is needed, use the provided tools to make tool calls.

        If clarification is needed, return a JSON object in this structure:
        {
            "detailed_analysis": "string (explanation of why clarification is needed)",
            "suggested_response_draft": "string (the actual clarification message to show to the user)",
            "internal_next_steps": [
                {
                    "action": "NEED_CLARIFICATION",
                    "details": {
                        "missing_parameters": ["list of parameters that need clarification"]
                    }
                }
            ],
            "tool_usage_log": [],
            "confidence_score": 1.0,
            "reasoning_trace": "string (explanation of which parameters are missing and why)"
        }
        """

        user_prompt = f"""Conversation History:
        {self._format_conversation_history(context.conversation_history)}

        Current Message:
        {context.current_prospect_message.sender}: {context.current_prospect_message.content}

        Analysis:
        Intent: {analysis.intent}
        Entities: {', '.join(analysis.entities)}
        Sentiment: {analysis.sentiment}
        Key Points: {', '.join(analysis.key_points)}

        Prospect ID: {context.prospect_id if context.prospect_id else 'Not provided'}

        Based on this information, which tools should be called to gather relevant information?
        Remember to only call tools when you have ALL required parameters with clear values.
        If any required parameters are missing or ambiguous, return a clarification request in the ActionableOutput format.
        """

        try:
            response = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
            )
            content = re.sub(r"```json\s*|\s*```", "", response["content"]).strip()
            if response["tool_calls"]:
                tool_calls = []
                for tool_data in response["tool_calls"]:
                    tool_calls.append(ToolCall(**tool_data))
                return tool_calls

            content_data = json.loads(content) if content else {}

            if "detailed_analysis" in content_data:
                return [
                    ToolCall(
                        tool_name="clarification_needed",
                        parameters=content_data,
                        result={},
                    )
                ]
            return []
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in tool usage decision: {str(e)}")
            return []

    async def generate_response(
        self,
        context: ConversationContext,
        analysis: AnalysisResult,
        tool_results: List[ToolResult],
    ) -> ActionableOutput:
        """Generate the final response and actions based on analysis and tool results."""
        prompt = self._create_response_prompt(context, analysis, tool_results)
        system_prompt = """You are an expert sales agent.
        Provide a JSON response with the following structure:
        {
            "detailed_analysis": "string (detailed understanding of the situation)",
            "suggested_response_draft": "string (response to send to prospect)",
            "internal_next_steps": [
                // Include only the relevant actions from the following types, but maintain their exact structure when used:
                {
                    "action": "UPDATE_CRM",
                    "details": {
                        "field": "interest_level",
                        "value": "high|medium|low"
                    }
                },
                {
                    "action": "SCHEDULE_FOLLOW_UP",
                    "details": {
                        "reason": "string (e.g., answered pricing query, sent case studies)",
                        "priority": "high|medium|low",
                        "suggested_delay_hours": number
                    }
                },
                {
                    "action": "FLAG_FOR_HUMAN_REVIEW",
                    "details": {
                        "reason": "string (e.g., complex objection, pricing negotiation)",
                        "priority": "high|medium|low",
                        "notes": "string (additional context for human reviewer)"
                    }
                },
                {
                    "action": "UPDATE_OPPORTUNITY",
                    "details": {
                        "stage": "string (e.g., discovery, proposal, negotiation)",
                        "probability": number (0-100),
                        "next_steps": "string"
                    }
                }
            ],
            "tool_usage_log": [
                // Include only the tools used from the Tool Results
                {
                    "tool_name": "string",
                    "parameters": {"param": "value"},
                    "result": {"result": "value"}
                }
            ],
            "confidence_score": float (0.0 to 1.0) (An estimated confidence score in the suggested_response_draft and internal_next_steps),
            "reasoning_trace": "string (explanation of decisions made)"
        }"""

        try:
            output = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            # Clean up the response - remove markdown code block if present
            output = re.sub(r"```json\s*|\s*```", "", output["content"]).strip()
            return ActionableOutput.model_validate_json(output)

        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call the LLM with the given prompt and optional tools.

        Returns:
            Dict containing:
            - content: str - The message content
            - tool_calls: Optional[List[Dict]] - List of tool calls if any
        """
        payload = {
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "tool_choice": "auto" if tools else None,
        }

        if tools:
            payload["tools"] = tools

        response = self.client.chat.completions.create(**payload)
        message = response.choices[0].message
        result = {"content": message.content or "", "tool_calls": None}
        # Handle tool calls in the response
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "tool_name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments),
                }
                for tool_call in message.tool_calls
            ]

        return result

    def _create_analysis_prompt(self, context: ConversationContext) -> str:
        """Create the prompt for message analysis."""
        conversation = self._format_conversation_history(context.conversation_history)

        return f"""Analyze the following sales conversation and provide a structured analysis:

        Conversation History:
        {conversation}

        Current Message:
        {context.current_prospect_message.sender}: {context.current_prospect_message.content}
        """

    def _create_response_prompt(
        self,
        context: ConversationContext,
        analysis: AnalysisResult,
        tool_results: List[ToolResult],
    ) -> str:
        """Create the prompt for response generation."""
        tool_results_str = "\n".join(
            [
                f"Tool: {tool.tool_name}\nParameters: {tool.parameters}\nResult: {tool.result}"
                for tool in tool_results
            ]
        )

        return f"""Based on the following information, generate a sales response and actions:

        Conversation Context:
        {context.current_prospect_message.sender}: {context.current_prospect_message.content}

        Analysis:
        Intent: {analysis.intent}
        Entities: {', '.join(analysis.entities)}
        Sentiment: {analysis.sentiment}
        Key Points: {', '.join(analysis.key_points)}

        Tool Results:
        {tool_results_str}
        """

    def _format_conversation_history(self, history: List[Message]) -> str:
        """Format conversation history for prompts."""
        return "\n".join([f"{msg.sender}: {msg.content}" for msg in history])
