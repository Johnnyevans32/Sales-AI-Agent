import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from app.models.base import (
    ActionableOutput,
    ConversationContext,
    Message,
)
from app.services.llm_service import LLMService
from app.tools.knowledge_tool import KnowledgeAugmentationTool

# Load environment variables
load_dotenv()


@pytest.fixture
def llm_service():
    return LLMService(api_key=os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def knowledge_tool():
    return KnowledgeAugmentationTool()


@pytest.fixture
def sample_conversation():
    return ConversationContext(
        conversation_history=[
            Message(
                sender="prospect",
                content="Hi, I'm interested in learning more about your AI sales assistant.",
                timestamp=datetime.now(),
            ),
            Message(
                sender="agent",
                content="Hello! I'd be happy to tell you more about our AI sales assistant. What specific aspects would you like to know about?",
                timestamp=datetime.now(),
            ),
        ],
        current_prospect_message=Message(
            sender="prospect",
            content="What's the pricing like? And do you have any case studies from the technology industry?",
            timestamp=datetime.now(),
        ),
        prospect_id="PROSPECT001",
    )


@pytest.mark.asyncio
async def test_message_analysis(
    llm_service: LLMService, sample_conversation: ConversationContext
) -> None:
    """Test the message analysis functionality."""
    analysis = await llm_service.analyze_message(sample_conversation)

    assert analysis.intent in ["inquiry", "pricing_request", "case_study_request"]
    assert "pricing" in [entity.lower() for entity in analysis.entities]
    assert "case studies" in [entity.lower() for entity in analysis.entities]
    assert 0 <= analysis.sentiment <= 1
    assert len(analysis.key_points) > 0


def test_knowledge_tool_fetch_prospect(
    knowledge_tool: KnowledgeAugmentationTool,
) -> None:
    """Test the prospect details fetching functionality."""
    details = knowledge_tool.fetch_prospect_details("PROSPECT001")

    assert isinstance(details, dict)
    assert "prospect_id" in details
    assert "company_name" in details
    assert "industry" in details


def test_knowledge_tool_query(knowledge_tool: KnowledgeAugmentationTool) -> None:
    """Test the knowledge base query functionality."""
    results = knowledge_tool.query_knowledge_base(
        "pricing information for technology companies"
    )

    assert isinstance(results, list)
    if results:  # If we have any results
        assert "content" in results[0]
        assert "source" in results[0]
        assert "similarity_score" in results[0]
        assert 0 <= results[0]["similarity_score"] <= 1


@pytest.mark.asyncio
async def test_full_response_generation(
    llm_service: LLMService,
    knowledge_tool: KnowledgeAugmentationTool,
    sample_conversation: ConversationContext,
) -> None:
    """Test the complete response generation process."""
    analysis = await llm_service.analyze_message(sample_conversation)

    tool_calls = await llm_service.decide_tool_usage(
        context=sample_conversation, analysis=analysis, tools=knowledge_tool.tools
    )

    if tool_calls and tool_calls[0].tool_name == "clarification_needed":
        output = ActionableOutput(**tool_calls[0].parameters)

    else:
        tool_results = await knowledge_tool.process_tool_calls(tool_calls)

        output = await llm_service.generate_response(
            context=sample_conversation, analysis=analysis, tool_results=tool_results
        )

    assert isinstance(output.detailed_analysis, str)
    assert isinstance(output.suggested_response_draft, str)
    assert isinstance(output.internal_next_steps, list)
    assert isinstance(output.tool_usage_log, list)
    assert 0 <= output.confidence_score <= 1
    assert len(output.detailed_analysis) > 0
    assert len(output.suggested_response_draft) > 0
