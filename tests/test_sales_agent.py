import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from app.models.base import ConversationContext, Message
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
        prospect_id="TEST123",
    )


@pytest.mark.asyncio
async def test_message_analysis(llm_service, sample_conversation):
    """Test the message analysis functionality."""
    analysis = await llm_service.analyze_message(sample_conversation)

    assert analysis.intent in ["inquiry", "pricing_request", "case_study_request"]
    assert "pricing" in [entity.lower() for entity in analysis.entities]
    assert "case studies" in [entity.lower() for entity in analysis.entities]
    assert 0 <= analysis.sentiment <= 1
    assert len(analysis.key_points) > 0


def test_knowledge_tool_fetch_prospect(knowledge_tool):
    """Test the prospect details fetching functionality."""
    details = knowledge_tool.fetch_prospect_details("TEST123")

    assert isinstance(details, dict)
    assert "prospect_id" in details
    assert "company_name" in details
    assert "industry" in details


def test_knowledge_tool_query(knowledge_tool):
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
    llm_service, knowledge_tool, sample_conversation
):
    """Test the complete response generation process."""
    # First analyze the message
    analysis = await llm_service.analyze_message(sample_conversation)

    # Get tool results
    tool_calls = []

    # Fetch prospect details
    prospect_details = knowledge_tool.process_tool_call(
        "fetch_prospect_details", {"prospect_id": sample_conversation.prospect_id}
    )
    tool_calls.append(
        {
            "tool_name": "fetch_prospect_details",
            "parameters": {"prospect_id": sample_conversation.prospect_id},
            "result": prospect_details,
        }
    )

    # Query knowledge base
    knowledge_results = knowledge_tool.process_tool_call(
        "query_knowledge_base",
        {
            "query": f"{analysis.intent} {sample_conversation.current_prospect_message.content}",
            "filters": {"entities": analysis.entities},
        },
    )
    tool_calls.append(
        {
            "tool_name": "query_knowledge_base",
            "parameters": {
                "query": f"{analysis.intent} {sample_conversation.current_prospect_message.content}",
                "filters": {"entities": analysis.entities},
            },
            "result": knowledge_results,
        }
    )

    # Generate final response
    output = await llm_service.generate_response(
        context=sample_conversation, analysis=analysis, tool_results=tool_calls
    )

    assert isinstance(output.detailed_analysis, str)
    assert isinstance(output.suggested_response_draft, str)
    assert isinstance(output.internal_next_steps, list)
    assert isinstance(output.tool_usage_log, list)
    assert 0 <= output.confidence_score <= 1
    assert len(output.detailed_analysis) > 0
    assert len(output.suggested_response_draft) > 0
