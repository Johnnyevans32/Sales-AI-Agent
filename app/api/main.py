import os
import time
import traceback
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..models.base import (
    ActionableOutput,
    ConversationContext,
    ProcessMessageResponse,
    ToolStatus,
    ToolUsageLog,
)
from ..services.llm_service import LLMService
from ..services.metrics_service import MetricsService
from ..tools.knowledge_tool import KnowledgeAugmentationTool

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Sales Agent API",
    description="An intelligent sales agent that processes prospect messages and generates appropriate responses",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService(api_key=os.getenv("OPENAI_API_KEY"))
knowledge_tool = KnowledgeAugmentationTool()
metrics_service = MetricsService()


@app.post("/process_message", response_model=ProcessMessageResponse)
async def process_message(context: ConversationContext):
    """
    Process a new prospect message and generate an appropriate response.

    Args:
        context: The conversation context including history and current message

    Returns:
        ProcessMessageResponse: The generated response and actions
    """
    start_time = time.time()

    try:
        # Step 1: Analyze the message
        analysis = await llm_service.analyze_message(context)

        # Step 2: Decide which tools to use
        tool_calls = await llm_service.decide_tool_usage(
            context=context, analysis=analysis, tools=knowledge_tool.tools
        )

        # Step 3: Check if we need clarification
        if tool_calls and tool_calls[0].tool_name == "clarification_needed":
            log = ToolUsageLog(
                tool_name="clarification_needed",
                parameters=tool_calls[0].parameters,
                latency_ms=(time.time() - start_time) * 1000,
                reasoning_trace=tool_calls[0].parameters.get(
                    "reasoning_trace", "Unknown reason"
                ),
                clarification_needed=True,
                clarification_reason=tool_calls[0].parameters.get(
                    "reasoning_trace", "Unknown reason"
                ),
            )
            metrics_service.log_tool_usage(log)
            return ProcessMessageResponse(
                output=ActionableOutput(**tool_calls[0].parameters),
                processing_time=time.time() - start_time,
                status="needs_clarification",
            )

        # Step 4: Process tool calls and collect results
        tool_results = await knowledge_tool.process_tool_calls(tool_calls)

        # Step 5: Generate final response
        output = await llm_service.generate_response(
            context=context, analysis=analysis, tool_results=tool_results
        )

        processing_time = time.time() - start_time
        log = ToolUsageLog(
            tool_name="generate_response",
            parameters={
                "context": context.model_dump(),
                "analysis": analysis.model_dump(),
                "tool_results": [result.model_dump() for result in tool_results],
            },
            status=(
                ToolStatus.FLAG_FOR_REVIEW
                if any(
                    step["action"] == "FLAG_FOR_HUMAN_REVIEW"
                    for step in output.internal_next_steps
                )
                else None
            ),
            latency_ms=processing_time * 1000,
            confidence_score=output.confidence_score,
            reasoning_trace=output.reasoning_trace,
        )
        metrics_service.log_tool_usage(log)

        return ProcessMessageResponse(
            output=output, processing_time=processing_time, status="success"
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get performance metrics for the knowledge tool."""
    try:
        metrics = metrics_service.get_performance_metrics()
        return {"status": "success", "data": metrics}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error getting performance metrics: {str(e)}"
        )


@app.get("/metrics/historical", response_model=Dict[str, Any])
async def get_historical_metrics(days: int = 7):
    """Get historical metrics for the specified number of days."""
    try:
        metrics = metrics_service.get_historical_metrics(days)
        return {"status": "success", "data": metrics}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error getting historical metrics: {str(e)}"
        )
