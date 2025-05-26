from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class Message(BaseModel):
    sender: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ConversationContext(BaseModel):
    conversation_history: List[Message]
    current_prospect_message: Message
    prospect_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_history": [
                    {
                        "sender": "prospect",
                        "content": "Hi, I'm looking for a sales tool that can help our team be more efficient. Can you tell me about your product?",
                        "timestamp": "2023-05-17T10:00:00",
                    },
                    {
                        "sender": "agent",
                        "content": "Hi there! Our AI-powered sales platform helps teams close more deals with less work. It includes automated follow-ups, smart lead prioritization, and AI-assisted email drafting. Would you like to know more about any specific features?",
                        "timestamp": "2023-05-17T10:01:00",
                    },
                ],
                "current_prospect_message": {
                    "sender": "prospect",
                    "content": "Are your services available in my location?",
                    "timestamp": "2025-05-25T18:27:19.371Z",
                },
                "prospect_id": "PROSPECT001",
            }
        },
    )


class AnalysisResult(BaseModel):
    intent: str
    entities: List[str]
    sentiment: float
    key_points: List[str]


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    FLAG_FOR_REVIEW = "flag_for_reviews"


class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolResult(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    result: Union[Dict[str, Any], List[Dict[str, Any]]]


class ActionableOutput(BaseModel):
    detailed_analysis: str
    suggested_response_draft: str
    internal_next_steps: List[Dict[str, Any]]
    tool_usage_log: List[ToolResult]
    confidence_score: float
    reasoning_trace: Optional[str] = None


class DocumentChunk(BaseModel):
    content: str
    source: str
    chunk_id: str
    embedding: Optional[List[float]] = None


class ToolUsageLog(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_name: str
    parameters: Dict[str, Any]
    status: Optional[ToolStatus] = None
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    error_message: Optional[str] = None
    reasoning_trace: Optional[Union[str, List[str]]] = None
    clarification_needed: Optional[bool] = None
    clarification_reason: Optional[str] = None

    model_config = {
        "json_schema_extra": {"json_encoders": {datetime: lambda v: v.isoformat()}}
    }


class ProcessMessageResponse(BaseModel):
    output: ActionableOutput
    processing_time: float
    status: str = "success"
