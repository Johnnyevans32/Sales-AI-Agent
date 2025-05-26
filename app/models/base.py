from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    sender: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ConversationContext(BaseModel):
    conversation_history: List[Message]
    current_prospect_message: Message
    prospect_id: Optional[str] = None


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
