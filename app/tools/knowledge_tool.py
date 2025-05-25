import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.models.base import (
    DocumentChunk,
    ToolCall,
    ToolResult,
    ToolStatus,
    ToolUsageLog,
)
from app.services.metrics_service import MetricsService

# Load environment variables
load_dotenv()


class KnowledgeAugmentationTool:
    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY")
        self.voyage_base_url = "https://api.voyageai.com"
        self.metrics_service = MetricsService()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_prospect_details",
                    "description": "Get prospect information from CRM including company details, interactions, and preferences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prospect_id": {
                                "type": "string",
                                "description": "The ID of the prospect to fetch details for",
                            },
                            "fields": {
                                "type": "array",
                                "description": "Optional list of specific fields to fetch. If not provided, returns all fields.",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "name",
                                        "company_name",
                                        "industry",
                                        "company_size",
                                        "annual_revenue",
                                        "location",
                                        "lead_score",
                                        "stage",
                                        "last_contact",
                                        "previous_interactions",
                                        "known_technologies",
                                        "pain_points",
                                        "budget_range",
                                        "decision_makers",
                                        "competitors",
                                        "custom_fields",
                                    ],
                                },
                            },
                        },
                        "required": ["prospect_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_base",
                    "description": "Search product documentation and sales playbooks for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information",
                            },
                            "filters": {
                                "type": "object",
                                "description": "Optional filters to narrow down search results",
                                "properties": {
                                    "entities": {
                                        "type": "array",
                                        "description": "List of entities to filter results by",
                                        "items": {"type": "string"},
                                    },
                                    "min_similarity": {
                                        "type": "number",
                                        "description": "Minimum similarity score (0.0 to 1.0) for results",
                                    },
                                    "max_results": {
                                        "type": "integer",
                                        "description": "Maximum number of results to return",
                                    },
                                    "sources": {
                                        "type": "array",
                                        "description": "Filter results by specific document sources",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

        self._load_knowledge_base()
        self._load_mock_crm_data()

    def _chunk_document(self, content: str, source: str) -> List[DocumentChunk]:
        """Split a document into chunks using LangChain's RecursiveCharacterTextSplitter."""
        chunks = []
        text_chunks = self.text_splitter.split_text(content)

        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{source}_{i}"
            chunks.append(
                DocumentChunk(content=chunk, source=source, chunk_id=chunk_id)
            )

        return chunks

    def _load_mock_crm_data(self):
        """Load mock CRM data for prospects."""
        self.mock_crm_data = {
            "PROSPECT001": {
                "prospect_id": "PROSPECT001",
                "name": "Alice Doe",
                "company_name": "TechCorp Solutions",
                "industry": "Technology",
                "company_size": "100-500",
                "annual_revenue": "$10M-$50M",
                "location": "San Francisco, CA",
                "lead_score": 0.85,
                "stage": "Qualified",
                "last_contact": "2024-03-15",
                "previous_interactions": [
                    {
                        "date": "2024-03-15",
                        "type": "email",
                        "summary": "Initial outreach - interested in AI solutions",
                        "sentiment": 0.8,
                    },
                    {
                        "date": "2024-03-10",
                        "type": "call",
                        "summary": "Product demo requested",
                        "sentiment": 0.9,
                    },
                ],
                "known_technologies": ["AWS", "Python", "Docker", "Kubernetes"],
                "pain_points": [
                    "Scaling AI infrastructure",
                    "Data processing bottlenecks",
                    "Team productivity",
                ],
                "budget_range": "$50K-$100K",
                "decision_makers": [
                    {
                        "name": "John Smith",
                        "title": "CTO",
                        "contact": "john.smith@techcorp.com",
                    },
                    {
                        "name": "Sarah Johnson",
                        "title": "VP of Engineering",
                        "contact": "sarah.j@techcorp.com",
                    },
                ],
                "competitors": ["CompetitorA", "CompetitorB"],
                "custom_fields": {
                    "ai_maturity": "Intermediate",
                    "cloud_migration": "In Progress",
                    "security_requirements": "High",
                },
            },
            "PROSPECT002": {
                "prospect_id": "PROSPECT002",
                "name": "Bob Smith",
                "company_name": "Global Finance Inc",
                "industry": "Financial Services",
                "company_size": "1000-5000",
                "annual_revenue": "$100M-$500M",
                "location": "New York, NY",
                "lead_score": 0.65,
                "stage": "Discovery",
                "last_contact": "2024-03-12",
                "previous_interactions": [
                    {
                        "date": "2024-03-12",
                        "type": "meeting",
                        "summary": "Initial discovery call - exploring automation solutions",
                        "sentiment": 0.7,
                    }
                ],
                "known_technologies": ["Java", "Oracle", "IBM Mainframe"],
                "pain_points": [
                    "Legacy system modernization",
                    "Compliance automation",
                    "Customer service efficiency",
                ],
                "budget_range": "$100K-$500K",
                "decision_makers": [
                    {
                        "name": "Michael Brown",
                        "title": "CIO",
                        "contact": "m.brown@globalfinance.com",
                    }
                ],
                "competitors": ["CompetitorC", "CompetitorD"],
                "custom_fields": {
                    "compliance_level": "Strict",
                    "digital_transformation": "Planned",
                    "risk_tolerance": "Low",
                },
            },
        }

    def fetch_prospect_details(
        self, prospect_id: str, fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch prospect details from the mock CRM.

        Args:
            prospect_id: The ID of the prospect to fetch
            fields: Optional list of specific fields to return. If None, returns all fields.

        Returns:
            Dictionary containing the requested prospect details
        """
        if prospect_id not in self.mock_crm_data:
            logger.warning(f"Prospect {prospect_id} not found in CRM")
            return {
                "prospect_id": prospect_id,
                "error": "Prospect not found",
                "available_prospects": list(self.mock_crm_data.keys()),
            }

        prospect_data = self.mock_crm_data[prospect_id]

        if fields:
            # Filter the data to only include requested fields
            filtered_data = {}
            for field in fields:
                if field in prospect_data:
                    filtered_data[field] = prospect_data[field]
            return filtered_data

        return prospect_data

    def _load_knowledge_base(self):
        """Load and index the knowledge base documents."""
        self.chunks: List[DocumentChunk] = []

        # Load all text files from the knowledge base directory
        for file_path in self.knowledge_base_path.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Split document into chunks
                document_chunks = self._chunk_document(content, file_path.name)
                self.chunks.extend(document_chunks)

        # Create embeddings for all chunks
        if self.chunks:
            texts = [chunk.content for chunk in self.chunks]
            embeddings = self._get_embeddings(texts)
            # Assign embeddings to chunks
            for chunk, embedding in zip(self.chunks, embeddings):
                chunk.embedding = embedding

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Voyage AI API."""
        config = {
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.voyage_api_key}",
            },
        }

        try:
            response = requests.post(
                f"{self.voyage_base_url}/v1/embeddings",
                json={"input": texts, "model": "voyage-3.5", "input_type": "document"},
                headers=config["headers"],
                timeout=30,
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            logger.error(f"Error getting embeddings from Voyage AI: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query."""
        config = {
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.voyage_api_key}",
            },
        }

        try:
            response = requests.post(
                f"{self.voyage_base_url}/v1/embeddings",
                json={"input": query, "model": "voyage-3.5", "input_type": "query"},
                headers=config["headers"],
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting query embedding from Voyage AI: {str(e)}")
            raise

    def query_knowledge_base(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query the knowledge base using semantic search."""
        start_time = time.time()
        reasoning_trace = []

        try:
            if not self.chunks:
                reasoning_trace.append("No documents in knowledge base")
                return []

            # Encode the query
            query_embedding = self._get_query_embedding(query)
            reasoning_trace.append("Generated query embedding")

            # Calculate similarity scores for all chunks
            similarities = []
            for chunk in self.chunks:
                if chunk.embedding is None:
                    continue
                similarity = np.dot(chunk.embedding, query_embedding) / (
                    np.linalg.norm(chunk.embedding) * np.linalg.norm(query_embedding)
                )
                similarities.append(similarity)

            reasoning_trace.append(
                f"Calculated similarities for {len(similarities)} chunks"
            )

            # Apply filters
            min_similarity = filters.get("min_similarity", 0.5) if filters else 0.5
            max_results = filters.get("max_results", 3) if filters else 3
            entities = filters.get("entities", []) if filters else []
            sources = filters.get("sources", []) if filters else []

            reasoning_trace.append(
                f"Applied filters: min_similarity={min_similarity}, max_results={max_results}"
            )

            # Get top results based on max_results
            top_indices = np.argsort(similarities)[-max_results:][::-1]

            results = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                if similarities[idx] > min_similarity:
                    if sources and chunk.source not in sources:
                        continue

                    result = {
                        "content": chunk.content,
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "similarity_score": float(similarities[idx]),
                    }

                    if entities:
                        content_lower = result["content"].lower()
                        if any(entity.lower() in content_lower for entity in entities):
                            results.append(result)
                    else:
                        results.append(result)

            reasoning_trace.append(f"Found {len(results)} matching results")

            # Log the tool usage
            latency_ms = (time.time() - start_time) * 1000
            log = ToolUsageLog(
                tool_name="query_knowledge_base",
                parameters={"query": query, "filters": filters},
                latency_ms=latency_ms,
                reasoning_trace=reasoning_trace,
            )
            self.metrics_service.log_tool_usage(log)

            return results

        except Exception as e:
            logger.error(f"Error in query_knowledge_base: {str(e)}")
            reasoning_trace.append(f"Error: {str(e)}")

            # Log the error
            latency_ms = (time.time() - start_time) * 1000
            log = ToolUsageLog(
                tool_name="query_knowledge_base",
                parameters={"query": query, "filters": filters},
                latency_ms=latency_ms,
                error_message=str(e),
                reasoning_trace=reasoning_trace,
            )
            self.metrics_service.log_tool_usage(log)

            raise

    async def process_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Process a list of tool calls and return their results."""
        tool_results = []
        start_time = time.time()
        reasoning_trace = []

        for tool_call in tool_calls:
            try:
                result = self.process_tool_call(
                    tool_call.tool_name, tool_call.parameters
                )

                # Log the tool usage with LLM analysis
                latency_ms = (time.time() - start_time) * 1000
                log = ToolUsageLog(
                    tool_name=tool_call.tool_name,
                    parameters=tool_call.parameters,
                    status=ToolStatus.SUCCESS,
                    latency_ms=latency_ms,
                    reasoning_trace=reasoning_trace,
                )
                self.metrics_service.log_tool_usage(log)

                tool_results.append(
                    ToolResult(
                        tool_name=tool_call.tool_name,
                        parameters=tool_call.parameters,
                        result=result,
                    )
                )
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.tool_name}: {str(e)}")
                reasoning_trace.append(f"Error in {tool_call.tool_name}: {str(e)}")

                # Log the error with LLM analysis
                latency_ms = (time.time() - start_time) * 1000
                log = ToolUsageLog(
                    tool_name=tool_call.tool_name,
                    parameters=tool_call.parameters,
                    status=ToolStatus.ERROR,
                    latency_ms=latency_ms,
                    error_message=str(e),
                    reasoning_trace=reasoning_trace,
                )
                self.metrics_service.log_tool_usage(log)

                tool_results.append(
                    ToolResult(
                        tool_name=tool_call.tool_name,
                        parameters=tool_call.parameters,
                        result={"error": str(e)},
                    )
                )
        return tool_results

    def process_tool_call(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process a tool call and return the results."""
        try:
            if tool_name == "fetch_prospect_details":
                fields = parameters.get("fields")
                return self.fetch_prospect_details(
                    parameters.get("prospect_id"), fields=fields
                )
            elif tool_name == "query_knowledge_base":
                return self.query_knowledge_base(
                    parameters.get("query", ""), parameters.get("filters")
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error in tool call {tool_name}: {str(e)}")
            raise
