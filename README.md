# AI Sales Agent

An intelligent sales agent that processes prospect messages, analyzes them in context, and generates appropriate responses using advanced LLM orchestration.

## Features

- 🤖 AI-powered conversation handling
- 📚 Knowledge base integration
- 📊 Performance metrics tracking
- 🔄 Automated follow-ups
- 🎯 Intent recognition and analysis
- 📝 Response generation
- 🔍 Tool usage logging
- 📈 Performance monitoring

## Project Structure
```
.
├── app/
│   ├── api/            # FastAPI routes
│   ├── core/           # Core functionality
│   ├── models/         # Pydantic models
│   ├── services/       # Business logic
│   └── tools/          # Tool implementations
├── data/               # Data files
├── scripts/            # Utility scripts
├── tests/              # Test files
└── docs/               # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-sales-assistant.git
cd ai-sales-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
```
OPENAI_API_KEY=your_api_key
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.api.main:app --reload
```

2. The API will be available at `http://localhost:8000`

Request body:
```json
{
    "conversation_history": [
        {
            "sender": "prospect",
            "content": "Hi, I'm interested in your product.",
            "timestamp": "2024-02-20T10:00:00Z"
        }
    ],
    "current_prospect_message": {
        "sender": "prospect",
        "content": "What's the pricing?",
        "timestamp": "2024-02-20T10:01:00Z"
    },
    "prospect_id": "optional_prospect_id"
}
```

Response:
```json
{
    "output": {
        "detailed_analysis": "string",
        "suggested_response_draft": "string",
        "internal_next_steps": [
            {
                "action": "string",
                "details": {}
            }
        ],
        "tool_usage_log": [
            {
                "tool_name": "string",
                "parameters": {},
                "result": {}
            }
        ],
        "confidence_score": 0.95,
        "reasoning_trace": "string"
    },
    "processing_time": 1.23,
    "status": "success"
}
```

- `POST /process_message`: Process a new customer message
- `GET /metrics`: Get performance metrics
- `GET /metrics/historical`: Get historical metrics
- `GET /docs`: API documentation (Swagger UI)

Health check endpoint.

## Testing

Run the test suite:
```bash
pytest tests/
```

## Evaluation Framework

The project includes a comprehensive evaluation framework:

1. **Offline Evaluation**
   - Golden dataset of conversation examples
   - Automated metrics for response quality
   - Prompt engineering testing rig

2. **Online Monitoring**
   - Key performance indicators
   - LLM performance scoring
   - Feedback loop integration

## Architecture

The system follows clean architecture principles:

1. **API Layer**
   - FastAPI endpoints
   - Request/response handling
   - Input validation

2. **Service Layer**
   - LLM orchestration
   - Business logic
   - Error handling

3. **Tool Layer**
   - Knowledge base integration
   - CRM data retrieval
   - External service integration

4. **Data Layer**
   - Pydantic models
   - Data validation
   - Type safety

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 