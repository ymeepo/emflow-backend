# EM Tools Backend

A FastAPI backend providing state management and data storage APIs for the EM Tools enterprise management dashboard.

## Quick Start

1. Install dependencies with uv (recommended):
```bash
uv sync
```

2. Start the development server:
```bash
uv run python main.py
```

The API will run on `http://localhost:8001`

**Alternative with pip (requires manual dependency management):**
```bash
pip install fastapi uvicorn pydantic neo4j sentence-transformers torch
python main.py
```

## Project Structure

```
backend/
├── main.py               # FastAPI application entry point
├── database.py           # Neo4j connection and management
├── embedding_service.py  # Qwen embedding model integration
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # UV dependency lock file with exact versions
└── .env.example         # Environment configuration template
```

## API Endpoints

### Health Check
- `GET /` - API status and information

### State Management
- `GET /state` - Get all state items
- `GET /state/{key}` - Get a specific state item by key
- `POST /state` - Add or update a state item
- `PUT /state/{key}` - Update a specific state item
- `DELETE /state/{key}` - Delete a specific state item
- `DELETE /state` - Clear all state items

### Hiring Funnel Data
- `GET /funnel` - Get saved funnel data
- `POST /funnel` - Save hiring funnel data
- `DELETE /funnel` - Clear all funnel data

## Data Models

### StateItem
```python
{
    "key": "string",
    "value": "any"
}
```

### StateValue
```python
{
    "value": "any"
}
```

### FunnelData
```python
{
    "applicants": "integer",
    "hm_review": "integer", 
    "hm_interview": "integer",
    "tech_screen": "integer",
    "panel_1": "integer",
    "panel_2": "integer",
    "hired": "integer"
}
```

## Features

### State Storage
- **In-memory storage** using Python dictionaries
- **RESTful API** with full CRUD operations
- **Type validation** with Pydantic models
- **Flexible data types** supporting any JSON-serializable value

### CORS Support
- **Cross-origin requests** enabled for frontend development
- **Configurable origins** for different environments
- **Full HTTP method support** (GET, POST, PUT, DELETE)

### API Documentation
- **Automatic OpenAPI documentation** at `/docs`
- **ReDoc documentation** at `/redoc`
- **JSON schema** for all endpoints and models

## Technology Stack

- **FastAPI** - Modern, fast Python web framework
- **Pydantic** - Data validation using Python type annotations
- **Uvicorn** - Lightning-fast ASGI server
- **Python 3.9+** - Modern Python features and type hints

## Development

### Dependencies
Key packages and their purposes:
- `fastapi` - Web framework and API
- `uvicorn` - ASGI server for development and production
- `pydantic` - Data validation and serialization

**Adding new dependencies:**
```bash
# Add runtime dependencies
uv add package-name

# Add development dependencies
uv add --dev pytest black

# Add optional dependencies
uv add --optional ai openai langchain
```

### Data Storage
Currently uses **in-memory storage** with Python dictionaries:
- `state_store: Dict[str, Any]` - General key-value storage
- `funnel_data: Dict[str, int]` - Hiring funnel specific data

**Important**: Data is lost when the server restarts.

### CORS Configuration
Configured for frontend development:
```python
allow_origins=["http://localhost:3000", "http://localhost:3001"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

## API Usage Examples

### State Management
```bash
# Add new state
curl -X POST "http://localhost:8001/state" \
  -H "Content-Type: application/json" \
  -d '{"key": "user_preference", "value": {"theme": "dark"}}'

# Get state
curl "http://localhost:8001/state/user_preference"

# Update state
curl -X PUT "http://localhost:8001/state/user_preference" \
  -H "Content-Type: application/json" \
  -d '{"value": {"theme": "light"}}'

# Delete state
curl -X DELETE "http://localhost:8001/state/user_preference"
```

### Funnel Data
```bash
# Save funnel data
curl -X POST "http://localhost:8001/funnel" \
  -H "Content-Type: application/json" \
  -d '{
    "applicants": 100,
    "hm_review": 80,
    "hm_interview": 60,
    "tech_screen": 40,
    "panel_1": 25,
    "panel_2": 15,
    "hired": 10
  }'

# Get funnel data
curl "http://localhost:8001/funnel"
```

## Production Considerations

### Database Migration
For production use, replace in-memory storage with:

**Recommended Options:**
- **PostgreSQL + pgvector** - For structured data with semantic search
- **MongoDB** - For document-based storage
- **Redis** - For high-performance caching and session storage

### Security
Add these features for production:
- Authentication and authorization (JWT tokens)
- Input validation and sanitization
- Rate limiting and request throttling
- HTTPS/TLS encryption
- Environment-based configuration

### Performance
Consider these optimizations:
- Database connection pooling
- Async database operations
- Response caching strategies
- API versioning
- Request/response compression

## Environment Configuration

Create a `.env` file for configuration:
```
# Server Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=True

# Database (for future use)
DATABASE_URL=postgresql://user:password@localhost/emtools

# CORS Configuration
FRONTEND_URL=http://localhost:3000

# API Keys (for future AI integration)
OPENAI_API_KEY=your_api_key_here
```

## Docker Support

### Dockerfile Example
```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

COPY . .
EXPOSE 8001

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8001:8001"
    environment:
      - DEBUG=False
```

## Testing

### Manual Testing
- Visit `http://localhost:8001/docs` for interactive API documentation
- Use curl commands or Postman for endpoint testing
- Check `http://localhost:8001/redoc` for alternative documentation

### Future Test Implementation
```python
# pytest example structure
def test_create_state():
    response = client.post("/state", json={"key": "test", "value": "data"})
    assert response.status_code == 200
    
def test_get_state():
    response = client.get("/state/test")
    assert response.status_code == 200
```

## Future Enhancements

- [ ] Database integration (PostgreSQL recommended)
- [ ] User authentication and authorization
- [ ] Real AI/LLM API integration
- [ ] Data validation and business logic
- [ ] Background task processing
- [ ] Logging and monitoring
- [ ] Unit and integration tests
- [ ] API versioning strategy
- [ ] Advanced error handling
- [ ] Performance optimization

## Contributing

1. Follow PEP 8 Python style guidelines
2. Use type hints for all function parameters and returns
3. Add docstrings for all public functions and classes
4. Validate input data with Pydantic models
5. Handle errors gracefully with appropriate HTTP status codes