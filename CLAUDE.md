# CLAUDE.md - Repository Guidelines

## Build Commands
- Install dependencies: `uv sync`
- Run the API: `uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- Run the simple API: `uv run python -m uvicorn simpleAPI:app --host 0.0.0.0 --port 8000`
- Run web client: `cd client && python -m http.server 8080`

## Code Style
- Follow PEP 8 for Python code
- Use type hints with optional parameters annotated as `Optional[Type]`
- Model classes use Pydantic for validation
- Use async/await for I/O operations
- Handle exceptions with try/except blocks
- Use descriptive variable names
- Document functions with docstrings
- Cache model instances to improve performance

## Architecture
- Use FastAPI for API endpoints
- Implement image generation with mflux library
- Track power usage with macmon when available
- Queue system for handling concurrent requests
- Store usage metrics in usage.yo file