# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the application:**
```bash
./run.sh
```

**Manual startup:**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Code Quality Tools:**
```bash
./scripts/format.sh    # Format code with Black and organize imports with isort
./scripts/lint.sh      # Run all quality checks (Black, isort, flake8)
uv run black backend/  # Format code with Black only
uv run isort backend/  # Organize imports with isort only
uv run flake8 backend/ # Run flake8 linting only
```

**Environment setup:**
- Copy `.env.example` to `.env`
- Set `ANTHROPIC_API_KEY` in `.env`

**Access points:**
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials. The architecture follows a layered approach with tool-based AI orchestration.

### Core Components

**RAG System (`rag_system.py`)** - Main orchestrator that coordinates all components:
- Manages conversation sessions via `SessionManager`
- Orchestrates AI generation with tool usage
- Handles document ingestion and course analytics

**AI Generator (`ai_generator.py`)** - Claude API integration with tool calling:
- Uses Anthropic's Claude Sonnet 4 model
- Implements tool-based search decision making
- Manages conversation context and tool execution flow

**Search Tools (`search_tools.py`)** - Tool interface for Claude:
- `CourseSearchTool` implements the Tool abstract base class  
- Provides semantic search with course/lesson filtering
- Tracks source citations for UI display

**Vector Store (`vector_store.py`)** - ChromaDB interface:
- Manages two collections: `course_metadata` and `course_content`
- Uses SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- Supports smart course name matching and lesson filtering

**Document Processor (`document_processor.py`)** - Text chunking and preprocessing:
- Chunks course documents (800 chars, 100 overlap)
- Extracts course/lesson structure from text
- Creates `Course` and `CourseChunk` objects

### Data Flow Architecture

**Query Processing:**
1. Frontend sends query to `/api/query` endpoint
2. RAG system retrieves conversation history
3. AI Generator calls Claude with tool definitions
4. If Claude decides to search: CourseSearchTool → VectorStore → ChromaDB
5. Tool results returned to Claude for response synthesis
6. Final answer with sources returned to frontend

**Key Decision Point:** Claude autonomously decides whether to use search tools based on query type (course-specific vs general knowledge).

### Configuration

All settings centralized in `config.py`:
- Model: `claude-sonnet-4-20250514`
- Embeddings: `all-MiniLM-L6-v2`
- Chunk size: 800 chars with 100 char overlap
- Max search results: 5
- Conversation history: 2 messages

### Frontend Integration

**Vanilla JavaScript frontend** (`frontend/`):
- Single-page application with no frameworks
- WebSocket-style loading animations
- Collapsible source citations
- Session management for conversation continuity

### Data Storage

**ChromaDB collections:**
- `course_metadata` - Course titles and descriptions for smart matching
- `course_content` - Chunked content with lesson metadata

**Course documents** stored in `docs/` folder (4 course scripts pre-loaded).

## Key Implementation Notes

- **Tool-based AI**: Claude makes autonomous decisions about when to search vs answer directly
- **Session persistence**: Conversation history maintained across queries  
- **Smart search**: Course name matching supports partial matches (e.g., "MCP" finds "MCP Introduction")
- **Source tracking**: Search tools track citations through the tool execution chain
- **Error handling**: Graceful degradation at each layer with user-friendly messages
- **Code Quality**: This codebase uses Black for formatting, isort for import organization, and flake8 for linting
- **Development workflow**: Always run `./scripts/lint.sh` before committing to ensure code quality
- always use uv do never use pip directly
- make sure to use uv for managing all the dependencies
- use uv to run all python files