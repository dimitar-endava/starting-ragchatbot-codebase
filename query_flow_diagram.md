# RAG System Query Flow Diagram

```mermaid
flowchart TD
    %% Frontend Layer
    A["User Types Query"] --> B["sendMessage() called"]
    B --> C["Disable UI & Show Loading"]
    C --> D["POST /api/query"]
    
    %% API Layer
    D --> E["FastAPI Endpoint /api/query"]
    E --> F{"Session ID exists?"}
    F -->|"No"| G["Create New Session"]
    F -->|"Yes"| H["Use Existing Session"]
    G --> I["rag_system.query()"]
    H --> I
    
    %% RAG System Layer
    I --> J["Get Conversation History"]
    J --> K["ai_generator.generate_response()"]
    
    %% AI Generator Layer
    K --> L["Build System Prompt + History"]
    L --> M["Call Claude API with Tools"]
    M --> N{"Claude decides to use tools?"}
    
    %% Direct Response Path
    N -->|"No"| O["Return Direct Response"]
    
    %% Tool Execution Path
    N -->|"Yes"| P["_handle_tool_execution()"]
    P --> Q["tool_manager.execute_tool()"]
    
    %% Search Tool Layer
    Q --> R["CourseSearchTool.execute()"]
    R --> S["vector_store.search()"]
    
    %% Vector Store Layer
    S --> T["ChromaDB Semantic Search"]
    T --> U["Return Relevant Chunks + Metadata"]
    U --> V["Format Results with Course/Lesson Context"]
    V --> W["Store Sources in last_sources"]
    W --> X["Return Formatted Search Results"]
    
    %% Back to AI Generator
    X --> Y["Send Tool Results to Claude"]
    Y --> Z["Claude Generates Final Response"]
    Z --> AA["Return Response Text"]
    
    %% Response Consolidation
    O --> BB["Get Sources from Tool Manager"]
    AA --> BB
    BB --> CC["Update Session History"]
    CC --> DD["Return (response, sources)"]
    
    %% API Response
    DD --> EE["FastAPI Returns JSON"]
    EE --> FF{"Request Successful?"}
    
    %% Frontend Response Handling
    FF -->|"Success"| GG["Remove Loading Animation"]
    FF -->|"Error"| HH["Show Error Message"]
    GG --> II["addMessage() with response & sources"]
    HH --> JJ["Re-enable UI"]
    II --> KK["Display Answer + Collapsible Sources"]
    KK --> LL["Re-enable UI for Next Query"]
    
    %% Styling for different layers
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef rag fill:#e8f5e8
    classDef ai fill:#fff3e0
    classDef tools fill:#fce4ec
    classDef vector fill:#e0f2f1
    classDef response fill:#f1f8e9
    
    class A,B,C,D,GG,HH,II,JJ,KK,LL frontend
    class E,F,G,H,EE,FF api
    class I,J,BB,CC,DD rag
    class K,L,M,N,O,P,Y,Z,AA ai
    class Q,R,V,W,X tools
    class S,T,U vector
    class response
```

## Key Components & Responsibilities

### **Frontend (script.js)**
- User interaction & UI management
- API communication
- Loading states & error handling

### **API Layer (app.py)** 
- FastAPI endpoints
- Session management coordination
- Request/response formatting

### **RAG System (rag_system.py)**
- Main orchestration logic
- Component coordination
- Session history management

### **AI Generator (ai_generator.py)**
- Claude API integration
- Tool execution coordination
- Response synthesis

### **Search Tools (search_tools.py)**
- Tool definitions for Claude
- Search execution logic
- Source tracking

### **Vector Store (vector_store.py)**
- ChromaDB interface
- Semantic search operations
- Course/lesson filtering

## Decision Points

1. **Session Creation**: New vs existing session
2. **Tool Usage**: Claude decides whether to search based on query type
3. **Response Type**: Direct answer vs tool-assisted response
4. **Error Handling**: Each layer has fallback mechanisms

## Data Flow

- **Request**: Query text + optional session ID
- **Tool Execution**: Search parameters → relevant content chunks
- **Response**: AI-generated answer + source citations
- **UI Update**: Display answer with expandable sources section