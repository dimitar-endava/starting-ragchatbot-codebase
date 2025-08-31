#!/bin/bash

# Format code quality script for the RAG chatbot project
# This script runs all code formatting tools

echo "🧹 Running code formatting tools..."

echo "📝 Formatting code with Black..."
uv run black backend/ .

echo "📦 Organizing imports with isort..."
uv run isort backend/ .

echo "✅ Code formatting complete!"