#!/bin/bash

# Lint and quality check script for the RAG chatbot project
# This script runs all code quality checks

echo "🔍 Running code quality checks..."

echo "🎯 Checking code formatting with Black..."
uv run black --check backend/ .
BLACK_EXIT=$?

echo "📦 Checking import organization with isort..."
uv run isort --check-only backend/ .
ISORT_EXIT=$?

echo "🔧 Running flake8 linting..."
uv run flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503
FLAKE8_EXIT=$?

# Check if any tool failed
if [ $BLACK_EXIT -ne 0 ] || [ $ISORT_EXIT -ne 0 ] || [ $FLAKE8_EXIT -ne 0 ]; then
    echo "❌ Code quality checks failed!"
    echo "💡 Run './scripts/format.sh' to auto-fix formatting issues"
    exit 1
else
    echo "✅ All code quality checks passed!"
fi