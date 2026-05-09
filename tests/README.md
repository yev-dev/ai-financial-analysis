# Tests

This directory contains the test suite for the fin_ai package.

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with coverage report
```bash
pytest --cov=src/fin_ai --cov-report=html
```

### Run tests by marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests excluding slow tests
pytest -m "not slow"
```

### Run specific test file
```bash
pytest tests/test_rag.py
```

### Run specific test class
```bash
pytest tests/test_rag.py::TestCSVConversion
```

### Run specific test function
```bash
pytest tests/test_rag.py::TestCSVConversion::test_convert_csv_to_markdown
```

### Run with verbose output
```bash
pytest -v
```

## Test Structure

- `conftest.py` - Shared pytest fixtures and configuration
- `test_rag.py` - Tests for RAG (Retrieval-Augmented Generation) module

## Markers

The following pytest markers are available:

- `@pytest.mark.unit` - Unit tests for individual functions
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.asyncio` - Asynchronous tests

## Coverage

Coverage reports are generated in `htmlcov/` directory. Open `htmlcov/index.html` in a browser to view the report.
