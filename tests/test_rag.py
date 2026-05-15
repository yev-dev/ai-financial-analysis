"""Tests for the RAG module."""
import pytest
from pathlib import Path
from src.fin_ai.core.rag import (
    _convert_csv_to_markdown,
    _convert_html_to_markdown,
    _convert_json_to_markdown,
    _convert_xml_to_markdown,
    _convert_word_to_markdown,
    _create_source_metadata,
    get_markdown_splits,
    filter_documents_by_metadata,
    load_and_convert_document,
)


class TestCSVConversion:
    """Test CSV conversion functionality."""
    
    @pytest.mark.unit
    def test_convert_csv_to_markdown(self, sample_csv_file):
        """Test converting CSV file to markdown format."""
        markdown = _convert_csv_to_markdown(sample_csv_file)
        
        assert "# sample_data.csv" in markdown
        assert "## Data Overview" in markdown
        assert "Rows: 3" in markdown
        assert "Columns: 3" in markdown
        assert "ticker" in markdown
        assert "AAPL" in markdown
    
    @pytest.mark.unit
    def test_convert_csv_to_markdown_includes_dtypes(self, sample_csv_file):
        """Test that CSV conversion includes column data types."""
        markdown = _convert_csv_to_markdown(sample_csv_file)
        
        assert "## Column Information" in markdown
        assert "ticker:" in markdown
        assert "price:" in markdown
    
    @pytest.mark.unit
    def test_convert_csv_nonexistent_file(self):
        """Test error handling for non-existent CSV file."""
        with pytest.raises(ValueError):
            _convert_csv_to_markdown(Path("nonexistent.csv"))


class TestHTMLConversion:
    """Test HTML conversion functionality."""
    
    @pytest.mark.unit
    def test_convert_html_to_markdown(self, sample_html_content):
        """Test converting HTML content to markdown format."""
        markdown = _convert_html_to_markdown(sample_html_content)
        
        assert "# Sample Financial Report" in markdown
        assert "Financial Analysis" in markdown
        assert "sample financial report" in markdown.lower()
    
    @pytest.mark.unit
    def test_convert_html_with_source_url(self, sample_html_content):
        """Test HTML conversion with source URL attribution."""
        url = "https://example.com/report"
        markdown = _convert_html_to_markdown(sample_html_content, source_url=url)
        
        assert "# Sample Financial Report" in markdown
        assert "**Source**:" in markdown
        assert url in markdown
    
    @pytest.mark.unit
    def test_convert_html_removes_scripts(self):
        """Test that HTML conversion removes script elements."""
        html = """
        <html>
            <head><script>alert('test');</script></head>
            <body>
                <p>Content</p>
            </body>
        </html>
        """
        markdown = _convert_html_to_markdown(html)
        
        assert "alert" not in markdown
        assert "Content" in markdown


class TestJSONConversion:
    """Test JSON conversion functionality."""
    
    @pytest.mark.unit
    def test_convert_json_to_markdown(self, sample_json_file):
        """Test converting JSON file to markdown format."""
        markdown = _convert_json_to_markdown(sample_json_file)
        
        assert "# sample_data.json" in markdown
        assert "## File Information" in markdown
        assert "## Content" in markdown
        assert "Acme Corp" in markdown
        assert "ACME" in markdown
    
    @pytest.mark.unit
    def test_convert_json_includes_nested_objects(self, sample_json_file):
        """Test that JSON conversion includes nested objects."""
        markdown = _convert_json_to_markdown(sample_json_file)
        
        assert "financials" in markdown
        assert "revenue" in markdown
        assert "1000000" in markdown
    
    @pytest.mark.unit
    def test_convert_json_includes_arrays(self, sample_json_file):
        """Test that JSON conversion includes array data."""
        markdown = _convert_json_to_markdown(sample_json_file)
        
        assert "stocks" in markdown
        assert "2026-01-01" in markdown
        assert "100.50" in markdown
    
    @pytest.mark.unit
    def test_convert_json_nonexistent_file(self):
        """Test error handling for non-existent JSON file."""
        with pytest.raises(ValueError):
            _convert_json_to_markdown(Path("nonexistent.json"))
    
    @pytest.mark.unit
    def test_convert_invalid_json_file(self, temp_dir):
        """Test error handling for invalid JSON file."""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{invalid json content")
        
        with pytest.raises(ValueError) as exc_info:
            _convert_json_to_markdown(invalid_json)
        
        assert "Invalid JSON" in str(exc_info.value)


class TestXMLConversion:
    """Test XML conversion functionality."""

    @pytest.mark.unit
    def test_convert_xml_to_markdown(self, sample_xml_file):
        """Test converting XML file to markdown format."""
        markdown = _convert_xml_to_markdown(sample_xml_file)

        assert "# sample_data.xml" in markdown
        assert "## File Information" in markdown
        assert "## Content" in markdown
        assert "report" in markdown
        assert "Acme Corp" in markdown

    @pytest.mark.unit
    def test_convert_xml_includes_nested_elements(self, sample_xml_file):
        """Test that XML conversion includes nested elements and values."""
        markdown = _convert_xml_to_markdown(sample_xml_file)

        assert "financials" in markdown
        assert "revenue" in markdown
        assert "1000000" in markdown
        assert "stock" in markdown
        assert "2026-01-01" in markdown

    @pytest.mark.unit
    def test_convert_xml_nonexistent_file(self):
        """Test error handling for non-existent XML file."""
        with pytest.raises(ValueError):
            _convert_xml_to_markdown(Path("nonexistent.xml"))

    @pytest.mark.unit
    def test_convert_invalid_xml_file(self, temp_dir):
        """Test error handling for invalid XML file."""
        invalid_xml = temp_dir / "invalid.xml"
        invalid_xml.write_text("<report><broken></report>")

        with pytest.raises(ValueError) as exc_info:
            _convert_xml_to_markdown(invalid_xml)

        assert "Invalid XML" in str(exc_info.value)


class TestWordConversion:
    """Test Word conversion functionality."""

    @pytest.mark.unit
    def test_convert_word_to_markdown(self, sample_word_file):
        """Test converting a DOCX file to markdown."""
        markdown = _convert_word_to_markdown(sample_word_file)

        assert "# sample_report.docx" in markdown
        assert "# Quarterly Financial Report" in markdown
        assert "Revenue increased year over year." in markdown
        assert "## Table 1" in markdown
        assert "| Metric | Value |" in markdown

    @pytest.mark.unit
    def test_convert_word_nonexistent_file(self):
        """Test error handling for non-existent DOCX file."""
        with pytest.raises(ValueError):
            _convert_word_to_markdown(Path("nonexistent.docx"))


class TestLoadAndConvertDocument:
    """Test the main document loading and conversion function."""
    
    @pytest.mark.unit
    def test_load_and_convert_csv(self, sample_csv_file):
        """Test loading and converting CSV document."""
        markdown = load_and_convert_document(str(sample_csv_file))
        
        assert "# sample_data.csv" in markdown
        assert "## Data Overview" in markdown
    
    @pytest.mark.unit
    def test_load_and_convert_html_file(self, sample_html_file):
        """Test loading and converting HTML file."""
        markdown = load_and_convert_document(str(sample_html_file))
        
        assert "# Sample Financial Report" in markdown
    
    @pytest.mark.unit
    def test_load_and_convert_json_file(self, sample_json_file):
        """Test loading and converting JSON file."""
        markdown = load_and_convert_document(str(sample_json_file))
        
        assert "# sample_data.json" in markdown
        assert "Acme Corp" in markdown

    @pytest.mark.unit
    def test_load_and_convert_xml_file(self, sample_xml_file):
        """Test loading and converting XML file."""
        markdown = load_and_convert_document(str(sample_xml_file))

        assert "# sample_data.xml" in markdown
        assert "report" in markdown

    @pytest.mark.unit
    def test_load_and_convert_word_file(self, sample_word_file):
        """Test loading and converting DOCX file."""
        markdown = load_and_convert_document(str(sample_word_file))

        assert "Quarterly Financial Report" in markdown
        assert "Revenue increased year over year." in markdown
    
    @pytest.mark.unit
    def test_unsupported_file_format(self, temp_dir):
        """Test error handling for unsupported file formats."""
        unsupported_file = temp_dir / "test.txt"
        unsupported_file.write_text("sample content")
        
        with pytest.raises(ValueError) as exc_info:
            load_and_convert_document(str(unsupported_file))
        
        assert "Unsupported file format" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)


class TestMetadataCreation:
    """Test metadata creation for documents."""
    
    @pytest.mark.unit
    def test_create_source_metadata_file(self, sample_csv_file):
        """Test creating metadata for a file source."""
        metadata = _create_source_metadata(sample_csv_file, "csv")
        
        assert "source" in metadata
        assert metadata["source_type"] == "csv"
        assert metadata["filename"] == "sample_data.csv"
        assert "loaded_at" in metadata
        assert "file_size" in metadata
    
    @pytest.mark.unit
    def test_create_source_metadata_url(self):
        """Test creating metadata for a URL source."""
        url = "https://example.com/report.html"
        metadata = _create_source_metadata(url, "url")
        
        assert metadata["source"] == url
        assert metadata["source_type"] == "url"
        assert metadata["filename"] == "report.html"
        assert "loaded_at" in metadata


class TestMarkdownSplitsWithMetadata:
    """Test markdown splitting with metadata attachment."""
    
    @pytest.mark.unit
    def test_get_markdown_splits_creates_documents(self):
        """Test that markdown splits are converted to Document objects."""
        markdown = "# Section 1\n\nContent 1\n\n## Subsection\n\nContent 2"
        metadata = {"source": "test.md", "source_type": "markdown"}
        
        splits = get_markdown_splits(markdown, metadata)
        
        assert len(splits) > 0
        assert all(hasattr(doc, 'metadata') for doc in splits)
        assert all(hasattr(doc, 'page_content') for doc in splits)
    
    @pytest.mark.unit
    def test_markdown_splits_include_source_metadata(self):
        """Test that splits include source metadata."""
        markdown = "# Section 1\n\nContent here"
        metadata = {"source": "report.md", "source_type": "markdown"}
        
        splits = get_markdown_splits(markdown, metadata)
        
        assert all(splits[0].metadata["source"] == "report.md" for split in splits)
        assert all(splits[0].metadata["source_type"] == "markdown" for split in splits)
    
    @pytest.mark.unit
    def test_markdown_splits_include_chunk_info(self):
        """Test that splits include chunk index and count."""
        markdown = "# Part 1\n\nText\n\n## Part 2\n\nMore text"
        
        splits = get_markdown_splits(markdown)
        
        assert len(splits) > 0
        assert all("chunk_index" in split.metadata for split in splits)
        assert all("chunk_count" in split.metadata for split in splits)


class TestMetadataFiltering:
    """Test metadata-based filtering of documents."""
    
    @pytest.mark.unit
    def test_filter_documents_by_exact_metadata(self):
        """Test filtering documents by exact metadata match."""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Content 1", metadata={"source_type": "csv", "filename": "data1.csv"}),
            Document(page_content="Content 2", metadata={"source_type": "json", "filename": "data2.json"}),
            Document(page_content="Content 3", metadata={"source_type": "csv", "filename": "data3.csv"}),
        ]
        
        filtered = filter_documents_by_metadata(docs, {"source_type": "csv"})
        
        assert len(filtered) == 2
        assert all(doc.metadata["source_type"] == "csv" for doc in filtered)
    
    @pytest.mark.unit
    def test_filter_documents_by_substring_metadata(self):
        """Test filtering documents by substring match in metadata."""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Content 1", metadata={"filename": "report_2026_01.pdf"}),
            Document(page_content="Content 2", metadata={"filename": "data_2026_02.csv"}),
            Document(page_content="Content 3", metadata={"filename": "report_2025_01.pdf"}),
        ]
        
        filtered = filter_documents_by_metadata(docs, {"filename": "2026"})
        
        assert len(filtered) == 2
        assert all("2026" in doc.metadata["filename"] for doc in filtered)
    
    @pytest.mark.unit
    def test_filter_documents_multiple_criteria(self):
        """Test filtering documents by multiple metadata criteria."""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Content 1", metadata={"source_type": "csv", "filename": "sales.csv"}),
            Document(page_content="Content 2", metadata={"source_type": "json", "filename": "sales.json"}),
            Document(page_content="Content 3", metadata={"source_type": "csv", "filename": "expenses.csv"}),
        ]
        
        filtered = filter_documents_by_metadata(docs, {"source_type": "csv", "filename": "sales"})
        
        assert len(filtered) == 1
        assert filtered[0].metadata["filename"] == "sales.csv"
    
    @pytest.mark.unit
    def test_filter_documents_no_matches(self):
        """Test filtering with no matching documents."""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Content 1", metadata={"source_type": "csv"}),
            Document(page_content="Content 2", metadata={"source_type": "json"}),
        ]
        
        filtered = filter_documents_by_metadata(docs, {"source_type": "pdf"})
        
        assert len(filtered) == 0
