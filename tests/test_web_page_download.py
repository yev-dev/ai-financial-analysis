"""Tests for the web page download script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "web_page_download.py"
    spec = importlib.util.spec_from_file_location("web_page_download", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


web_page_download = _load_module()


class TestWebPageDownload:
    @pytest.mark.unit
    def test_main_scrapes_google_page_and_downloads_matching_links(self, monkeypatch, temp_dir, capsys):
        google_html = """
        <html>
            <body>
                <a href="/finance/report.pdf">Investor report</a>
                <a href="https://www.google.com/finance/data.csv">Finance data</a>
                <a href="https://support.google.com/help">Support</a>
            </body>
        </html>
        """

        args = type(
            "Args",
            (),
            {
                "url": "https://www.google.com",
                "output_dir": str(temp_dir),
                "selector": "a[href]",
                "attr": "href",
                "pattern": r"\.(pdf|csv)$",
                "same_domain": True,
                "limit": None,
                "timeout": 10,
                "save_page": False,
                "save_context": False,
                "context_only": False,
            },
        )()

        downloaded_urls: list[str] = []

        monkeypatch.setattr(web_page_download, "parse_args", lambda: args)
        monkeypatch.setattr(web_page_download, "fetch_html", lambda url, timeout: google_html)

        def fake_download_files(urls: list[str], output_dir: Path, timeout: int) -> int:
            downloaded_urls.extend(urls)
            assert output_dir == temp_dir
            assert timeout == 10
            return len(urls)

        monkeypatch.setattr(web_page_download, "download_files", fake_download_files)

        result = web_page_download.main()
        captured = capsys.readouterr()

        assert result == 0
        assert downloaded_urls == [
            "https://www.google.com/finance/report.pdf",
            "https://www.google.com/finance/data.csv",
        ]
        assert "Found 2 matching link(s)." in captured.out
        assert f"Downloaded 2 file(s) to {temp_dir}" in captured.out

    @pytest.mark.unit
    def test_build_destination_uses_content_disposition_filename(self, temp_dir):
        destination = web_page_download.build_destination(
            temp_dir,
            "https://www.google.com/download?id=123",
            1,
            content_disposition='attachment; filename="Alphabet Q1 2026 Results.pdf"',
            content_type="application/pdf",
        )

        assert destination.name == "Alphabet_Q1_2026_Results.pdf"

    @pytest.mark.unit
    def test_build_destination_uses_domain_and_path_when_url_has_no_filename(self, temp_dir):
        destination = web_page_download.build_destination(
            temp_dir,
            "https://www.google.com/finance/quotes/",
            1,
            content_type="text/html; charset=utf-8",
        )

        assert destination.name == "google_com_finance_quotes.html"

    @pytest.mark.unit
    def test_save_page_context_writes_extracted_content(self, temp_dir):
        html = """
        <html>
            <head><title>Yahoo Finance</title></head>
            <body>
                <main>
                    <h1>Markets Overview</h1>
                    <p>Stocks rose after earnings surprises.</p>
                    <a href="/finance">Finance</a>
                </main>
            </body>
        </html>
        """

        destination = web_page_download.save_page_context(temp_dir, "https://uk.yahoo.com", html)
        payload = json.loads(destination.read_text(encoding="utf-8"))

        assert destination.name == "uk.yahoo.com_context.json"
        assert payload["source_url"] == "https://uk.yahoo.com"
        assert payload["title"] == "Yahoo Finance"
        assert payload["headings"] == ["Markets Overview"]
        assert payload["paragraphs"] == ["Stocks rose after earnings surprises."]
        assert payload["links"] == ["https://uk.yahoo.com/finance"]
        assert "Markets Overview" in payload["text"]

    @pytest.mark.unit
    def test_main_can_save_context_without_matching_download_links(self, monkeypatch, temp_dir, capsys):
        html = """
        <html>
            <head><title>Context Page</title></head>
            <body>
                <article>
                    <h1>Daily Brief</h1>
                    <p>Context should still be saved.</p>
                </article>
            </body>
        </html>
        """

        args = type(
            "Args",
            (),
            {
                "url": "https://example.com",
                "output_dir": str(temp_dir),
                "selector": "a[href]",
                "attr": "href",
                "pattern": r"\.pdf$",
                "same_domain": True,
                "limit": None,
                "timeout": 10,
                "save_page": False,
                "save_context": True,
                "context_only": False,
            },
        )()

        monkeypatch.setattr(web_page_download, "parse_args", lambda: args)
        monkeypatch.setattr(web_page_download, "fetch_html", lambda url, timeout: html)

        result = web_page_download.main()
        captured = capsys.readouterr()

        assert result == 0
        assert "Saved page context to" in captured.out
        assert (temp_dir / "example.com_context.json").exists()
        assert "No matching links found." in captured.out

    @pytest.mark.unit
    def test_main_context_only_saves_context_without_downloading(self, monkeypatch, temp_dir, capsys):
        html = """
        <html>
            <head><title>Yahoo</title></head>
            <body>
                <main>
                    <h1>Briefing</h1>
                    <a href="/finance/report.pdf">Report</a>
                </main>
            </body>
        </html>
        """

        args = type(
            "Args",
            (),
            {
                "url": "https://uk.yahoo.com",
                "output_dir": str(temp_dir),
                "selector": "a[href]",
                "attr": "href",
                "pattern": r"\.(pdf|csv)$",
                "same_domain": True,
                "limit": 5,
                "timeout": 10,
                "save_page": False,
                "save_context": True,
                "context_only": True,
            },
        )()

        called = False

        monkeypatch.setattr(web_page_download, "parse_args", lambda: args)
        monkeypatch.setattr(web_page_download, "fetch_html", lambda url, timeout: html)

        def fail_download(*args, **kwargs):
            raise AssertionError("download_files should not be called in context-only mode")

        monkeypatch.setattr(web_page_download, "download_files", fail_download)

        result = web_page_download.main()
        captured = capsys.readouterr()

        assert result == 0
        assert called is False
        assert (temp_dir / "uk.yahoo.com_context.json").exists()
        assert "Saved page context to" in captured.out
        assert "Found" not in captured.out