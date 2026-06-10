import argparse
import json
import mimetypes
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape a web page for file links and download the matched files.",
    )
    parser.add_argument("url", help="Page URL to scrape.")
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory where downloaded files will be saved.",
    )
    parser.add_argument(
        "--selector",
        default="a[href]",
        help="CSS selector used to find candidate elements on the page.",
    )
    parser.add_argument(
        "--attr",
        default="href",
        help="Element attribute that contains the file URL.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional regex filter applied to absolute URLs before downloading.",
    )
    parser.add_argument(
        "--same-domain",
        action="store_true",
        help="Only keep links that point to the same domain as the page URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to download.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--save-page",
        action="store_true",
        help="Save the scraped HTML page alongside the downloaded files.",
    )
    parser.add_argument(
        "--save-context",
        action="store_true",
        help="Save extracted page context as JSON alongside the downloaded files.",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Skip link downloads and only save extracted page context artifacts.",
    )
    return parser.parse_args()


def fetch_html(url: str, timeout: int) -> str:
    response = requests.get(url, headers=_default_headers(), timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_links(
    html: str,
    page_url: str,
    selector: str,
    attr_name: str,
    pattern: str | None,
    same_domain: bool,
) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    page_domain = urlparse(page_url).netloc
    regex = re.compile(pattern) if pattern else None

    discovered: list[str] = []
    seen: set[str] = set()
    for element in soup.select(selector):
        candidate = (element.get(attr_name) or "").strip()
        if not candidate:
            continue

        absolute_url = urljoin(page_url, candidate)
        parsed = urlparse(absolute_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if same_domain and parsed.netloc != page_domain:
            continue
        if regex and not regex.search(absolute_url):
            continue
        if absolute_url in seen:
            continue

        seen.add(absolute_url)
        discovered.append(absolute_url)

    return discovered


def download_files(urls: list[str], output_dir: Path, timeout: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for index, file_url in enumerate(urls, start=1):
        try:
            with requests.get(
                file_url,
                headers=_default_headers(),
                timeout=timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                destination = build_destination(
                    output_dir,
                    file_url,
                    index,
                    content_disposition=response.headers.get("Content-Disposition"),
                    content_type=response.headers.get("Content-Type"),
                )
                print(f"[{index}/{len(urls)}] Downloading {file_url} -> {destination}")
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
        except requests.RequestException as exc:
            print(f"Failed: {file_url} ({exc})", file=sys.stderr)
            continue

        downloaded += 1

    return downloaded


def build_destination(
    output_dir: Path,
    file_url: str,
    index: int,
    content_disposition: str | None = None,
    content_type: str | None = None,
) -> Path:
    parsed = urlparse(file_url)
    name = _filename_from_content_disposition(content_disposition)
    if not name:
        name = _filename_from_url(parsed, index)

    safe_name = _sanitize_filename(name)
    if "." not in safe_name:
        extension = _extension_from_content_type(content_type)
        if extension:
            safe_name = f"{safe_name}{extension}"

    destination = output_dir / safe_name

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 2
    while True:
        candidate = output_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_page_snapshot(output_dir: Path, page_url: str, html: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(page_url)
    base_name = Path(parsed.path).stem or "index"
    destination = output_dir / f"{base_name}.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def extract_page_context(html: str, page_url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    headings = [
        heading.get_text(" ", strip=True)
        for heading in soup.select("h1, h2, h3")
        if heading.get_text(" ", strip=True)
    ]
    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in soup.select("p")
        if paragraph.get_text(" ", strip=True)
    ]
    links = extract_links(
        html=html,
        page_url=page_url,
        selector="a[href]",
        attr_name="href",
        pattern=None,
        same_domain=False,
    )

    main_content = soup.find(["main", "article", "body"]) or soup
    text_content = main_content.get_text("\n", strip=True)

    return {
        "source_url": page_url,
        "title": title,
        "headings": headings,
        "paragraphs": paragraphs,
        "links": links,
        "text": text_content,
    }


def save_page_context(output_dir: Path, page_url: str, html: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(page_url)
    base_name = Path(parsed.path).stem or parsed.netloc.lower().removeprefix("www.") or "page"
    destination = output_dir / f"{_sanitize_filename(base_name)}_context.json"
    context = extract_page_context(html, page_url)
    destination.write_text(json.dumps(context, ensure_ascii=True, indent=2), encoding="utf-8")
    return destination


def _filename_from_content_disposition(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None

    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition, re.IGNORECASE)
    if not match:
        return None

    return match.group(1).strip()


def _filename_from_url(parsed_url, index: int) -> str:
    path_parts = [part for part in parsed_url.path.split("/") if part]
    path_name = Path(parsed_url.path).name
    if path_name and Path(path_name).suffix:
        return path_name

    domain = parsed_url.netloc.lower().removeprefix("www.")
    slug_parts = [_slugify(domain)]
    slug_parts.extend(_slugify(part) for part in path_parts[-2:])
    slug_parts = [part for part in slug_parts if part]
    return "_".join(slug_parts) or f"download_{index}"


def _sanitize_filename(name: str) -> str:
    base_name = name.strip().replace(" ", "_")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._")
    return sanitized or "download"


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def _extension_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""

    mime_type = content_type.split(";", 1)[0].strip().lower()
    extension = mimetypes.guess_extension(mime_type) or ""
    if extension == ".jpe":
        return ".jpg"
    return extension


def _default_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        )
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        html = fetch_html(args.url, args.timeout)
        links: list[str] = []
        if not args.context_only:
            links = extract_links(
                html=html,
                page_url=args.url,
                selector=args.selector,
                attr_name=args.attr,
                pattern=args.pattern,
                same_domain=args.same_domain,
            )
    except requests.RequestException as exc:
        print(f"Failed to fetch page: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to process page: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None and not args.context_only:
        links = links[: args.limit]

    if args.save_page:
        snapshot = save_page_snapshot(output_dir, args.url, html)
        print(f"Saved page snapshot to {snapshot}")

    if args.save_context:
        context_path = save_page_context(output_dir, args.url, html)
        print(f"Saved page context to {context_path}")

    if args.context_only:
        return 0 if args.save_context or args.save_page else 1

    if not links:
        print("No matching links found.")
        return 0

    print(f"Found {len(links)} matching link(s).")
    downloaded = download_files(links, output_dir, args.timeout)
    print(f"Downloaded {downloaded} file(s) to {output_dir}")
    return 0 if downloaded else 1


if __name__ == "__main__":
    raise SystemExit(main())