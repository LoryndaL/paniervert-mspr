import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://agriculture.ec.europa.eu/farming/organic-farming/organic-logo_fr"
ROBOTS_URL = "https://agriculture.ec.europa.eu/robots.txt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "agriculture_ec"
HTML_PATH = OUTPUT_DIR / "organic_logo_page.html"
JSON_PATH = OUTPUT_DIR / "organic_logo_data.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ProjetDataScraping/1.0; +https://example.com)"}


def check_robots_txt():
    robots_response = requests.get(ROBOTS_URL, headers=HEADERS, timeout=30)
    robots_response.raise_for_status()
    robots_text = robots_response.text
    robot_lines = [line.strip() for line in robots_text.splitlines() if line.strip() and not line.startswith('#')]
    disallow_entries = []
    for i, line in enumerate(robot_lines):
        if line.lower().startswith('user-agent:'):
            continue
        if line.lower().startswith('disallow:'):
            value = line.split(':', 1)[1].strip()
            if value:
                disallow_entries.append(value)

    target_path = "/farming/organic-farming/organic-logo_fr"
    is_blocked = any(target_path.startswith(path) for path in disallow_entries)

    if is_blocked:
        print("robots.txt bloque explicitement la page cible ; il faut vérifier la conformité avant scraping.")
    else:
        print("robots.txt n'interdit pas explicitement la page cible : scraping autorisé selon les règles robots.")

    return robots_text


def fetch_page(url: str):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def clean_text(value):
    return " ".join(value.split())


def extract_sections(soup: BeautifulSoup):
    sections = []
    all_headings = soup.select("main h2, main h3")

    for heading in all_headings:
        title = clean_text(heading.get_text(" ", strip=True))
        if not title:
            continue
        content_parts = []
        current = heading.find_next_sibling()
        while current is not None and current.name not in {"h2", "h3"}:
            text = current.get_text(" ", strip=True)
            if text:
                content_parts.append(text)
            current = current.find_next_sibling()
        sections.append({
            "heading": title,
            "text": " ".join(content_parts),
        })

    return sections


def scrape_page():
    html = fetch_page(BASE_URL)
    soup = BeautifulSoup(html, "html.parser")

    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    h1 = clean_text(soup.select_one("main h1").get_text(" ", strip=True)) if soup.select_one("main h1") else ""

    paragraphs = [
        clean_text(p.get_text(" ", strip=True))
        for p in soup.select("main p")
        if clean_text(p.get_text(" ", strip=True))
    ]

    links = [
        {
            "text": clean_text(a.get_text(" ", strip=True)),
            "href": a.get("href", ""),
        }
        for a in soup.select("main a[href]")
        if clean_text(a.get_text(" ", strip=True))
    ][:30]

    data = {
        "source_url": BASE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "h1": h1,
        "paragraphs": paragraphs,
        "links": links,
        "sections": extract_sections(soup),
    }

    return html, data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_robots_txt()
    html, data = scrape_page()

    HTML_PATH.write_text(html, encoding="utf-8")
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Page HTML enregistrée : {HTML_PATH}")
    print(f"Données structurées enregistrées : {JSON_PATH}")
    print(f"Titre : {data['title']}")
    print(f"Nombre de paragraphes : {len(data['paragraphs'])}")
    print(f"Nombre de liens : {len(data['links'])}")


if __name__ == "__main__":
    main()

