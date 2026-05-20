import os
import re
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")

PAGES = [
    ("Offside (association football)", "offside.txt",
     "https://en.wikipedia.org/wiki/Offside_(association_football)"),
    ("Pressing (association football)", "pressing.txt",
     "https://en.wikipedia.org/wiki/Pressing_(association_football)"),
    ("Association football positions", "association_football_positions.txt",
     "https://en.wikipedia.org/wiki/Association_football_positions"),
    ("Formation (association football)", "formation.txt",
     "https://en.wikipedia.org/wiki/Formation_(association_football)"),
    ("Total Football", "total_football.txt",
     "https://en.wikipedia.org/wiki/Total_Football"),
    ("Tiki-taka", "tiki_taka.txt",
     "https://en.wikipedia.org/wiki/Tiki-taka"),
    ("Catenaccio", "catenaccio.txt",
     "https://en.wikipedia.org/wiki/Catenaccio"),
    ("FIFA World Cup", "fifa_world_cup.txt",
     "https://en.wikipedia.org/wiki/FIFA_World_Cup"),
    ("2014 FIFA World Cup", "2014_fifa_world_cup.txt",
     "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup"),
    ("2022 FIFA World Cup", "2022_fifa_world_cup.txt",
     "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup"),
    ("UEFA Champions League", "uefa_champions_league.txt",
     "https://en.wikipedia.org/wiki/UEFA_Champions_League"),
    ("Lionel Messi", "lionel_messi.txt",
     "https://en.wikipedia.org/wiki/Lionel_Messi"),
    ("Diego Maradona", "diego_maradona.txt",
     "https://en.wikipedia.org/wiki/Diego_Maradona"),
    ("Gerd Müller", "gerd_muller.txt",
     "https://en.wikipedia.org/wiki/Gerd_M%C3%BCller"),
    ("Leicester City 2015-16", "leicester_city_2015_16.txt",
     "https://en.wikipedia.org/wiki/2015%E2%80%9316_Leicester_City_F.C._season"),
    ("Arsenal 2003-04 season", "arsenal_2003_04.txt",
     "https://en.wikipedia.org/wiki/2003%E2%80%9304_Arsenal_F.C._season"),
    ("A.C. Milan", "ac_milan.txt",
     "https://en.wikipedia.org/wiki/A.C._Milan"),
    ("Pep Guardiola", "pep_guardiola.txt",
     "https://en.wikipedia.org/wiki/Pep_Guardiola"),
    ("Jürgen Klopp", "jurgen_klopp.txt",
     "https://en.wikipedia.org/wiki/J%C3%BCrgen_Klopp"),
    ("Ballon d'Or", "ballon_dor.txt",
     "https://en.wikipedia.org/wiki/Ballon_d%27Or"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; knowledge-base-builder/1.0)"}

EXCLUDED_SECTIONS = {
    "references", "external links", "further reading", "see also",
    "notes", "bibliography", "sources",
}


def extract_text(soup):
    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return ""

    # Remove unwanted tags wholesale
    for tag in content_div.find_all(["table", "sup", "span.reference",
                                     "div.reflist", "div.navbox",
                                     "div.sidebar", "div.infobox",
                                     "ol.references"]):
        tag.decompose()

    paragraphs = []
    skip_section = False

    for element in content_div.find_all(["h2", "h3", "h4", "p"]):
        if element.name in ("h2", "h3", "h4"):
            heading = element.get_text(" ", strip=True).lower()
            # Strip edit-section markers like "[edit]"
            heading = re.sub(r"\[.*?\]", "", heading).strip()
            skip_section = heading in EXCLUDED_SECTIONS
            continue

        if skip_section:
            continue

        text = element.get_text(" ", strip=True)
        # Drop citation markers like [1], [2], [nb 1]
        text = re.sub(r"\[[^\]]{0,10}\]", "", text).strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, filename, url in PAGES:
        print(f"Scraping: {label}...")
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  ERROR fetching {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        text = extract_text(soup)

        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  Saved {len(text):,} chars -> {filename}")

    print("\nDone.")


if __name__ == "__main__":
    main()