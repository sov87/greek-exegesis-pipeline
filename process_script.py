#!/usr/bin/env python3
import re
import xml.etree.ElementTree as ET

XML_PATH = r"F:\Books\Codex\xml\codex_sinaiticus.xml"
OUT_PATH = r"F:\Books\Codex\xml\Sinaiticus_NT_Processed.txt"

def local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag

# EXACT titles from your list -> your English keys
BOOK_MAP = {
    "Κατὰ Μαθθαῖον": "Matthew",
    "Κατὰ Μάρκον": "Mark",
    "Κατὰ Λουκᾶν": "Luke",
    "Κατὰ Ἰωάννην": "John",
    "Πράξειϲ Ἀποϲτόλων": "Acts",

    "Προϲ Ρωμαίουϲ": "Romans",
    "Προϲ Κορινθίουϲ Α": "1 Corinthians",
    "Προϲ Κορινθίουϲ Β": "2 Corinthians",
    "Πρὸϲ Γαλατάϲ": "Galatians",
    "Πρὸϲ Εφεϲίουϲ": "Ephesians",
    "Πρὸϲ Φιλιππηϲίουϲ": "Philippians",
    "Πρὸϲ Κολοϲϲαεῖϲ": "Colossians",
    "Πρὸϲ Θεϲϲαλονικεῖϲ Α'": "1 Thessalonians",
    "Πρὸϲ Θεϲϲαλονικεῖϲ Β'": "2 Thessalonians",
    "Πρὸϲ Ἑβραίουϲ": "Hebrews",
    "Πρὸϲ Τιμόθεον Α'": "1 Timothy",
    "Πρὸϲ Τιμόθεον Β'": "2 Timothy",
    "Πρὸϲ Τίτον": "Titus",
    "Πρὸϲ Φιλήμονα": "Philemon",

    "Ἰακώβου": "James",
    "Πέτρου α´": "1 Peter",
    "Πέτρου β´": "2 Peter",
    "Ἰωάννου α": "1 John",
    "Ἰωάννου β": "2 John",
    "Ἰωάννου γ": "3 John",
    "Ἰούδα": "Jude",
    "Ἀποκάλυψιϲ Ἰωάννου": "Revelation",
}

# allow quick normalization of whitespace only (do NOT alter accents/sigmas)
def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def main():
    context = ET.iterparse(XML_PATH, events=("start", "end"))

    current_book_raw = None
    current_book = None
    current_chapter = None
    current_verse = None
    in_ab = False
    verse_words = []

    books_written = set()
    chapters_written = 0
    verses_written = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for event, elem in context:
            tag = local_name(elem.tag)

            # book div
            if tag == "div" and elem.get("type") == "book":
                if event == "start":
                    current_book_raw = norm_title(elem.get("title"))
                    current_book = BOOK_MAP.get(current_book_raw)
                elif event == "end":
                    current_book_raw = None
                    current_book = None
                continue

            # chapter div (only if this is an NT book)
            if current_book and tag == "div" and elem.get("type") == "chapter":
                if event == "start":
                    current_chapter = (elem.get("n") or "0").strip()
                    chapters_written += 1
                elif event == "end":
                    current_chapter = None
                continue

            # verse ab (only if in NT book+chapter)
            if current_book and current_chapter is not None and tag == "ab":
                if event == "start":
                    current_verse = (elem.get("n") or "").strip()
                    in_ab = True
                    verse_words = []
                elif event == "end":
                    in_ab = False
                    text = " ".join(w for w in verse_words if w).strip()
                    if current_verse and text:
                        out.write(f"[Book: {current_book}] [Chapter: {current_chapter}] [Verse: {current_verse}]\n")
                        out.write(text + "\n\n")
                        verses_written += 1
                        books_written.add(current_book)

                    current_verse = None
                    verse_words = []
                    elem.clear()
                continue

            # words inside verse
            if in_ab and current_book and tag == "w" and event == "end":
                if elem.text:
                    t = re.sub(r"\s+", " ", elem.text).strip()
                    if t:
                        verse_words.append(t)
                elem.clear()
                continue

            if event == "end":
                elem.clear()

    print("Done.")
    print("NT books written:", len(books_written), sorted(books_written))
    print("NT chapters written:", chapters_written)
    print("NT verses written:", verses_written)
    print("Output file:", OUT_PATH)

if __name__ == "__main__":
    main()
