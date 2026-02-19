#!/usr/bin/env python3
"""
Codex Sinaiticus NT Exegesis Runner (Gemini 3 Pro Preview)

Key hardening in this version (driven by early Matthew outputs):
- Rejects "correction/substitution" phrasing in exegesis (rather than / instead of / should be / etc.)
- Rejects speculative hedging in exegesis (perhaps / likely / might / could / etc.)
- Rejects introduction of NEW proper nouns in exegesis that are not present in that verse's English Translation
  (except a small global whitelist: God/Lord/Christ/Spirit/Father/Son)

Auth:
  GEMINI_API_KEY=...
  or GOOGLE_API_KEY=... (if both set, GOOGLE_API_KEY takes precedence) :contentReference[oaicite:2]{index=2}
"""

import os
import re
import time
import csv
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

from google import genai
from google.genai import types

# =========================
# CONFIG
# =========================
BASE_DIR = r"F:\Books\Codex"
FILE_PATH = os.path.join(BASE_DIR, "xml", "Sinaiticus_NT_Processed.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "NT_Exegesis_Final")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

MODEL_NAME = "gemini-3-pro-preview"

# Pricing (optional; logging only)
IN_PRICE = 0.000002
OUT_PRICE = 0.000012

# Gemini 3: keep temperature at default 1.0 (official guidance) :contentReference[oaicite:3]{index=3}
TEMPERATURE = 1.0
FALLBACK_TEMPERATURE = 1.0

MAX_OUTPUT_TOKENS_SINGLE = 20000
MAX_OUTPUT_TOKENS_FALLBACK = 25000

THINKING_LEVEL = types.ThinkingLevel.HIGH  # supported for Gemini 3 thinking config :contentReference[oaicite:4]{index=4}

MAX_RETRIES = 8
RETRY_BACKOFF_FACTOR = 2
SLEEP_BETWEEN_CALLS = 8

# Chunking thresholds
CHAR_LENGTH_CHUNK_THRESHOLD = 14000
VERSE_COUNT_CHUNK_THRESHOLD = 24
ADAPTIVE_CHUNK_SIZES = [12, 10, 8, 6]

COMPLETION_MARKER = "— Codex Sinaiticus Exegesis | Chapter Complete —"

# Optional resume control (leave None for full canonical run)
# Example: START_FROM = ("John", 1)
START_FROM: Optional[Tuple[str, int]] = None

# =========================
# Regex + Rules
# =========================
MARKER_VARIANT_RE = re.compile(
    r"(?im)^\s*[—–-]*\s*Codex\s+Sinaiticus\s+Exegesis\s*\|\s*Chapter\s+Complete\s*[—–-]*\s*$"
)

VERSE_HEADER_RE = re.compile(
    r"^\[Book:\s*(?P<book>.*?)\]\s*\[Chapter:\s*(?P<chapter>\d+)\]\s*\[Verse:\s*(?P<verse>\d+)\]\s*$"
)

# Words commonly capitalized in English that are not proper nouns
COMMON_CAP_WORDS = {
    "The","And","But","For","Nor","Or","So","Yet","A","An",
    "This","That","These","Those","Here","There","Now","Then","Thus","Therefore",
    "If","In","On","At","To","From","Of","By","With","As","Because","When","Where","While","After","Before",
    "He","She","It","They","We","You","I","His","Her","Their","Our","Your",
    "Not","No","Yes","Let","Be","Do","Does","Did","Have","Has","Had","Will","Shall","May","Must","Can",
    "Who","Whom","Which","What","Why","How",
}

PROPER_NOUN_GLOBAL_WHITELIST = {"God","Lord","Christ","Spirit","Father","Son"}

SPECULATION_PATTERNS = [
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\blikely\b", re.IGNORECASE),
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bit\s+seems\b", re.IGNORECASE),
    re.compile(r"\bit\s+appears\b", re.IGNORECASE),
    re.compile(r"\bmight\b", re.IGNORECASE),
    re.compile(r"\bcould\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\bi\s+suspect\b", re.IGNORECASE),
]

CORRECTION_PATTERNS = [
    re.compile(r"\brather\s+than\b", re.IGNORECASE),
    re.compile(r"\bin\s+place\s+of\b", re.IGNORECASE),
    re.compile(r"\binstead\s+of\b", re.IGNORECASE),
    re.compile(r"\bshould\s+be\b", re.IGNORECASE),
    re.compile(r"\bcorrect(ed|ion)?\b", re.IGNORECASE),
]

# Broad bans: comparisons, editions, etc. (applies to whole output)
BANNED_OUTPUT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(other manuscripts?|other copies|some manuscripts?|many manuscripts?)\b", re.IGNORECASE),
    re.compile(r"\b(others\s+read|where\s+others\s+read|some\s+read)\b", re.IGNORECASE),
    re.compile(r"\b(textual\s+variant(s)?|variant\s+reading(s)?)\b", re.IGNORECASE),
    re.compile(r"\b(critical\s+edition)\b", re.IGNORECASE),
    re.compile(r"\b(NA\d+|nestle|aland)\b", re.IGNORECASE),
    re.compile(r"\b(byzantine|majority\s+text|textus\s+receptus|TR)\b", re.IGNORECASE),
    re.compile(r"\b(vulgate|peshitta|syriac|coptic)\b", re.IGNORECASE),
    re.compile(r"\b(kjv|esv|niv|nasb|nrsv)\b", re.IGNORECASE),
]

SYSTEM_INSTRUCTION = (
    "You are a late-4th-century Bishop, scholar-priest (~360 AD). Tone: pastoral, authoritative, analytically dense.\n\n"
    "NON-NEGOTIABLE RULES:\n"
    "1) SINGLE-WITNESS: Use ONLY the provided Codex Sinaiticus Greek text in the user message.\n"
    "2) NO COMPARISONS: Never reference other manuscripts, other copies, variants, versions, or what 'others read'.\n"
    "3) NO CORRECTIONS: Do not propose what the text 'should be' (no 'rather than', 'instead of', 'in place of').\n"
    "4) NO NEW NAMES: In Episcopal Exegesis, do NOT introduce any named person/place/book not present in the verse's English Translation.\n"
    "5) NO SPECULATION: Avoid hedging ('perhaps', 'likely', 'might', 'could', etc.).\n"
    "6) TRIAD PER VERSE: Produce EXACTLY one triad for EACH verse in STRICT sequential order using the verse numbers provided.\n"
    "7) NO EXTRA MATERIAL: No introductions, conclusions, headings, or commentary outside the per-verse triads.\n\n"
    "REQUIRED OUTPUT FORMAT FOR EVERY VERSE:\n"
    "**Verse X**\n"
    "**Greek Text**\n"
    "[Greek for that verse]\n\n"
    "**English Translation**\n"
    "[literal English]\n\n"
    "**Episcopal Exegesis**\n"
    "[pastoral theological reflection bound to the words given]\n\n"
    f"AFTER the final verse, append exactly this marker on its own line: {COMPLETION_MARKER}\n"
    "Begin immediately with the first verse in the provided chapter."
)

# =========================
# Utilities
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def init_metadata_csv() -> None:
    if os.path.exists(METADATA_CSV):
        return
    os.makedirs(os.path.dirname(METADATA_CSV), exist_ok=True)
    with open(METADATA_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "book", "chapter", "prompt_tokens", "output_tokens", "cost_usd", "complete", "retries_used"])

def append_metadata(book: str, ch: int, prompt_tokens: int, output_tokens: int, cost: float, complete: bool, retries_used: int) -> None:
    with open(METADATA_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([now_str(), book, ch, prompt_tokens, output_tokens, f"{cost:.6f}", int(complete), retries_used])

def usage_to_dict(usage) -> Dict[str, int]:
    if not usage:
        return {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
    return {
        "prompt_token_count": int(getattr(usage, "prompt_token_count", 0) or 0),
        "candidates_token_count": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_billable_characters": int(getattr(usage, "total_billable_characters", 0) or 0),
    }

def add_usage_dict(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_token_count": int(a.get("prompt_token_count", 0)) + int(b.get("prompt_token_count", 0)),
        "candidates_token_count": int(a.get("candidates_token_count", 0)) + int(b.get("candidates_token_count", 0)),
        "total_billable_characters": int(a.get("total_billable_characters", 0)) + int(b.get("total_billable_characters", 0)),
    }

def strip_marker_variants(text: str) -> str:
    return MARKER_VARIANT_RE.sub("", text or "").strip()

def normalize_completion_marker(text: str) -> str:
    if not text:
        return text
    cleaned = strip_marker_variants(text).strip()
    return cleaned + "\n\n" + COMPLETION_MARKER

# =========================
# Parsing
# =========================
def load_manuscript_data_nt_verses(file_path: str) -> Dict[Tuple[str, int], str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FILE_PATH not found: {file_path}")

    chapters: Dict[Tuple[str, int], List[str]] = {}
    current: Optional[Tuple[str, int, int]] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal current, buf
        if not current:
            buf = []
            return
        book, ch, v = current
        verse_text = "\n".join(buf).strip()
        buf = []
        if ch == 0 or v == 0:
            return
        if not verse_text:
            return
        chapters.setdefault((book, ch), []).append(f"[Verse: {v}]\n{verse_text}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            m = VERSE_HEADER_RE.match(raw.strip())
            if m:
                flush()
                current = (m.group("book").strip(), int(m.group("chapter")), int(m.group("verse")))
            else:
                if current is not None and raw.strip() != "":
                    buf.append(raw.strip())
        flush()

    return {k: "\n\n".join(v).strip() for k, v in chapters.items()}

def extract_verses(chapter_text: str) -> List[Tuple[int, str]]:
    verses: List[Tuple[int, str]] = []
    pattern = re.compile(r"\[Verse:\s*(\d+)\]\s*\n(.*?)(?=\n\n\[Verse:\s*\d+\]\s*\n|\Z)", re.DOTALL)
    for m in pattern.finditer(chapter_text or ""):
        verses.append((int(m.group(1)), m.group(2).strip()))
    return verses

def extract_output_verse_numbers(output_text: str) -> List[int]:
    if not output_text:
        return []
    nums = [int(n) for n in re.findall(r"\*\*Verse\s+(\d+)\*\*", output_text, flags=re.IGNORECASE)]
    if nums:
        return nums
    return [int(n) for n in re.findall(r"\[Verse:\s*(\d+)\]", output_text)]

# =========================
# Validation
# =========================
def contains_banned_language(output_text: str) -> Optional[str]:
    for pat in BANNED_OUTPUT_PATTERNS:
        m = pat.search(output_text or "")
        if m:
            return m.group(0)
    return None

def _extract_cap_tokens(text: str) -> List[str]:
    return re.findall(r"\b[A-Z][a-zA-Z][A-Za-z'\-]*\b", text or "")

def _proper_noun_set(text: str) -> set:
    toks = set(_extract_cap_tokens(text))
    return {t for t in toks if t not in COMMON_CAP_WORDS}

def _parse_verse_sections(output_text: str):
    sections = list(re.finditer(
        r"^\*\*Verse\s+(\d+)\*\*\s*(.*?)(?=^\*\*Verse\s+\d+\*\*|\Z)",
        output_text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    ))
    out = []
    for s in sections:
        vnum = int(s.group(1))
        body = s.group(2) or ""

        g = re.search(r"^\*\*Greek Text\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE)
        e = re.search(r"^\*\*English Translation\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE)
        x = re.search(r"^\*\*Episcopal Exegesis\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE)
        if not (g and e and x):
            continue

        greek = body[g.end():e.start()].strip()
        eng = body[e.end():x.start()].strip()
        exe = body[x.end():].strip()
        out.append({"vnum": vnum, "greek": greek, "eng": eng, "exe": exe})
    return out

def validate_output_structure(expected_verses: List[int], output_text: str) -> bool:
    if not output_text:
        return False

    sections = list(re.finditer(
        r"^\*\*Verse\s+(\d+)\*\*\s*(.*?)(?=^\*\*Verse\s+\d+\*\*|\Z)",
        output_text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    ))
    if not sections:
        return False

    out_nums = [int(s.group(1)) for s in sections]
    if out_nums != expected_verses:
        return False

    for s in sections:
        body = s.group(2) or ""
        if not re.search(r"^\*\*Greek Text\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE):
            return False
        if not re.search(r"^\*\*English Translation\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE):
            return False
        if not re.search(r"^\*\*Episcopal Exegesis\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE):
            return False

        g = re.search(r"^\*\*Greek Text\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE)
        e = re.search(r"^\*\*English Translation\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE)
        x = re.search(r"^\*\*Episcopal Exegesis\*\*\s*$", body, flags=re.IGNORECASE | re.MULTILINE)
        if not (g and e and x and g.start() < e.start() < x.start()):
            return False

    return True

def validate_verse_content_rules(output_text: str) -> Tuple[bool, str]:
    parsed = _parse_verse_sections(output_text)
    if not parsed:
        return False, "unable to parse verse sections"

    for sec in parsed:
        exe = sec["exe"]

        # Speculation bans only (high value, low false positives)
        for pat in SPECULATION_PATTERNS:
            if pat.search(exe):
                return False, f"speculation in exegesis (Verse {sec['vnum']})"

        # Correction-language gate disabled (too brittle; triggers on normal rhetorical contrasts)

    return True, "ok"

def validate_output(chapter_text: str, output_text: str, require_marker: bool) -> Tuple[bool, int, int, str]:
    verses = extract_verses(chapter_text)
    expected_nums = [v for v, _ in verses]
    expected_count = len(expected_nums)

    text = (output_text or "").strip()
    if not text:
        return False, expected_count, 0, "empty output"

    # Marker rules
    if require_marker:
        if not text.endswith(COMPLETION_MARKER):
            return False, expected_count, len(extract_output_verse_numbers(text)), "marker not at end"
        prefix = text[:-len(COMPLETION_MARKER)]
        if MARKER_VARIANT_RE.search(prefix):
            return False, expected_count, len(extract_output_verse_numbers(text)), "marker-like line found before end"

    banned_hit = contains_banned_language(text)
    if banned_hit:
        return False, expected_count, len(extract_output_verse_numbers(text)), f"banned language: {banned_hit}"

    out_nums = extract_output_verse_numbers(text)
    found_count = len(out_nums)
    if expected_count == 0:
        return True, 0, found_count, "ok(empty)"

    if out_nums != expected_nums:
        return False, expected_count, found_count, "verse numbers/order mismatch"

    if not validate_output_structure(expected_nums, text):
        return False, expected_count, found_count, "triad structure mismatch"

    ok2, reason2 = validate_verse_content_rules(text)
    if not ok2:
        return False, expected_count, found_count, reason2

    return True, expected_count, found_count, "ok"

# =========================
# Prompt builders
# =========================
def make_standard_user_message(book: str, ch: int, chapter_text: str) -> str:
    verses = extract_verses(chapter_text)
    first_v = verses[0][0] if verses else 1
    last_v = verses[-1][0] if verses else 1

    return (
        f"Codex Sinaiticus — {book} Chapter {ch}\n\n"
        f"Verse range present: {first_v}–{last_v}. Do NOT invent missing verses.\n\n"
        "Greek text (with verse anchors):\n"
        "'''\n" + chapter_text + "\n'''\n\n"
        "Produce EXACTLY one triad for EACH verse in the anchors, in STRICT order, without renumbering.\n"
        "Each verse must contain: Greek Text, English Translation, Episcopal Exegesis.\n"
        "Hard rules for Episcopal Exegesis:\n"
        "- Do not introduce any new named person/place/book not present in that verse's English Translation.\n"
        "- Do not speculate (no 'perhaps', 'likely', 'might', 'could', etc.).\n"
        "- Do not correct the text (no 'rather than', 'instead of', 'should be').\n"
        "- Do not reference other manuscripts, variants, or other biblical books.\n\n"
        f"After the final verse, append exactly: {COMPLETION_MARKER}\n"
        f"Begin immediately with **Verse {first_v}**. No introductory text."
    )

def make_enumerated_user_message(book: str, ch: int, verses: List[Tuple[int, str]], append_marker: bool = True) -> str:
    header = f"Codex Sinaiticus — {book} Chapter {ch} (STRICT PER-VERSE MODE)\n\n"
    body = [
        "Produce EXACTLY one triad per listed verse, in the exact order listed. Do NOT renumber verses.\n\n",
        "Hard rules for Episcopal Exegesis:\n",
        "- Do not introduce any new named person/place/book not present in the verse's English Translation.\n",
        "- Do not speculate (no 'perhaps', 'likely', 'might', 'could', etc.).\n",
        "- Do not correct the text (no 'rather than', 'instead of', 'should be').\n",
        "- Do not reference other manuscripts, variants, or other biblical books.\n\n",
        "Template for each verse:\n"
        "**Verse X**\n"
        "**Greek Text**\n"
        "[Greek]\n\n"
        "**English Translation**\n"
        "[literal English]\n\n"
        "**Episcopal Exegesis**\n"
        "[bound to the words given]\n\n"
        "Verses:\n",
    ]
    for vnum, vtxt in verses:
        body.append(f"[Verse: {vnum}]\n{vtxt}\n")

    if append_marker:
        body.append(f"\nAfter the final triad append exactly: {COMPLETION_MARKER}\n")

    if verses:
        body.append(f"Begin immediately with **Verse {verses[0][0]}**.\n")

    return header + "\n".join(body)

def split_verses_into_chunks(verses: List[Tuple[int, str]], chunk_size: int) -> List[List[Tuple[int, str]]]:
    return [verses[i:i + chunk_size] for i in range(0, len(verses), chunk_size)]

# =========================
# Gemini client / calls
# =========================
CLIENT: Optional[genai.Client] = None

def get_client() -> genai.Client:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("Missing API key env var. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return genai.Client()

def call_model_once(user_message: str, temperature: float, max_output_tokens: int):
    global CLIENT
    if CLIENT is None:
        CLIENT = get_client()

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(
            thinking_level=THINKING_LEVEL,
            include_thoughts=False,
        ),
    )

    resp = CLIENT.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(parts=[types.Part.from_text(text=user_message)])],
        config=cfg,
    )

    text_out = (getattr(resp, "text", None) or "").strip()
    usage = getattr(resp, "usage_metadata", None)
    return text_out, usage

def call_with_retries(
    book: str,
    ch: int,
    user_message: str,
    temperature: float,
    max_output_tokens: int,
    chapter_text: str,
    max_attempts: int = MAX_RETRIES,
    require_marker: bool = True,
):
    attempt = 0
    wait = 2

    while attempt < max_attempts:
        attempt += 1
        try:
            text_out, usage = call_model_once(user_message, temperature=temperature, max_output_tokens=max_output_tokens)
            ok, in_ct, out_ct, reason = validate_output(chapter_text, text_out, require_marker=require_marker)
            if not ok:
                print(f"{now_str()}  Attempt {attempt}/{max_attempts} for {book} {ch}: invalid ({reason}) (in:{in_ct} out:{out_ct}).")
                time.sleep(wait)
                wait = min(180, wait * RETRY_BACKOFF_FACTOR)
                continue
            return text_out, usage, attempt

        except Exception as e:
            s = str(e)
            print(f"{now_str()}  API error attempt {attempt}/{max_attempts} for {book} {ch}: {s}")

            if "503" in s or "UNAVAILABLE" in s or "high demand" in s.lower():
                cooldown = min(300, 60 * attempt)
                print(f"{now_str()}  503 overload detected; sleeping {cooldown}s...")
                time.sleep(cooldown)
                continue

            if "429" in s or "Too Many Requests" in s or "rate limit" in s.lower() or "RESOURCE_EXHAUSTED" in s:
                print(f"{now_str()}  Rate limit/quota detected; sleeping 90s.")
                time.sleep(90)
                continue

            time.sleep(wait)
            wait = min(180, wait * RETRY_BACKOFF_FACTOR)

    print(f"{now_str()}  All {max_attempts} attempts failed for {book} {ch}.")
    return None, None, attempt

# =========================
# Chunk generation
# =========================
def generate_with_chunking(book: str, ch: int, chapter_text: str, chunk_size: int):
    verses = extract_verses(chapter_text)
    if not verses:
        return None, None, 0

    chunks = split_verses_into_chunks(verses, chunk_size)
    stitched: List[str] = []
    agg_usage = {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
    total_retries = 0

    for chunk in chunks:
        chunk_text = "\n\n".join([f"[Verse: {v}]\n{t.strip()}" for v, t in chunk]).strip()
        user_msg = make_enumerated_user_message(book, ch, chunk, append_marker=False)

        out, usage, retries = call_with_retries(
            book, ch,
            user_msg,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS_SINGLE,
            chapter_text=chunk_text,
            require_marker=False,
        )
        total_retries += retries

        if not out:
            return None, None, total_retries

        out = strip_marker_variants(out)
        stitched.append(out)
        agg_usage = add_usage_dict(agg_usage, usage_to_dict(usage))
        time.sleep(1)

    final_text = normalize_completion_marker("\n\n".join(stitched).strip())
    ok, in_ct, out_ct, reason = validate_output(chapter_text, final_text, require_marker=True)
    if not ok:
        print(f"{now_str()}  Stitched chunk output invalid ({reason}) (in:{in_ct} out:{out_ct}).")
        return None, agg_usage, total_retries

    return final_text, agg_usage, total_retries

# =========================
# Main runner
# =========================
def print_progress(chapters_done: int, total_chapters: int, cumulative_cost: float, start_time: datetime) -> None:
    elapsed = datetime.now() - start_time
    if chapters_done <= 0:
        print(f"{now_str()}  Progress: 0/{total_chapters} | Cumulative cost: ${cumulative_cost:.2f}")
        return
    avg_per_ch = elapsed.total_seconds() / chapters_done
    remaining = total_chapters - chapters_done
    eta = datetime.now() + timedelta(seconds=avg_per_ch * remaining) if remaining > 0 else datetime.now()
    print(f"{now_str()}  Progress: {chapters_done}/{total_chapters} | ETA (rough): {eta.strftime('%Y-%m-%d %H:%M')} | Cumulative cost: ${cumulative_cost:.2f}")

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_metadata_csv()

    manuscript = load_manuscript_data_nt_verses(FILE_PATH)

    NT_BOOK_ORDER = [
        "Matthew","Mark","Luke","John","Acts","Romans",
        "1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians","Colossians",
        "1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon",
        "Hebrews","James","1 Peter","2 Peter","1 John","2 John","3 John","Jude","Revelation",
    ]
    order_index = {b: i for i, b in enumerate(NT_BOOK_ORDER)}
    keys = sorted(manuscript.keys(), key=lambda k: (order_index.get(k[0], 10**9), k[0], k[1]))

    # Optional START_FROM filter
    if START_FROM is not None:
        start_book, start_ch = START_FROM
        new_keys = []
        passed = False
        for (b, c) in keys:
            if not passed and (b == start_book and c >= start_ch):
                passed = True
            if passed:
                new_keys.append((b, c))
        keys = new_keys

    total_chapters = len(keys)
    print(f"{now_str()}  Loaded {total_chapters} chapters from: {FILE_PATH}")

    cumulative_cost = 0.0
    chapters_done = 0
    start_time = datetime.now()

    for (book, ch) in keys:
        file_name = f"{book.replace(' ', '_')}_Ch{ch}.txt"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        debug_path = save_path + ".debug.txt"

        chapter_text = manuscript.get((book, ch), "")
        verses = extract_verses(chapter_text)
        in_count = len(verses)

        if in_count == 0:
            print(f"{now_str()}  No verses detected for {book} {ch}; skipping.")
            chapters_done += 1
            print_progress(chapters_done, total_chapters, cumulative_cost, start_time)
            continue

        # Resumability: validate existing file; regenerate if invalid
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                existing = f.read()
            ok, _, _, _ = validate_output(chapter_text, existing, require_marker=True)
            if ok:
                print(f"{now_str()}  Skipping: {book} {ch} (already complete)")
                chapters_done += 1
                print_progress(chapters_done, total_chapters, cumulative_cost, start_time)
                continue
            else:
                print(f"{now_str()}  Deleting invalid: {file_name}")
                try:
                    os.remove(save_path)
                except OSError:
                    pass

        print(f"{now_str()}  Processing: {book} {ch} ({in_count} verses)")

        output_text: Optional[str] = None
        usage_info = {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
        retries_used = 0

        needs_chunking = (len(chapter_text) > CHAR_LENGTH_CHUNK_THRESHOLD) or (in_count >= VERSE_COUNT_CHUNK_THRESHOLD)

        # Chunking path (and adaptive chunk sizes)
        if needs_chunking:
            for sz in ADAPTIVE_CHUNK_SIZES:
                out, agg_usage, chunk_retries = generate_with_chunking(book, ch, chapter_text, chunk_size=sz)
                retries_used += chunk_retries
                if out:
                    output_text = out
                    usage_info = agg_usage
                    break

        # Non-chunk path: standard chapter prompt
        if not output_text and not needs_chunking:
            user_msg = make_standard_user_message(book, ch, chapter_text)
            out, usage, attempts = call_with_retries(
                book, ch,
                user_msg,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS_SINGLE,
                chapter_text=chapter_text,
                require_marker=True,
            )
            retries_used += attempts
            if out:
                output_text = out
                usage_info = usage_to_dict(usage)

        # Non-chunk path: strict enumerated fallback
        if not output_text and not needs_chunking:
            enum_msg = make_enumerated_user_message(book, ch, verses, append_marker=True)
            out, usage, attempts = call_with_retries(
                book, ch,
                enum_msg,
                temperature=FALLBACK_TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS_FALLBACK,
                chapter_text=chapter_text,
                require_marker=True,
            )
            retries_used += attempts
            if out:
                output_text = out
                usage_info = usage_to_dict(usage)

        # Hard failure: write debug + placeholder (so next run retries)
        if not output_text:
            dbg = [
                f"Book: {book} Chapter: {ch}",
                f"Input verse count: {in_count}",
                "=== INPUT (first 2000 chars) ===",
                chapter_text[:2000],
            ]
            safe_write(debug_path, "\n\n".join(dbg))
            output_text = f"API ERROR: generation failed to produce valid full coverage for {book} {ch}. See {os.path.basename(debug_path)}"
            usage_info = {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}

        output_text = normalize_completion_marker(output_text)
        safe_write(save_path, output_text)

        prompt_tokens = int(usage_info.get("prompt_token_count", 0))
        out_tokens = int(usage_info.get("candidates_token_count", 0))
        cost = (prompt_tokens * IN_PRICE) + (out_tokens * OUT_PRICE)
        cumulative_cost += cost
        chapters_done += 1

        complete_flag, _, _, _ = validate_output(chapter_text, output_text, require_marker=True)
        append_metadata(book, ch, prompt_tokens, out_tokens, cost, complete_flag, retries_used)

        print(f"{now_str()}  Stored: {file_name} | Retries: {retries_used}")
        print_progress(chapters_done, total_chapters, cumulative_cost, start_time)
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"{now_str()}  Run complete.")
    print(f"{now_str()}  Outputs: {OUTPUT_DIR}")
    print(f"{now_str()}  Metadata: {METADATA_CSV}")

if __name__ == "__main__":
    run()
