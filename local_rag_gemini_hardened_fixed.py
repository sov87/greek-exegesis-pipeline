#!/usr/bin/env python3
"""
Codex Sinaiticus NT Exegesis Runner (Gemini 3 Pro Preview) — production run

Input format expected (your new file):
[Book: <English Book>] [Chapter: <N>] [Verse: <V>]
<Greek text...>

Key guarantees:
- Groups per-verse source into full chapters
- Skips Chapter 0 / Verse 0 superscriptions automatically
- Strict validation: output must contain EXACTLY one triad per input verse
- Retries on bad output (missing marker OR verse-count mismatch), not just exceptions
- Chunking for long chapters using enumerated per-chunk prompts
- Atomic writes + resumable run + metadata CSV + debug files on failure

BEFORE RUN:
1) pip install google-genai
2) Set your key in an env var:
   - PowerShell:  $env:GOOGLE_API_KEY="YOUR_KEY"
   - CMD:        set GOOGLE_API_KEY=YOUR_KEY
3) Verify FILE_PATH points to your new processed NT file.
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
# CONFIG (your directory)
# =========================
BASE_DIR = r"F:\Books\Codex"

# Use the NEW processed file you generated from XML:
FILE_PATH = os.path.join(BASE_DIR, "xml", "Sinaiticus_NT_Processed.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "NT_Exegesis_Final")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

MODEL_NAME = "gemini-3-pro-preview"

# Pricing (adjust if provider prices change)
IN_PRICE = 0.000002
OUT_PRICE = 0.000012

# Generation tuning
TEMPERATURE = 0.55
FALLBACK_TEMPERATURE = 0.40
MAX_OUTPUT_TOKENS_SINGLE = 20000
MAX_OUTPUT_TOKENS_FALLBACK = 25000
THINKING_LEVEL = types.ThinkingLevel.HIGH

# Operational
MAX_RETRIES = 8
RETRY_BACKOFF_FACTOR = 2
SLEEP_BETWEEN_CALLS = 8
CHAR_LENGTH_CHUNK_THRESHOLD = 14000
CHUNK_VERSE_COUNT = 12


# Also chunk by verse-count to avoid output-token truncation on short-but-verbose chapters
VERSE_COUNT_CHUNK_THRESHOLD = 24
COMPLETION_MARKER = "— Codex Sinaiticus Exegesis | Chapter Complete —"

SYSTEM_INSTRUCTION = (
    "You are a 4th-century Bishop, scholar-priest (~360 AD). Tone: pastoral, authoritative, analytically dense.\n\n"
    "STANDARDS:\n"
    "1) ZERO-INFERENCE: Base every statement ONLY on the provided Codex Sinaiticus Greek text. "
    "Never import modern readings or standardized Bible text.\n"
    "2) TRIAD PER VERSE: Produce EXACTLY one triad for EACH verse in STRICT sequential order starting from verse 1.\n"
    "3) EXHAUSTIVE COVERAGE: If the chapter has N verses, you MUST produce exactly N triads. "
    "Do not skip, summarize, merge, or reorder.\n"
    "4) OUTPUT FORMAT for EVERY verse:\n"
    "**Verse X**\n"
    "**Greek Text**\n"
    "[verbatim uncial with nomina sacra]\n\n"
    "**English Translation**\n"
    "[literal English]\n\n"
    "**Episcopal Exegesis**\n"
    "[deep pastoral analysis]\n\n"
    f"5) After the final verse, append exactly: '{COMPLETION_MARKER}'\n"
    "Begin immediately with **Verse 1**. No introductory paragraphs."
)

# ==================================
# Helpers / parsing / validation
# ==================================

VERSE_HEADER_RE = re.compile(
    r"^\[Book:\s*(?P<book>.*?)\]\s*\[Chapter:\s*(?P<chapter>\d+)\]\s*\[Verse:\s*(?P<verse>\d+)\]\s*$"
)

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

def append_metadata(book: str, ch: int, prompt_tokens: int, out_tokens: int, cost: float, complete: bool, retries_used: int) -> None:
    with open(METADATA_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([now_str(), book, ch, prompt_tokens, out_tokens, f"{cost:.6f}", int(complete), retries_used])

def usage_to_dict(usage) -> dict:
    if usage is None:
        return {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
    return {
        "prompt_token_count": int(getattr(usage, "prompt_token_count", 0) or 0),
        "candidates_token_count": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_billable_characters": int(getattr(usage, "total_billable_characters", 0) or 0),
    }

def add_usage_dict(a: Optional[dict], b: Optional[dict]) -> dict:
    a = a or {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
    b = b or {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
    return {
        "prompt_token_count": int(a.get("prompt_token_count", 0)) + int(b.get("prompt_token_count", 0)),
        "candidates_token_count": int(a.get("candidates_token_count", 0)) + int(b.get("candidates_token_count", 0)),
        "total_billable_characters": int(a.get("total_billable_characters", 0)) + int(b.get("total_billable_characters", 0)),
    }

def load_manuscript_data_nt_verses(file_path: str) -> Dict[Tuple[str, int], str]:
    """
    Reads your per-verse file and groups into (book, chapter) -> full chapter text
    while preserving the [Verse: N] anchors for extraction.

    Skips:
    - Chapter 0 / Verse 0 superscriptions
    - Any verse header with missing content lines (rare)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FILE_PATH not found: {file_path}")

    chapters: Dict[Tuple[str, int], List[str]] = {}
    current = None  # (book, chapter, verse)
    buf: List[str] = []

    def flush():
        nonlocal current, buf
        if not current:
            buf = []
            return
        book, ch, v = current
        text = "\n".join(buf).strip()
        buf = []
        # skip superscriptions
        if ch == 0 or v == 0:
            return
        if not text:
            return
        key = (book, ch)
        if key not in chapters:
            chapters[key] = []
        chapters[key].append(f"[Verse: {v}]\n{text}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            m = VERSE_HEADER_RE.match(raw.strip())
            if m:
                flush()
                book = m.group("book").strip()
                ch = int(m.group("chapter"))
                v = int(m.group("verse"))
                current = (book, ch, v)
            else:
                if current is not None:
                    if raw.strip() != "":
                        buf.append(raw.strip())
        flush()

    # Join chapter verses with blank lines
    out: Dict[Tuple[str, int], str] = {}
    for key, verse_blocks in chapters.items():
        out[key] = "\n\n".join(verse_blocks).strip()

    return out

def extract_verses(chapter_text: str) -> List[Tuple[int, str]]:
    """
    Extracts [(verse_number, verse_text)] from chapter_text.
    Expected anchors: [Verse: N] followed by verse text until next [Verse:] or end.
    """
    verses = []
    pattern = re.compile(r"\[Verse:\s*(\d+)\]\s*\n(.*?)(?=\n\n\[Verse:\s*\d+\]\s*\n|\Z)", re.DOTALL)
    for m in pattern.finditer(chapter_text):
        vnum = int(m.group(1))
        vtxt = m.group(2).strip()
        if vtxt:
            verses.append((vnum, vtxt))
    # fallback (if formatting is slightly different)
    if not verses:
        pattern2 = re.compile(r"\[Verse:\s*(\d+)\]\s*(.*?)(?=\[Verse:\s*\d+\]|\Z)", re.DOTALL)
        for m in pattern2.finditer(chapter_text):
            vnum = int(m.group(1))
            vtxt = re.sub(r"\s+", " ", m.group(2)).strip()
            if vtxt:
                verses.append((vnum, vtxt))
    return verses


def extract_output_verse_numbers(output_text: str) -> List[int]:
    """Best-effort parse of verse numbers from the model output."""
    if not output_text:
        return []
    # Prefer explicit verse headings.
    nums = [int(n) for n in re.findall(r"\*\*Verse\s+(\d+)\*\*", output_text, flags=re.IGNORECASE)]
    if nums:
        return nums
    # Fallback to preserved anchors.
    nums = [int(n) for n in re.findall(r"\[Verse:\s*(\d+)\]", output_text)]
    return nums

def normalize_completion_marker(text: str) -> str:
    """Ensure the completion marker appears exactly once, at the end."""
    if not text:
        return text
    if COMPLETION_MARKER not in text:
        return text
    # Remove all occurrences and re-append once.
    parts = [p.strip() for p in text.split(COMPLETION_MARKER) if p.strip()]
    cleaned = "\n\n".join(parts).strip()
    return cleaned + "\n\n" + COMPLETION_MARKER

def validate_output_coverage_strict(chapter_text: str, output_text: str) -> Tuple[bool, int, int]:
    """
    Strict coverage validation:
    - Verse-count must match
    - Verse numbers (and order) must match the input anchors
    """
    expected = [v for v, _ in extract_verses(chapter_text)]
    in_count = len(expected)

    out_nums = extract_output_verse_numbers(output_text or "")
    out_count = len(out_nums) if out_nums else count_output_greek_blocks(output_text or "")

    if in_count == 0:
        # For empty inputs, just require marker.
        return (COMPLETION_MARKER in (output_text or "")), 0, out_count

    if out_nums:
        return (out_nums == expected), in_count, out_count

    # If parsing fails, fall back to count-only.
    return (out_count == in_count), in_count, out_count

def count_output_greek_blocks(output_text: str) -> int:
    c1 = len(re.findall(r"\*\*Verse \d+\*\*", output_text, flags=re.IGNORECASE))
    c2 = len(re.findall(r"\[Verse:\s*\d+\]", output_text))
    c3 = len(re.findall(r"^\*\*Greek Text\*\*$", output_text, flags=re.IGNORECASE | re.MULTILINE))
    return max(c1, c2, c3)

def validate_output_coverage(chapter_text: str, output_text: str) -> Tuple[bool, int, int]:
    return validate_output_coverage_strict(chapter_text, output_text)
def split_verses_into_chunks(verses: List[Tuple[int, str]], chunk_size: int) -> List[List[Tuple[int, str]]]:
    return [verses[i:i + chunk_size] for i in range(0, len(verses), chunk_size)]

def make_standard_user_message(book: str, ch: int, chapter_text: str) -> str:
    return (
        f"Codex Sinaiticus — {book} Chapter {ch}\n\n"
        "Greek text (full chapter):\n"
        "'''\n" + chapter_text + "\n'''\n\n"
        "Process every verse in strict sequential order from verse 1. Produce the full triad for each using this exact format:\n"
        "**Verse X**\n"
        "**Greek Text**\n"
        "[verbatim uncial with nomina sacra]\n\n"
        "**English Translation**\n"
        "[literal English]\n\n"
        "**Episcopal Exegesis**\n"
        "[deep pastoral analysis]\n\n"
        f"Produce exactly one triad per verse. After the last verse append exactly: {COMPLETION_MARKER}\n"
        "Begin immediately with **Verse 1**. No introductory text."
    )

def make_enumerated_user_message(book: str, ch: int, verses: List[Tuple[int, str]]) -> str:
    header = f"Codex Sinaiticus — {book} Chapter {ch} (STRICT PER-VERSE MODE)\n\n"
    body = [
        "Produce EXACTLY one triad per listed verse, in the exact order listed. Do NOT renumber verses.\n",
        "Each triad MUST follow this exact template:\n",
        "**Verse X**\n",
        "**Greek Text**\n",
        "[verbatim uncial with nomina sacra]\n\n",
        "**English Translation**\n",
        "[literal English]\n\n",
        "**Episcopal Exegesis**\n",
        "[deep pastoral analysis]\n\n"
    ]
    for vnum, vtxt in verses:
        body.append(f"**Verse {vnum}**")
        body.append(f"[Verse: {vnum}]\n{vtxt}\n")
    body.append(f"\nAfter the final triad append exactly: {COMPLETION_MARKER}\n")
    return header + "\n".join(body)

# =========================
# Gemini call / retry logic
# =========================

def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key or len(api_key) < 20:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY env var.\n"
            "PowerShell:  $env:GOOGLE_API_KEY=\"YOUR_KEY\"\n"
            "CMD:        set GOOGLE_API_KEY=YOUR_KEY"
        )
    return genai.Client(api_key=api_key)

CLIENT = None  # init lazily

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
            include_thoughts=False  # keep outputs clean for the final document
        )
    )

    resp = CLIENT.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=cfg,
    )
    text_out = getattr(resp, "text", None) or ""
    usage = getattr(resp, "usage_metadata", None)
    return text_out, usage

def call_with_retries(book: str, ch: int, user_message: str, temperature: float, max_output_tokens: int,
                      chapter_text: str, max_attempts: int = MAX_RETRIES, require_marker: bool = True):
    attempt = 0
    wait = 2
    while attempt < max_attempts:
        attempt += 1
        try:
            text_out, usage = call_model_once(user_message, temperature=temperature, max_output_tokens=max_output_tokens)

            if require_marker and (not text_out or (COMPLETION_MARKER not in text_out)):
                print(f"{now_str()}  ⚠️ Attempt {attempt}/{max_attempts} for {book} {ch}: missing completion marker.")
                time.sleep(wait)
                wait *= RETRY_BACKOFF_FACTOR
                continue

            ok, in_ct, out_ct = validate_output_coverage(chapter_text, text_out)
            if not ok:
                print(f"{now_str()}  ⚠️ Attempt {attempt}/{max_attempts} for {book} {ch}: coverage mismatch (in:{in_ct} out:{out_ct}).")
                time.sleep(wait)
                wait *= RETRY_BACKOFF_FACTOR
                continue

            return text_out, usage, attempt

        except Exception as e:
            s = str(e)
            print(f"{now_str()}  ❌ API error attempt {attempt}/{max_attempts} for {book} {ch}: {s}")

            # 503 overload (high demand) — back off hard
            if "503" in s or "UNAVAILABLE" in s or "high demand" in s.lower():
                cooldown = min(300, 60 * attempt)  # 60s → 120s → 180s... cap at 5 min
                print(f"{now_str()}  ⏳ 503 overload detected — sleeping {cooldown}s before retry...")
                time.sleep(cooldown)
                continue

            # 429 rate limit
            if "429" in s or "Too Many Requests" in s or "rate limit" in s.lower():
                print(f"{now_str()}  ℹ️ Rate limit detected; sleeping 90s.")
                time.sleep(90)
                continue

            # other errors — normal exponential backoff
            time.sleep(wait)
            wait *= RETRY_BACKOFF_FACTOR

    # all attempts failed
    print(f"{now_str()}  ❌ All {max_attempts} attempts failed for {book} {ch}.")
    return None, None, attempt

def generate_with_chunking(book: str, ch: int, chapter_text: str):
    verses = extract_verses(chapter_text)
    if not verses:
        return None, None, 0

    chunks = split_verses_into_chunks(verses, CHUNK_VERSE_COUNT)
    stitched = []
    agg_usage = {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
    total_retries = 0

    for idx, chunk in enumerate(chunks, 1):
        # Build chunk_text with verse anchors so validation is consistent
        chunk_text = "\n\n".join([f"[Verse: {v}]\n{t}" for v, t in chunk]).strip()

        user_msg = make_enumerated_user_message(book, ch, chunk)
        out, usage, retries = call_with_retries(
            book, ch,
            user_msg,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS_SINGLE,
            chapter_text=chunk_text,
            require_marker=False
        )
        total_retries += retries

        if not out:
            print(f"{now_str()}  ⚠️ Chunk {idx}/{len(chunks)} failed. Aborting chunking.")
            return None, None, total_retries

        valid, in_count, out_count = validate_output_coverage(chunk_text, out)
        if not valid:
            print(f"{now_str()}  ⚠️ Chunk {idx}/{len(chunks)} coverage mismatch (in:{in_count} out:{out_count}). Aborting chunking.")
            return None, None, total_retries

        out = out.replace(COMPLETION_MARKER, "").strip()
        stitched.append(out)
        agg_usage = add_usage_dict(agg_usage, usage_to_dict(usage))
        time.sleep(1)

    final_text = "\n\n".join(stitched)
    final_text = normalize_completion_marker(final_text.strip() + "\n\n" + COMPLETION_MARKER)

    # Final chapter-level validation (strict)
    ok, in_ct, out_ct = validate_output_coverage(chapter_text, final_text)
    if not ok:
        print(f"{now_str()}  ⚠️ Stitched chunk output mismatch (in:{in_ct} out:{out_ct}).")
        return None, agg_usage, total_retries

    return final_text, agg_usage, total_retries

# =========================
# MAIN RUN
# =========================

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_metadata_csv()

    manuscript = load_manuscript_data_nt_verses(FILE_PATH)
    NT_BOOK_ORDER = [
        "Matthew","Mark","Luke","John","Acts","Romans",
        "1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians","Colossians",
        "1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy","Titus","Philemon",
        "Hebrews","James","1 Peter","2 Peter","1 John","2 John","3 John","Jude","Revelation"
    ]
    order_index = {b: i for i, b in enumerate(NT_BOOK_ORDER)}
    keys = sorted(manuscript.keys(), key=lambda k: (order_index.get(k[0], 10**9), k[0], k[1]))
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

        # Skip empty chapters (shouldn't happen, but safe)
        if in_count == 0:
            print(f"{now_str()}  ⚠️ No verses detected for {book} {ch}; skipping.")
            continue

        # Resumability check
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                existing = f.read()
            if (COMPLETION_MARKER in existing) and validate_output_coverage(chapter_text, existing)[0]:
                print(f"{now_str()}  ⏭️ Skipping: {book} {ch} (already complete)")
                chapters_done += 1
                continue
            else:
                print(f"{now_str()}  🗂️ Existing output is incomplete: {file_name}")
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bak = save_path + f".incomplete_{ts}.bak"
                    os.replace(save_path, bak)
                    print(f"{now_str()}  🗂️ Moved -> {bak}")
                except OSError:
                    pass

        print(f"{now_str()}  📖 Processing: {book} {ch} ...")
        print(f"{now_str()}  ℹ️ Detected {in_count} verse(s) in source.")

        output_text = None
        usage_info = {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}
        retries_used = 0

        # Chunk if long
        if (len(chapter_text) > CHAR_LENGTH_CHUNK_THRESHOLD) or (in_count >= VERSE_COUNT_CHUNK_THRESHOLD):
            out, agg_usage, chunk_retries = generate_with_chunking(book, ch, chapter_text)
            retries_used += chunk_retries
            if out:
                output_text = out
                usage_info = agg_usage

        # Standard full-chapter attempt
        if not output_text:
            user_msg = make_standard_user_message(book, ch, chapter_text)
            out, usage, attempts = call_with_retries(
                book, ch,
                user_msg,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS_SINGLE,
                chapter_text=chapter_text
            )
            retries_used += attempts
            if out:
                output_text = out
                usage_info = usage_to_dict(usage)

        # Enumerated strict fallback
        if not output_text:
            enum_msg = make_enumerated_user_message(book, ch, verses)
            out, usage, attempts = call_with_retries(
                book, ch,
                enum_msg,
                temperature=FALLBACK_TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS_FALLBACK,
                chapter_text=chapter_text
            )
            retries_used += attempts
            if out:
                output_text = out
                usage_info = usage_to_dict(usage)

        # If still bad, write debug and error marker
        if not output_text:
            dbg = [
                f"Book: {book} Chapter: {ch}",
                f"Input verse count: {in_count}",
                "=== INPUT (first 1500 chars) ===",
                chapter_text[:1500],
            ]
            safe_write(debug_path, "\n\n".join(dbg))
            output_text = f"API ERROR: generation failed to produce full coverage for {book} {ch}. See {os.path.basename(debug_path)}"
            usage_info = {"prompt_token_count": 0, "candidates_token_count": 0, "total_billable_characters": 0}

        output_text = normalize_completion_marker(output_text)
        safe_write(save_path, output_text)

        prompt_tokens = int(usage_info.get("prompt_token_count", 0))
        out_tokens = int(usage_info.get("candidates_token_count", 0))
        cost = (prompt_tokens * IN_PRICE) + (out_tokens * OUT_PRICE)
        cumulative_cost += cost
        chapters_done += 1

        complete_flag = (COMPLETION_MARKER in output_text) and validate_output_coverage(chapter_text, output_text)[0]
        append_metadata(book, ch, prompt_tokens, out_tokens, cost, complete_flag, retries_used)

        elapsed = datetime.now() - start_time
        avg_per_ch = elapsed.total_seconds() / chapters_done if chapters_done else 0
        remaining = total_chapters - chapters_done
        eta = datetime.now() + timedelta(seconds=avg_per_ch * remaining) if remaining > 0 else datetime.now()

        print(f"{now_str()}  ✅ Stored: {file_name} | Cost: ${cost:.4f} | Cumulative: ${cumulative_cost:.2f}")
        print(f"    Progress: {chapters_done}/{total_chapters} | ETA: {eta.strftime('%Y-%m-%d %H:%M')}\n")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"{now_str()}  🎉 Run complete. Approx total cost: ${cumulative_cost:.2f}")
    print(f"{now_str()}  Outputs: {OUTPUT_DIR}")
    print(f"{now_str()}  Metadata: {METADATA_CSV}")

if __name__ == "__main__":
    run()
