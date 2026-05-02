"""
arXiv Daily Digest for Operator Algebras, Free Probability, and Random Matrix Theory
Runs daily via GitHub Actions. Fetches new papers, ranks by relevance,
and emails the top 3 with bilingual summaries.
"""

import os
import sys
import json
import datetime
import smtplib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Tuple, Optional

# ============================================================
# CONFIGURATION — Modify these via GitHub Secrets
# ============================================================
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
TO_EMAIL = os.environ.get("TO_EMAIL", GMAIL_ADDRESS)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")       # optional: DeepSeek / OpenAI
LLM_API_URL = os.environ.get("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ============================================================
# ARXIV CATEGORIES & KEYWORDS
# ============================================================
ARXIV_CATEGORIES = ["math.OA", "math.PR", "math.FA", "math.MP", "quant-ph"]

KEYWORDS_HIGH = [
    "operator algebra", "C*-algebra", "von Neumann algebra",
    "free probability", "random matrix", "noncommutative",
    "cuntz algebra", "k-theory", "subfactor",
    "free entropy", "free convolution", "free independence",
    "wigner matrix", "eigenvalue distribution", "spectral distribution",
    "quantum information", "quantum group"
]

KEYWORDS_MEDIUM = [
    "classification", "nuclear dimension", "amenable",
    "cartan subalgebra", "bounded cohomology", "approximation property",
    "universality", "largest eigenvalue", "tracy-widom",
    "gaussian orthogonal ensemble", "gaussian unitary ensemble",
    "graph of operators", "operator space"
]

# ============================================================
# ARXIV API
# ============================================================
def fetch_arxiv_papers(
    categories: List[str],
    max_results: int = 100,
    days_back: int = 1
) -> List[Dict]:
    """
    Fetch recent papers from arXiv API.
    """
    cat_query = "+OR+".join(f"cat:{cat}" for cat in categories)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={cat_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "DailyDigest/1.0"})
    response = urllib.request.urlopen(req, timeout=30)
    data = response.read().decode("utf-8")

    root = ET.fromstring(data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }

    papers = []
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        category_els = entry.findall("atom:category", ns)
        primary_cat = entry.find("arxiv:primary_category", ns)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_id = id_el.text.strip() if id_el is not None else ""

        # Extract pure arXiv ID
        arxiv_id_short = arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id

        published_str = published_el.text.strip() if published_el is not None else ""
        try:
            published_date = datetime.datetime.strptime(
                published_str, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=datetime.timezone.utc)
        except (ValueError, TypeError):
            published_date = datetime.datetime.now(datetime.timezone.utc)

        # Only include papers from the specified window
        if published_date < cutoff_date:
            continue

        cats = [c.get("term") for c in category_els if c.get("term")]
        primary = primary_cat.get("term") if primary_cat is not None else (cats[0] if cats else "")

        papers.append({
            "title": title,
            "summary": summary,
            "authors": authors,
            "arxiv_id": arxiv_id_short,
            "url": f"https://arxiv.org/abs/{arxiv_id_short}",
            "published": published_date,
            "categories": cats,
            "primary_category": primary,
        })

    return papers


# ============================================================
# RELEVANCE SCORING
# ============================================================
def score_paper(paper: Dict) -> float:
    """
    Score a paper by keyword relevance.
    """
    title_lower = paper["title"].lower()
    summary_lower = paper["summary"].lower()
    text = title_lower + " " + summary_lower

    score = 0.0

    # High-weight keywords
    for kw in KEYWORDS_HIGH:
        count = text.count(kw.lower())
        score += count * 10.0
        if kw.lower() in title_lower:
            score += 20.0  # Title match bonus

    # Medium-weight keywords
    for kw in KEYWORDS_MEDIUM:
        count = text.count(kw.lower())
        score += count * 5.0
        if kw.lower() in title_lower:
            score += 10.0

    # Category bonus
    if paper["primary_category"] == "math.OA":
        score += 15.0
    elif paper["primary_category"] == "math.PR":
        score += 8.0
    elif paper["primary_category"] == "math.FA":
        score += 8.0

    # Multi-category papers are often more significant
    score += len(paper["categories"]) * 3.0

    # Author count heuristic (single-author papers sometimes very important)
    if len(paper["authors"]) == 1:
        score += 2.0
    elif len(paper["authors"]) >= 4:
        score += 2.0

    return score


# ============================================================
# LLM RANKING & SUMMARIZATION (Optional)
# ============================================================
def llm_rank_and_summarize(
    papers: List[Dict], api_key: str, api_url: str, model: str
) -> List[Dict]:
    """
    Use an LLM to select the 3 most important papers and generate bilingual summaries.
    Falls back to keyword-based ranking if LLM is unavailable.
    """
    if not api_key or len(papers) <= 3:
        # Fall back to keyword scoring
        scored = sorted(papers, key=score_paper, reverse=True)
        return scored[:3]

    # Prepare prompt with candidate papers
    candidate_text = ""
    for i, p in enumerate(papers):
        candidate_text += (
            f"Paper {i+1}:\n"
            f"Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'][:5])}\n"
            f"Abstract: {p['summary'][:800]}\n"
            f"arXiv ID: {p['arxiv_id']}\n"
            f"Categories: {', '.join(p['categories'])}\n\n"
        )

    prompt = f"""You are a world-class mathematician specializing in operator algebras, free probability theory, and random matrix theory.

Below are new papers posted on arXiv today. Select the 3 MOST IMPORTANT papers for someone deeply interested in:
- Operator algebras (C*-algebras, von Neumann algebras, K-theory, classification)
- Free probability theory (free convolution, free entropy, free independence)
- Random matrix theory (spectral distributions, universality, Wigner matrices)
- Noncommutative geometry and quantum groups
- Connections between any of the above

Rank them by: novelty, depth, potential impact, and relevance to the above fields. Do NOT pick papers from completely unrelated fields.

{candidate_text}

Return your answer in the following JSON format (ONLY the JSON, no other text):
{{
  "papers": [
    {{
      "arxiv_id": "xxxx.xxxxx",
      "rank": 1,
      "reason_english": "1-2 sentence explanation of why this paper is important",
      "reason_japanese": "この論文が重要な理由を1〜2文の日本語で説明",
      "summary_english": "3-4 sentence technical summary of key results",
      "summary_japanese": "主要結果を3〜4文の日本語で技術的に要約"
    }}
  ]
}}
"""
    try:
        req_body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 3000,
        })
        req = urllib.request.Request(
            api_url,
            data=req_body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        response = urllib.request.urlopen(req, timeout=60)
        resp_data = json.loads(response.read().decode("utf-8"))
        content = resp_data["choices"][0]["message"]["content"]

        # Try to parse JSON from response
        # Strip possible markdown code fences
        content_stripped = content.strip()
        if content_stripped.startswith("```"):
            content_stripped = content_stripped.split("\n", 1)[1]
            if content_stripped.endswith("```"):
                content_stripped = content_stripped[:-3]

        llm_result = json.loads(content_stripped)
        selected = llm_result.get("papers", [])

        # Merge with original paper data
        paper_map = {p["arxiv_id"]: p for p in papers}
        merged = []
        for sel in selected:
            aid = sel.get("arxiv_id", "")
            if aid in paper_map:
                p = paper_map[aid].copy()
                p["rank"] = sel.get("rank", 0)
                p["reason_english"] = sel.get("reason_english", "")
                p["reason_japanese"] = sel.get("reason_japanese", "")
                p["summary_english"] = sel.get("summary_english", "")
                p["summary_japanese"] = sel.get("summary_japanese", "")
                merged.append(p)

        merged.sort(key=lambda x: x.get("rank", 999))
        return merged[:3]

    except Exception as e:
        print(f"[WARN] LLM ranking failed: {e}. Falling back to keyword scoring.")
        scored = sorted(papers, key=score_paper, reverse=True)
        return scored[:3]


def simple_japanese_title(title: str) -> str:
    """
    Very basic keyword-based Japanese title hint (fallback).
    Real Japanese translation needs LLM.
    """
    # This is a simple fallback; LLM mode provides real Japanese
    return f"(英題) {title}"


# ============================================================
# EMAIL
# ============================================================
def compose_email_html(papers: List[Dict], date_str: str, used_llm: bool) -> str:
    """
    Compose an HTML email with the daily digest.
    """
    lines = [
        "<html><body>",
        f"<h1>📚 arXiv Daily Digest — {date_str}</h1>",
        f"<p><b>Fields:</b> Operator Algebras · Free Probability · Random Matrix Theory</p>",
        f"<p><b>Mode:</b> {'🤖 LLM-ranked & summarized' if used_llm else '🔑 Keyword-ranked'}</p>",
        "<hr>",
    ]

    for i, paper in enumerate(papers):
        title = paper["title"]
        url = paper["url"]
        authors = ", ".join(paper["authors"])
        arxiv_id = paper["arxiv_id"]
        cats = ", ".join(paper.get("categories", []))
        reason_en = paper.get("reason_english", "")
        reason_jp = paper.get("reason_japanese", "")
        summary_en = paper.get("summary_english", paper["summary"][:600])
        summary_jp = paper.get("summary_japanese", "")

        lines.append(f"<h2>#{i+1}: {title}</h2>")
        lines.append(f"<p><b>Authors:</b> {authors}</p>")
        lines.append(f"<p><b>arXiv:</b> <a href='{url}'>{arxiv_id}</a> · <b>Categories:</b> {cats}</p>")

        if reason_en:
            lines.append(f"<p><b>Why important (EN):</b> {reason_en}</p>")
        if reason_jp:
            lines.append(f"<p><b>重要性 (JP):</b> {reason_jp}</p>")

        lines.append(f"<h3>📝 English Summary</h3>")
        lines.append(f"<p>{summary_en}</p>")

        if summary_jp:
            lines.append(f"<h3>🇯🇵 日本語要約</h3>")
            lines.append(f"<p>{summary_jp}</p>")
        else:
            lines.append(f"<p><i>(LLM未設定のため日本語要約は省略 — Japanese summary requires LLM API key)</i></p>")

        lines.append("<hr>")

    lines.append("<p><small>Generated by arXiv Daily Digest Bot · GitHub Actions</small></p>")
    lines.append("</body></html>")
    return "\n".join(lines)


def send_email(html_content: str, date_str: str):
    """
    Send the digest via Gmail SMTP.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("[ERROR] Gmail credentials not set. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD secrets.")
        sys.exit(1)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📚 arXiv Daily Digest — {date_str}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [TO_EMAIL], msg.as_string())
        print(f"[OK] Email sent to {TO_EMAIL}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"[{datetime.datetime.now()}] Starting arXiv daily digest...")

    # 1. Fetch papers
    print("[INFO] Fetching papers from arXiv...")
    papers = fetch_arxiv_papers(ARXIV_CATEGORIES, max_results=200, days_back=2)
    print(f"[INFO] Fetched {len(papers)} recent papers from {ARXIV_CATEGORIES}")

    if not papers:
        print("[WARN] No recent papers found. Exiting.")
        return

    # 2. Rank and summarize
    if LLM_API_KEY:
        print("[INFO] Using LLM for ranking and summarization...")
        top_papers = llm_rank_and_summarize(papers, LLM_API_KEY, LLM_API_URL, LLM_MODEL)
        used_llm = True
    else:
        print("[INFO] Using keyword-based ranking (no LLM configured)...")
        scored = sorted(papers, key=score_paper, reverse=True)
        top_papers = scored[:3]
        # Add fallback summaries
        for p in top_papers:
            p["summary_english"] = p["summary"][:600]
        used_llm = False

    if not top_papers:
        print("[WARN] No relevant papers found after filtering.")
        return

    print(f"[INFO] Selected top {len(top_papers)} papers:")
    for p in top_papers:
        print(f"  - [{p['primary_category']}] {p['title'][:80]}...")

    # 3. Compose and send email
    today_str = datetime.date.today().strftime("%Y-%m-%d (%A)")
    html = compose_email_html(top_papers, today_str, used_llm)
    send_email(html, today_str)

    print(f"[{datetime.datetime.now()}] Done.")


if __name__ == "__main__":
    main()
