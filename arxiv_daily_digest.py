"""
arXiv Daily Digest — Operator Algebras · Free Probability · Random Matrix Theory
GitHub Actionsで毎日自動実行。新着論文を取得→重要度順に3本選定→メール送信。
"""
import os
import sys
import json
import datetime
import smtplib
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional

# ═══════════════════════════════════════════════════════════════
# 設定（GitHub Secrets から読み取り）
# ═══════════════════════════════════════════════════════════════
GMAIL_ADDRESS   = os.environ.get("GMAIL_ADDRESS", "").strip()
GMAIL_APP_PW    = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
TO_EMAIL        = os.environ.get("TO_EMAIL", "").strip() or GMAIL_ADDRESS
LLM_API_KEY     = os.environ.get("LLM_API_KEY", "").strip()
LLM_API_URL     = os.environ.get("LLM_API_URL", "").strip() or "https://api.deepseek.com/v1/chat/completions"
LLM_MODEL       = os.environ.get("LLM_MODEL", "").strip() or "deepseek-chat"

# ═══════════════════════════════════════════════════════════════
# arXiv カテゴリ・キーワード
# ═══════════════════════════════════════════════════════════════
ARXIV_CATEGORIES = ["math.OA", "math.PR", "math.FA", "math.MP", "quant-ph"]

KEYWORDS_HIGH = [
    "operator algebra", "c*-algebra", "von neumann algebra",
    "free probability", "random matrix", "noncommutative",
    "cuntz algebra", "k-theory", "subfactor",
    "free entropy", "free convolution", "free independence",
    "wigner matrix", "eigenvalue distribution", "spectral distribution",
    "quantum information", "quantum group",
]

KEYWORDS_MEDIUM = [
    "classification", "nuclear dimension", "amenable", "cartan subalgebra",
    "bounded cohomology", "approximation property", "universality",
    "largest eigenvalue", "tracy-widom", "gaussian orthogonal ensemble",
    "gaussian unitary ensemble", "graph of operators", "operator space",
    "tensor category", "kirchberg", "elliott",
]


# ═══════════════════════════════════════════════════════════════
# arXiv API
# ═══════════════════════════════════════════════════════════════
def fetch_arxiv_papers(categories: List[str], max_results: int = 150, days_back: int = 2) -> List[Dict]:
    cat_q = "+OR+".join(f"cat:{c}" for c in categories)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={cat_q}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "DigestBot/1.2"})
    try:
        resp = urllib.request.urlopen(req, timeout=45)
    except Exception as e:
        print(f"[FATAL] arXiv API unreachable: {e}")
        return []

    root = ET.fromstring(resp.read().decode("utf-8"))
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_back)
    papers = []

    for e in root.findall("atom:entry", ns):
        title_el  = e.find("atom:title", ns)
        summ_el   = e.find("atom:summary", ns)
        pub_el    = e.find("atom:published", ns)
        id_el     = e.find("atom:id", ns)
        prim_el   = e.find("arxiv:primary_category", ns)
        authors   = [a.find("atom:name", ns).text for a in e.findall("atom:author", ns)]
        cats      = [c.get("term") for c in e.findall("atom:category", ns) if c.get("term")]
        primary   = prim_el.get("term") if prim_el is not None else (cats[0] if cats else "")
        title     = title_el.text.strip().replace("\n"," ") if title_el is not None and title_el.text else ""
        summary   = summ_el.text.strip().replace("\n"," ")  if summ_el is not None and summ_el.text else ""
        aid_full  = id_el.text.strip() if id_el is not None and id_el.text else ""
        aid       = aid_full.split("/abs/")[-1]
        pub_str   = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        try:
            pub_dt = datetime.datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        except Exception:
            pub_dt = datetime.datetime.now(datetime.timezone.utc)
        if pub_dt < cutoff:
            continue

        papers.append(dict(title=title, summary=summary, authors=authors,
                           arxiv_id=aid, url=f"https://arxiv.org/abs/{aid}",
                           published=pub_dt, categories=cats, primary_category=primary))
    return papers


# ═══════════════════════════════════════════════════════════════
# キーワードスコアリング
# ═══════════════════════════════════════════════════════════════
def score_paper(p: Dict) -> float:
    t = (p["title"] + " " + p["summary"]).lower()
    s = 0.0
    for kw in KEYWORDS_HIGH:
        n = t.count(kw)
        if n:
            s += n * 10.0
            if kw in p["title"].lower():
                s += 20.0
    for kw in KEYWORDS_MEDIUM:
        n = t.count(kw)
        if n:
            s += n * 5.0
            if kw in p["title"].lower():
                s += 10.0
    if p["primary_category"] == "math.OA":  s += 15.0
    elif p["primary_category"] == "math.PR": s += 8.0
    elif p["primary_category"] == "math.FA": s += 8.0
    s += len(p["categories"]) * 3.0
    if len(p["authors"]) == 1: s += 2.0
    elif len(p["authors"]) >= 4: s += 2.0
    return s


# ═══════════════════════════════════════════════════════════════
# LLM ランキング（オプション）
# ═══════════════════════════════════════════════════════════════
def llm_rank_and_summarize(papers: List[Dict]) -> Optional[List[Dict]]:
    if not LLM_API_KEY or not LLM_API_URL:
        return None
    if len(papers) <= 3:
        return papers[:3]

    buf = ""
    for i, p in enumerate(papers):
        buf += (f"Paper {i+1}:\nTitle: {p['title']}\n"
                f"Authors: {', '.join(p['authors'][:5])}\n"
                f"Abstract: {p['summary'][:800]}\n"
                f"arXiv ID: {p['arxiv_id']}\nCategories: {', '.join(p['categories'])}\n\n")

    prompt = f"""You are a mathematician specialized in operator algebras, free probability, and random matrix theory.
From the list below, pick the 3 most important NEW papers. Rank by novelty, depth, relevance.
Exclude papers clearly outside the above fields.

Return ONLY this JSON (no markdown):
{{"papers":[
  {{"arxiv_id":"xxxx.xxxxx","rank":1,
    "reason_english":"Why important (1-2 sentences)",
    "reason_japanese":"重要性の日本語説明 (1〜2文)",
    "summary_english":"Technical summary (3-4 sentences)",
    "summary_japanese":"日本語の技術的要約 (3〜4文)"}}
]}}

{buf}"""

    try:
        body = json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 3000
        })
        req = urllib.request.Request(LLM_API_URL, data=body.encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {LLM_API_KEY}"})
        raw = json.loads(urllib.request.urlopen(req, timeout=90).read())
        content = raw["choices"][0]["message"]["content"].strip()
        # strip optional code fences
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        selected = json.loads(content).get("papers", [])
        pmap = {p["arxiv_id"]: p for p in papers}
        out = []
        for s in selected:
            aid = s.get("arxiv_id", "")
            if aid in pmap:
                pp = pmap[aid].copy()
                for k in ("rank","reason_english","reason_japanese","summary_english","summary_japanese"):
                    pp[k] = s.get(k, "")
                out.append(pp)
        out.sort(key=lambda x: x.get("rank", 99))
        return out[:3] if out else None
    except Exception as e:
        print(f"[WARN] LLM failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# HTMLメール構築
# ═══════════════════════════════════════════════════════════════
def build_html(papers: List[Dict], used_llm: bool) -> str:
    today = datetime.date.today().strftime("%Y-%m-%d (%A)")
    h = [f"<html><body style='font-family:Meiryo,Hiragino Sans,sans-serif'>",
         f"<h1>📚 arXiv Daily Digest — {today}</h1>",
         f"<p><b>Operator Algebras · Free Probability · Random Matrix Theory</b></p>",
         f"<p>Mode: {'🤖 LLM' if used_llm else '🔑 Keyword'}</p><hr>"]
    for i, p in enumerate(papers):
        h.append(f"<h2>#{i+1} 「{p['title']}」</h2>")
        h.append(f"<p><b>Authors:</b> {', '.join(p['authors'])}</p>")
        h.append(f"<p><b>arXiv:</b> <a href='{p['url']}'>{p['arxiv_id']}</a> · "
                 f"<b>Categories:</b> {', '.join(p.get('categories',[]))}</p>")
        if p.get("reason_english"):
            h.append(f"<p><b>Why:</b> {p['reason_english']}</p>")
        if p.get("reason_japanese"):
            h.append(f"<p><b>重要性:</b> {p['reason_japanese']}</p>")
        h.append(f"<h3>📝 English Summary</h3><p>{p.get('summary_english', p['summary'][:600])}</p>")
        if p.get("summary_japanese"):
            h.append(f"<h3>🇯🇵 日本語要約</h3><p>{p['summary_japanese']}</p>")
        h.append("<hr>")
    h.append("<p><small>arXiv Daily Digest · GitHub Actions</small></p></body></html>")
    return "\n".join(h)


# ═══════════════════════════════════════════════════════════════
# メール送信（Gmail SMTP）
# ═══════════════════════════════════════════════════════════════
def send_email(html: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PW:
        print("[SKIP] Gmail secrets not set. Email not sent.")
        print("[SKIP] → Set GMAIL_ADDRESS + GMAIL_APP_PASSWORD in GitHub Secrets.")
        return  # exit code 0

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"=?utf-8?B?{datetime.date.today().strftime('%Y-%m-%d')}?= =?utf-8?B?IPCfmoQg?= arXiv Daily Digest"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText(html, "html", "utf-8"))

    # 方法1: STARTTLS (port 587)
    errors = []
    for method, host, port in [
        ("STARTTLS", "smtp.gmail.com", 587),
        ("SSL",      "smtp.gmail.com", 465),
    ]:
        try:
            if method == "SSL":
                server = smtplib.SMTP_SSL(host, port, timeout=20, local_hostname="github-actions")
            else:
                server = smtplib.SMTP(host, port, timeout=20, local_hostname="github-actions")
                server.ehlo()
                server.starttls()
                server.ehlo()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PW)
            server.sendmail(GMAIL_ADDRESS, [TO_EMAIL], msg.as_string())
            server.quit()
            print(f"[OK] Email sent via {method} ({host}:{port}) → {TO_EMAIL}")
            return
        except smtplib.SMTPAuthenticationError:
            print("[FATAL] Gmail rejected login. Check GMAIL_APP_PASSWORD (16 chars, no spaces).")
            print("[FATAL] Also check: 2-step verification is ON at myaccount.google.com/security")
            sys.exit(1)
        except Exception as e:
            errors.append(f"{method}:{port} → {e}")
            continue

    # どちらも失敗
    print(f"[ERROR] All SMTP methods failed:")
    for err in errors:
        print(f"  - {err}")
    print("[HINT] GitHub Actions IP might be blocked by Gmail. Try:")
    print("       1. Use a different email provider (SendGrid, Mailgun)")
    print("       2. Or run this script on your own PC/server")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"[{datetime.datetime.now().isoformat()}] === START ===")

    # 1. 論文取得
    papers = fetch_arxiv_papers(ARXIV_CATEGORIES)
    print(f"[INFO] Fetched {len(papers)} recent papers")
    if not papers:
        print("[DONE] No recent papers. Exiting normally.")
        return

    # 2. ランキング
    top = llm_rank_and_summarize(papers) if LLM_API_KEY else None
    used_llm = top is not None

    if not used_llm:
        print("[INFO] Using keyword-based ranking")
        papers.sort(key=score_paper, reverse=True)
        top = papers[:3]
        for p in top:
            p["summary_english"] = p["summary"][:600]

    print(f"[INFO] Selected top {len(top)} papers:")
    for p in top:
        print(f"  [{p.get('primary_category','?')}] {p['title'][:80]}")

    # 3. HTML構築 → メール送信
    html = build_html(top, used_llm)

    # デバッグ用: ログにも簡易表示
    print("─" * 60)
    for i, p in enumerate(top):
        print(f"#{i+1}: {p['title']}  ({p['arxiv_id']})")
    print("─" * 60)

    send_email(html)
    print(f"[{datetime.datetime.now().isoformat()}] === DONE ===")


if __name__ == "__main__":
    main()
