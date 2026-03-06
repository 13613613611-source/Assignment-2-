"""
Demo: News Analysis Agent using the WordCloudGenerator tool
===========================================================
Demonstrates the full pipeline:
  1. Scrape a live BBC Business article via RSS
  2. Dispatch the WordCloudGenerator tool through a minimal rule-based agent
  3. Print results and error-handling cases

Usage
-----
    python demo.py                          # scrape latest BBC Business article
    python demo.py --url <article_url>      # use a custom article URL
    python demo.py --no-scrape              # skip web; use built-in sample text
    python demo.py --top-n 25              # change keyword count (default 20)
"""

from __future__ import annotations

import argparse
import re
import textwrap
from typing import Optional

import requests
from bs4 import BeautifulSoup

from tool import Tool, wordcloud_tool


# ---------------------------------------------------------------------------
# News scraper
# ---------------------------------------------------------------------------
_BBC_RSS = "https://feeds.bbci.co.uk/news/business/rss.xml"


def _latest_bbc_url(feed_url: str) -> Optional[str]:
    """Return the URL of the first article in a BBC Business RSS feed."""
    try:
        resp = requests.get(
            feed_url,
            timeout=10,
            headers={"User-Agent": "NewsAnalyser/1.0"},
        )
        resp.raise_for_status()
        # <link> in RSS is plain text content – regex is more reliable than
        # html.parser here because the HTML parser treats <link> as self-closing.
        urls = re.findall(r"<link>(https?://[^<]+)</link>", resp.text)
        # First hit is the channel homepage; second onward are articles.
        return urls[1] if len(urls) > 1 else (urls[0] if urls else None)
    except Exception:
        return None


def scrape_article(url: str) -> dict:
    """
    Fetch and extract the body text of a news article.

    Parameters
    ----------
    url : str
        Full HTTP/HTTPS URL of a news article page.

    Returns
    -------
    dict
        Success: ``{"success": True, "url": str, "text": str, "char_count": int}``
        Failure: ``{"success": False, "error": str}``
    """
    if not isinstance(url, str) or not url.strip():
        return {"success": False, "error": "url must be a non-empty string"}

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"success": False, "error": "url must start with http:// or https://"}

    # ── Fetch ─────────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsAnalyser/1.0)"},
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out after 15 s"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the server"}
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code
        return {"success": False, "error": f"HTTP {code}: {exc.response.reason}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    # ── Parse ─────────────────────────────────────────────────────────────
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate regions
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    # Try article-specific selectors (most specific first)
    body = ""
    for selector in [
        "article",
        "[data-component='text-block']",
        "[data-testid='article-body']",
        ".ssrcss-uf6wea-RichTextComponentWrapper",   # BBC rich text
        ".article-body__content",
        ".story-body__inner",
        ".ArticleBody",
        "main",
    ]:
        el = soup.select_one(selector)
        if el:
            paras = el.find_all("p")
            candidate = " ".join(p.get_text(" ", strip=True) for p in paras)
            if len(candidate) > 300:
                body = candidate
                break

    if len(body) < 100:
        # Last resort: all <p> tags on the page
        body = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))

    body = body.strip()
    if not body:
        return {"success": False, "error": "Could not extract article text from page"}

    return {"success": True, "url": url, "text": body, "char_count": len(body)}


# ---------------------------------------------------------------------------
# Minimal rule-based agent
# ---------------------------------------------------------------------------
class NewsAnalysisAgent:
    """
    A minimal agent that selects tools by keyword matching and runs them.

    Parameters
    ----------
    tools : list[Tool]
        Available tools the agent can dispatch.
    """

    _TRIGGER_WORDS = frozenset(
        {"word cloud", "analyse", "analyze", "keywords", "frequency", "text analysis"}
    )

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def _select_tool(self, task: str) -> Optional[str]:
        """Return the best-matching tool name for the given task string."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in self._TRIGGER_WORDS):
            return "WordCloudGenerator"
        return None

    def run(self, task: str, url: str, top_n: int = 20) -> dict:
        """
        Execute the full pipeline: scrape → tool selection → analysis.

        Parameters
        ----------
        task : str
            Natural-language description of what the agent should do.
        url : str
            News article URL to process.
        top_n : int
            Number of top keywords to surface.

        Returns
        -------
        dict
            Tool result dict, or an error dict with a ``"stage"`` key
            indicating where the failure occurred.
        """
        print(f"[Agent] Task  : {task}")
        print(f"[Agent] Source: {url}\n")

        # Step 1 – fetch article
        print("[Agent] Step 1 – Scraping article...")
        scrape = scrape_article(url)
        if not scrape["success"]:
            print(f"[Agent] Scraping failed: {scrape['error']}")
            return {"success": False, "stage": "scraping", "error": scrape["error"]}
        print(f"[Agent] Retrieved {scrape['char_count']:,} characters.\n")

        # Step 2 – select tool
        tool_name = self._select_tool(task)
        if tool_name is None:
            return {
                "success": False,
                "stage": "tool_selection",
                "error": "No suitable tool found for this task",
            }
        tool = self._tools[tool_name]
        print(f"[Agent] Step 2 – Selected tool: {tool.name}")

        # Step 3 – execute
        print("[Agent] Step 3 – Running analysis...\n")
        return tool.execute(text=scrape["text"], top_n=top_n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAMPLE_TEXT = textwrap.dedent("""
    Apple reported record quarterly revenue of $124.3 billion, driven by strong iPhone 16
    sales and robust Services growth. CEO Tim Cook highlighted the company's expansion into
    artificial intelligence, with Apple Intelligence features rolling out across iOS 18.
    The stock rose 3.2 percent following the earnings announcement. Analysts noted that iPhone
    demand in China remained steady despite geopolitical tensions, while the Indian market
    continued to post double-digit growth. Services revenue reached 26.3 billion dollars,
    marking a 17 percent year-over-year increase. CFO Luca Maestri indicated that the company
    plans to increase capital expenditure in semiconductor supply chains, reinforcing Apple's
    commitment to reducing dependence on third-party chip manufacturers. The board also
    approved a 110 billion dollar share buyback programme, one of the largest in corporate
    history. Investors reacted positively to the announcement, pushing the stock to an
    all-time high. Analysts from Goldman Sachs and Morgan Stanley both raised their price
    targets, citing confidence in Apple's long-term growth strategy and product pipeline.
""").strip()


def _print_result(result: dict, label: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)

    if not result.get("success"):
        print(f"  [FAIL] {result.get('error', 'Unknown error')}\n")
        return

    print(f"  Tokens analysed : {result['word_count']:,}")
    print(f"  Unique words    : {result['unique_words']:,}")
    print(f"  Image generated : {result['image_generated']}")
    if result.get("image_path"):
        print(f"  Image saved to  : {result['image_path']}")

    print("\n  Top keywords:")
    print(f"  {'Rank':<6} {'Word':<20} {'Count':>6}")
    print(f"  {'-'*6} {'-'*20} {'-'*6}")
    for rank, (word, count) in enumerate(result["top_words"][:15], start=1):
        bar = "█" * min(count, 30)
        print(f"  {rank:<6} {word:<20} {count:>6}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate the WordCloudGenerator tool on live business news."
    )
    parser.add_argument("--url", help="Custom article URL to scrape")
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip web scraping; use the built-in sample text instead",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Number of top keywords to display (default: 20)",
    )
    args = parser.parse_args()

    agent = NewsAnalysisAgent(tools=[wordcloud_tool])

    # ------------------------------------------------------------------
    # CASE 1 – Successful execution on a real/sample article
    # ------------------------------------------------------------------
    print("\n" + "─" * 62)
    print("  CASE 1 – Successful execution")
    print("─" * 62)

    if args.no_scrape:
        print("[Demo] Using built-in sample text (--no-scrape flag set).\n")
        result = wordcloud_tool.execute(text=_SAMPLE_TEXT, top_n=args.top_n)
        _print_result(result, "CASE 1 – Sample text analysis")
    else:
        article_url = args.url
        if not article_url:
            print("[Demo] Fetching latest BBC Business article via RSS...\n")
            article_url = _latest_bbc_url(_BBC_RSS)
            if not article_url:
                print("[Demo] RSS feed unavailable – falling back to sample text.\n")
                result = wordcloud_tool.execute(text=_SAMPLE_TEXT, top_n=args.top_n)
                _print_result(result, "CASE 1 – Sample text analysis (fallback)")
                article_url = None

        if article_url:
            result = agent.run(
                task="Generate a word cloud and keyword analysis of this business news article",
                url=article_url,
                top_n=args.top_n,
            )
            _print_result(result, "CASE 1 – Live BBC Business article")

    # ------------------------------------------------------------------
    # CASE 2 – Error: empty input
    # ------------------------------------------------------------------
    print("─" * 62)
    print("  CASE 2 – Error handling: empty input")
    print("─" * 62 + "\n")
    bad = wordcloud_tool.execute(text="")
    _print_result(bad, "CASE 2 – Empty text")

    # ------------------------------------------------------------------
    # CASE 3 – Error: text too short
    # ------------------------------------------------------------------
    print("─" * 62)
    print("  CASE 3 – Error handling: text too short")
    print("─" * 62 + "\n")
    short = wordcloud_tool.execute(text="Revenue up. Stocks fell.")
    _print_result(short, "CASE 3 – Text too short")

    # ------------------------------------------------------------------
    # CASE 4 – Error: invalid parameter (top_n out of range)
    # ------------------------------------------------------------------
    print("─" * 62)
    print("  CASE 4 – Error handling: invalid top_n")
    print("─" * 62 + "\n")
    invalid = wordcloud_tool.execute(text=_SAMPLE_TEXT, top_n=-5)
    _print_result(invalid, "CASE 4 – top_n = -5 (invalid)")

    # ------------------------------------------------------------------
    # CASE 5 – Error: wrong input type
    # ------------------------------------------------------------------
    print("─" * 62)
    print("  CASE 5 – Error handling: non-string input")
    print("─" * 62 + "\n")
    wrong_type = wordcloud_tool.execute(text=12345)
    _print_result(wrong_type, "CASE 5 – text is an integer")


if __name__ == "__main__":
    main()
