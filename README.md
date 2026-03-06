# WordCloudGenerator — Business News Text Analysis Tool

## What the Tool Does

`WordCloudGenerator` takes raw business or news article text and returns:

- **Top-N keyword frequencies** — a ranked list of the most significant terms after stop-word removal
- **Summary statistics** — total token count and unique word count
- **Word-cloud image** — a PNG visualisation (requires `wordcloud` + `matplotlib`)

All output is returned as a JSON-serialisable `dict`, making it directly consumable by any agent framework or downstream pipeline.

**Realistic use case:** A business-intelligence agent monitors news feeds, scrapes incoming articles, and routes each one through this tool to surface key themes (e.g. *"tariffs", "revenue", "acquisition"*) without reading the full text.

---

## How to Run the Demo

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
# Scrape the latest BBC Business headline automatically
python demo.py

# Use a specific article URL
python demo.py --url https://www.bbc.com/news/articles/XXXXX

# Skip web scraping (use built-in sample text — no internet required)
python demo.py --no-scrape

# Change number of top keywords shown
python demo.py --top-n 25
```

The demo prints five cases:

| Case | Description |
|------|-------------|
| 1 | Successful analysis of a live BBC Business article |
| 2 | Error — empty string input |
| 3 | Error — text too short (< 50 characters) |
| 4 | Error — `top_n` out of valid range |
| 5 | Error — non-string input type |

---

## Design Decisions

**Why a word cloud for business news?**  
Keyword frequency is the simplest signal for topic detection in unstructured text. Word clouds have intuitive visual impact and are easy to explain to non-technical stakeholders — a core skill in Business AI.

**Heuristic stop-word filtering over a model**  
The tool uses a curated stop-word list (augmented by the `wordcloud` library's built-in list when available) rather than a language model. This keeps the implementation deterministic, fast, and free of external API calls — appropriate for a utility tool inside an agent.

**Graceful degradation**  
The `wordcloud` and `matplotlib` packages are optional. If they are not installed, the tool still returns full keyword-frequency data; only the PNG generation is skipped. This ensures the tool works in minimal environments.

**Structured error returns**  
Every failure path returns `{"success": False, "error": "<reason>"}` rather than raising an exception, so the calling agent can handle errors without a try/except wrapper.

## Limitations

- English text only (stop-word list and regex tokeniser are English-specific)
- Frequency-based ranking does not account for TF-IDF or positional importance
- Scraper relies on BBC's current HTML structure; layout changes may break extraction
- Very short articles (< 50 chars) are rejected; paywalled content may return little text
