"""
WordCloud Generator Tool for Business News Text Analysis
=========================================================
Processes raw news/business text to extract keyword frequencies and
optionally render a PNG word-cloud image.  Always returns a
JSON-serialisable dict so it can be consumed by any agent framework.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# Optional image-generation backend
try:
    from wordcloud import WordCloud, STOPWORDS as _WC_STOPS
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend – safe for scripts
    import matplotlib.pyplot as plt
    _IMG_AVAILABLE = True
except ImportError:
    _IMG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
_BASE_STOPS: set[str] = {
    # articles / prepositions / conjunctions
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "into", "about", "as", "if", "while",
    "after", "before", "between", "through", "during", "against", "without",
    # common auxiliary / reporting verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "may", "might",
    "can", "could", "said", "says", "told", "according", "reported",
    # pronouns / determiners
    "it", "its", "he", "she", "they", "we", "you", "i", "me", "him",
    "her", "us", "them", "their", "our", "your", "my", "his",
    "who", "which", "what", "where", "when", "how",
    "this", "that", "these", "those", "all", "any", "some", "each",
    "few", "more", "most", "other", "such",
    # filler adverbs
    "only", "also", "just", "not", "no", "nor", "than", "too", "very",
    "now", "then", "still", "even", "much", "well",
    # noise tokens left after regex tokenisation
    "s", "t", "re", "ll", "ve", "d", "m", "n",
}


# ---------------------------------------------------------------------------
# Tool wrapper (matches the suggested structure in the assignment)
# ---------------------------------------------------------------------------
class Tool:
    """Associates a callable function with agent-readable name and description."""

    def __init__(self, name: str, description: str, fn):
        self.name = name
        self.description = description
        self.fn = fn

    def execute(self, **kwargs) -> dict:
        """Invoke the wrapped function with keyword arguments."""
        return self.fn(**kwargs)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def generate_wordcloud(
    text: str,
    top_n: int = 30,
    output_path: str = "wordcloud.png",
    custom_stopwords: Optional[list] = None,
    min_word_length: int = 3,
) -> dict:
    """
    Generate keyword frequencies and an optional word-cloud image from
    business/news text.

    Parameters
    ----------
    text : str
        Raw article text.  Must be a non-empty string of at least 50 characters.
    top_n : int
        Number of top keywords to return (1–200, default 30).
    output_path : str
        File path for the saved PNG (default ``"wordcloud.png"``).
        Ignored when ``wordcloud`` / ``matplotlib`` are not installed.
    custom_stopwords : list[str] | None
        Additional words to suppress beyond the built-in list.
    min_word_length : int
        Minimum token length to retain (default 3).

    Returns
    -------
    dict
        Success::

            {
                "success": True,
                "top_words": [[word, count], ...],  # top_n entries
                "word_count": int,                  # tokens after filtering
                "unique_words": int,                # distinct tokens
                "image_path": str | None,           # PNG path or None
                "image_generated": bool,
            }

        Failure::

            {"success": False, "error": "<reason>"}
    """
    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    if not isinstance(text, str):
        return {"success": False, "error": "text must be a string"}

    text = text.strip()
    if not text:
        return {"success": False, "error": "text is empty"}

    if len(text) < 50:
        return {
            "success": False,
            "error": f"text too short ({len(text)} chars); minimum is 50",
        }

    if not isinstance(top_n, int) or not (1 <= top_n <= 200):
        return {"success": False, "error": "top_n must be an integer between 1 and 200"}

    if not isinstance(min_word_length, int) or min_word_length < 1:
        return {"success": False, "error": "min_word_length must be a positive integer"}

    if custom_stopwords is not None and not isinstance(custom_stopwords, list):
        return {"success": False, "error": "custom_stopwords must be a list of strings"}

    # ------------------------------------------------------------------ #
    # Build stopword set                                                   #
    # ------------------------------------------------------------------ #
    stops = set(_BASE_STOPS)
    if _IMG_AVAILABLE:
        stops |= {w.lower() for w in _WC_STOPS}
    if custom_stopwords:
        stops |= {str(w).lower() for w in custom_stopwords}

    # ------------------------------------------------------------------ #
    # Tokenise and filter                                                  #
    # ------------------------------------------------------------------ #
    raw_tokens = re.findall(r"[a-zA-Z]+", text.lower())
    filtered = [t for t in raw_tokens if len(t) >= min_word_length and t not in stops]

    if not filtered:
        return {"success": False, "error": "No meaningful words found after filtering"}

    counter = Counter(filtered)
    top_words = counter.most_common(top_n)

    # ------------------------------------------------------------------ #
    # Optional: render PNG word cloud                                      #
    # ------------------------------------------------------------------ #
    image_path: Optional[str] = None
    image_generated = False

    if _IMG_AVAILABLE:
        try:
            wc = WordCloud(
                width=900,
                height=450,
                background_color="white",
                stopwords=stops,
                max_words=top_n,
                collocations=False,
            ).generate(text)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            image_path = output_path
            image_generated = True
        except Exception:
            # Image generation is non-fatal; frequency data is still returned.
            pass

    return {
        "success": True,
        "top_words": [[word, count] for word, count in top_words],
        "word_count": len(filtered),
        "unique_words": len(counter),
        "image_path": image_path,
        "image_generated": image_generated,
    }


# ---------------------------------------------------------------------------
# Exported tool instance
# ---------------------------------------------------------------------------
wordcloud_tool = Tool(
    name="WordCloudGenerator",
    description=(
        "Analyses business/news text, extracts keyword frequencies, "
        "and renders a PNG word-cloud image. "
        "Returns a JSON-serialisable dict with top_words, word_count, "
        "unique_words, and the image path."
    ),
    fn=generate_wordcloud,
)
