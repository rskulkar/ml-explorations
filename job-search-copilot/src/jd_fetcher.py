"""Job description fetching and portal detection."""

import requests
from bs4 import BeautifulSoup
from typing import Tuple


PORTAL_PATTERNS = {
    "linkedin": ["linkedin.com"],
    "naukri": ["naukri.com"],
    "indeed": ["indeed.com"],
    "glassdoor": ["glassdoor.com"],
}


def detect_portal(url: str) -> str:
    """Detect the job portal from a URL.

    Args:
        url: URL to analyze

    Returns:
        Portal name or "unknown"
    """
    url_lower = url.lower()

    for portal, patterns in PORTAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in url_lower:
                return portal

    return "unknown"


def fetch_jd(url: str, timeout: int = 15) -> str:
    """Fetch job description text from a URL.

    Args:
        url: URL to fetch from
        timeout: Request timeout in seconds

    Returns:
        Extracted text content, or empty string on failure
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        return text
    except Exception as e:
        print(f"Warning: Failed to fetch JD from {url}: {e}")
        return ""


def load_jd(source: str) -> Tuple[str, str]:
    """Load job description from URL or pasted text.

    Args:
        source: Either a URL (starts with 'http') or pasted JD text

    Returns:
        Tuple of (jd_text, portal) where portal is detected for URLs or 'paste' for pasted text
    """
    if source.strip().lower().startswith("http"):
        # It's a URL
        jd_text = fetch_jd(source)

        if len(jd_text) < 200:
            print(f"Warning: Fetched JD is very short ({len(jd_text)} chars). URL may be incorrect or content blocked.")

        portal = detect_portal(source)
        return jd_text, portal
    else:
        # It's pasted text
        return source, "paste"
