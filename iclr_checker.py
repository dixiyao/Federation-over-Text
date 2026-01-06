"""
Check ICLR Accept (Oral) papers for skill guidance using Gemini.
- Fetches ICLR papers for a given year directly from OpenReview API.
- Uses a provided skills encyclopedia (JSON mapping of name->description or plain text) as guidance.
- Sends title + abstract + skills to Gemini and records which skills apply.
- Outputs a summary count (guided/total) and a JSON report with per-paper results.

Usage example:
  python guided_accept_oral_checker.py \
      --gemini-key $GEMINI_API_KEY \
      --encyclopedia important_checkpoints/client_aime25_server_math500/encyclopedia.json \
      --year 2024 \
      --output guided_oral_results.json
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import requests
import re
import warnings
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Prefer new google.genai; fall back to deprecated google.generativeai
HAS_GENAI = False
HAS_GEMINI = False
try:
    import google.genai as genai_new  # type: ignore
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import google.generativeai as genai_old  # type: ignore
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not (HAS_GENAI or HAS_GEMINI):
            raise ImportError(
                "Install google-genai (preferred) or google-generativeai. Example: pip install google-genai"
            )
        self.model_name = model_name
        self.backend = "new" if HAS_GENAI else "old"
        if self.backend == "new":
            self.client = genai_new.Client(api_key=api_key)
        else:
            genai_old.configure(api_key=api_key)
            self.model = genai_old.GenerativeModel(model_name)

    def generate_text(self, prompt: str) -> str:
        if self.backend == "new":
            resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
            # Try primary accessor
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            # Fallback: attempt to stitch candidate parts
            try:
                candidates = getattr(resp, "candidates", []) or []
                parts = []
                for c in candidates:
                    content = getattr(c, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            if hasattr(p, "text") and p.text:
                                parts.append(p.text)
                if parts:
                    return "\n".join(parts).strip()
            except Exception:
                pass
            raise RuntimeError("Failed to extract text from google.genai response")
        else:
            resp = self.model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            # Fallback similar to client.py logic
            try:
                candidate = resp.candidates[0]
                if getattr(candidate, "finish_reason", None) == "RECITATION" and getattr(candidate, "safety_ratings", None):
                    raise RuntimeError("Gemini API blocked the response due to recitation.")
                if candidate.content and candidate.content.parts:
                    parts = [
                        part.text for part in candidate.content.parts if hasattr(part, "text") and part.text
                    ]
                    if parts:
                        return "\n".join(parts).strip()
            except Exception:
                pass
            raise RuntimeError("Failed to extract text from google.generativeai response")


def load_skills(encyclopedia_path: str) -> Tuple[List[Tuple[str, str]], str]:
    """Load skills from encyclopedia file.

    Returns:
        A list of (name, description) tuples and a formatted string for prompting.
    """
    if not os.path.exists(encyclopedia_path):
        raise FileNotFoundError(f"Encyclopedia not found at {encyclopedia_path}")

    skills: List[Tuple[str, str]] = []
    if encyclopedia_path.endswith(".json"):
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            skills = [(k, v) for k, v in data.items()]
        elif isinstance(data, list):
            # Support list of {name, description}
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("skill") or "skill"
                    desc = item.get("description") or item.get("insight") or ""
                    skills.append((name, desc))
        else:
            raise ValueError("Unsupported JSON format for encyclopedia")
    else:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # Fallback: treat entire text as one skill chunk
        skills = [("encyclopedia_text", text)]

    if not skills:
        raise ValueError("No skills found in encyclopedia")

    prompt_block = []
    for idx, (name, desc) in enumerate(skills, 1):
        prompt_block.append(f"{idx}. {name}: {desc}")
    return skills, "\n".join(prompt_block)


def fetch_accept_tracks(
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
) -> List[Dict]:
    """Fetch accepted papers using OpenReview client.

    Uses openreview-py to query ICLR submissions and filter by venue field.
    """
    try:
        import openreview
        use_or_client = True
    except ImportError:
        use_or_client = False
        print("Warning: openreview-py not installed, falling back to requests")
    
    # Default to oral if nothing specified (backward compatible)
    accept_any = accept_oral or accept_spotlight or accept_poster
    accept_oral = accept_oral or not accept_any
    
    decisions: List[Dict] = []
    
    # Try OpenReview client first
    if use_or_client:
        try:
            client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
            print(f"Fetching ICLR {year} submissions via OpenReview client...")
            submissions = list(client.get_all_notes(
                invitation=f'ICLR.cc/{year}/Conference/-/Submission',
                details='directReplies'
            ))
            print(f"Retrieved {len(submissions)} submissions, filtering by venue...")
            
            for sub in submissions:
                content = sub.content
                # API v2 nests values
                venue = str(content.get('venue', {}).get('value', ''))
                
                # Check if accepted and what track
                track = None
                if 'oral' in venue.lower():
                    track = 'oral'
                    if not accept_oral:
                        continue
                elif 'spotlight' in venue.lower():
                    track = 'spotlight'
                    if not accept_spotlight:
                        continue
                elif 'poster' in venue.lower():
                    track = 'poster'
                    if not accept_poster:
                        continue
                else:
                    continue
                
                decisions.append({"forum": sub.forum, "track": track})
                if max_papers and len(decisions) >= max_papers:
                    break
            
            if decisions:
                print(f"Found {len(decisions)} accepted papers via client")
                return _hydrate_papers_from_client(client, decisions)
            else:
                print("No accepted papers found via client")
                return []
        except Exception as e:
            print(f"OpenReview client error: {e}")
            return []
    
    # Fallback: old requests-based approach
    print("Using requests-based fallback (may not work for ICLR 2024+)...")
    return []


def _hydrate_papers_from_client(client, forums: List[Dict]) -> List[Dict]:
    """Hydrate paper details using OpenReview client."""
    import openreview
    hydrated: List[Dict] = []
    for stub in forums:
        fid = stub.get("forum") or stub.get("id")
        try:
            note = client.get_note(fid)
            content = note.content
            hydrated.append({
                "id": note.id,
                "forum": note.forum,
                "title": content.get("title", {}).get("value", ""),
                "abstract": content.get("abstract", {}).get("value", ""),
                "venue": content.get("venue", {}).get("value", ""),
                "venueid": content.get("venueid", {}).get("value", ""),
                "track": stub.get("track", ""),
            })
        except Exception as e:
            print(f"  Warning: failed to hydrate {fid}: {e}")
            hydrated.append({
                "id": fid,
                "forum": fid,
                "title": "",
                "abstract": "",
                "venue": "",
                "venueid": "",
                "track": stub.get("track", ""),
            })
        time.sleep(0.2)
    return hydrated


def _scrape_accept_forum_ids(
    year: int,
    accept_oral: bool,
    accept_spotlight: bool,
    accept_poster: bool,
    session: requests.Session,
) -> List[Dict]:
    """Scrape accept tabs to collect forum IDs when API filtering yields none."""
    base = "https://openreview.net"
    urls = []
    if accept_oral or (not accept_spotlight and not accept_poster):
        urls.append(("oral", f"{base}/group?id=ICLR.cc/{year}/Conference#tab-accept-oral"))
    if accept_spotlight:
        urls.append(("spotlight", f"{base}/group?id=ICLR.cc/{year}/Conference#tab-accept-spotlight"))
    if accept_poster:
        urls.append(("poster", f"{base}/group?id=ICLR.cc/{year}/Conference#tab-accept-poster"))

    forum_ids: List[Tuple[str, str]] = []  # (track, forum_id)
    for track, url in urls:
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            seen = set()

            # Method 1: anchor tags with forum links (relative or absolute)
            anchors = soup.find_all("a", href=True)
            anchor_total = len(anchors)
            anchor_matched = 0
            for a in anchors:
                href = a.get("href", "")
                if "forum?id=" in href:
                    anchor_matched += 1
                    fid = href.split("id=")[-1].split("&")[0].split("#")[0]
                    if fid and fid not in seen:
                        seen.add(fid)
                        forum_ids.append((track, fid))

            # Method 2: elements carrying data-note-id attributes
            data_nodes = soup.find_all(attrs={"data-note-id": True})
            data_matched = 0
            for el in data_nodes:
                fid = el.get("data-note-id")
                if fid and fid not in seen:
                    seen.add(fid)
                    forum_ids.append((track, fid))
                    data_matched += 1

            # Method 3: scan scripts for forum ids or generic id patterns
            script_matched = 0
            scripts = soup.find_all("script")
            for script in scripts:
                txt = script.string
                if not txt:
                    continue
                # Look for explicit forum links
                for m in re.findall(r"forum\?id=([A-Za-z0-9_\-]+)", txt):
                    fid = m.strip()
                    if fid and fid not in seen:
                        seen.add(fid)
                        forum_ids.append((track, fid))
                        script_matched += 1
                # As a fallback, generic JSON-like id fields (filter short ones)
                for m in re.findall(r'"id"\s*:\s*"([A-Za-z0-9_\-]{6,})"', txt):
                    fid = m.strip()
                    if fid and fid not in seen:
                        seen.add(fid)
                        forum_ids.append((track, fid))
                        script_matched += 1

            print(
                f"  {track}: anchors {anchor_total}, matched {anchor_matched}; data-note-id {data_matched}; script ids {script_matched}; unique forums {len(seen)}"
            )
        except Exception as e:
            print(f"  Warning: failed to scrape {url}: {e}")
            continue
    # Build paper stubs with forum id and track
    papers: List[Dict] = []
    for track, fid in forum_ids:
        papers.append({"forum": fid, "id": fid, "track": track})
    return papers


def _hydrate_papers_from_forums(session: requests.Session, forums: List[Dict]) -> List[Dict]:
    """Fetch title/abstract for each forum id via API; fallback to page scraping if needed."""
    api = "https://api.openreview.net/notes"
    base = "https://openreview.net"
    hydrated: List[Dict] = []
    for stub in forums:
        fid = stub.get("forum") or stub.get("id")
        title = ""
        abstract = ""
        venue = ""
        venueid = ""
        # Try by ids= or id=, then forum=
        data = None
        for params in (
            {"id": fid},
            {"ids": fid},
            {"forum": fid, "limit": 10},
        ):
            try:
                r = session.get(api, params=params, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    break
            except Exception:
                continue
        note = None
        if data:
            # notes may be under key 'notes'
            notes = data.get("notes") if isinstance(data, dict) else None
            if isinstance(notes, list):
                # pick the one whose id == forum (root submission)
                for n in notes:
                    if n.get("id") == fid or n.get("forum") == fid and not n.get("replyto"):
                        note = n
                        break
                if note is None and notes:
                    note = notes[0]
        if note:
            content = note.get("content", {})
            title = content.get("title", "")
            abstract = content.get("abstract", "")
            venue = str(content.get("venue", ""))
            venueid = str(content.get("venueid", ""))
        else:
            # Fallback: scrape forum page for title/abstract
            try:
                page = session.get(urljoin(base, f"/forum?id={fid}"), timeout=30)
                page.raise_for_status()
                soup = BeautifulSoup(page.content, "html.parser")
                # Title often in h2 or h1 tags with class 'note_content_title'
                h = soup.find(["h1", "h2"], class_=re.compile("title|note", re.I)) or soup.find(
                    ["h1", "h2"]
                )
                if h:
                    title = h.get_text(strip=True)
                # Abstract often present in a div with label 'Abstract'
                # Find label span or strong containing 'Abstract'
                abstract_label = soup.find(text=re.compile(r"^\s*Abstract\s*:?\s*$", re.I))
                if abstract_label:
                    # Abstract text may be in parent/next sibling
                    parent = abstract_label.parent
                    if parent and parent.next_sibling:
                        abstract = parent.next_sibling.get_text(strip=True)
                if not abstract:
                    # heuristic: look for element with class containing 'abstract'
                    abs_div = soup.find(attrs={"class": re.compile("abstract", re.I)})
                    if abs_div:
                        abstract = abs_div.get_text(strip=True)
            except Exception:
                pass
        hydrated.append(
            {
                "id": fid,
                "forum": fid,
                "title": title,
                "abstract": abstract,
                "venue": venue,
                "venueid": venueid,
                "track": stub.get("track", ""),
            }
        )
        time.sleep(0.2)
    return hydrated


def call_gemini(client: GeminiClient, prompt: str) -> str:
    """Call Gemini via wrapper and return raw text."""
    return client.generate_text(prompt)


def score_paper(
    model: GeminiClient,
    skills_prompt: str,
    paper: Dict,
) -> Dict:
    """Ask Gemini which skills guide the paper."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    prompt = f"""
You are checking whether a research paper is guided by any skills from an encyclopedia.

Skills:
{skills_prompt}

Paper:
- Title: {title}
- Abstract: {abstract}

Instructions:
- Identify which skills clearly apply based on title and abstract.
- Respond ONLY in JSON with keys: guided (boolean), matched_skills (array of skill names).
- Set guided=true when at least one skill is relevant.
- Use only skill names provided in the Skills list.
"""
    raw = call_gemini(model, prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage by finding JSON block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError(f"Failed to parse Gemini JSON: {raw}")


def main():
    parser = argparse.ArgumentParser(
        description="Check ICLR Accept papers (oral/spotlight/poster) for skill guidance using Gemini"
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-pro-preview",
        help="Gemini model name (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        required=True,
        help="Path to skills encyclopedia (JSON preferred)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="ICLR conference year (default: 2024)",
    )
    parser.add_argument(
        "--accept-oral",
        action="store_true",
        help="Include Accept (Oral) papers",
    )
    parser.add_argument(
        "--accept-spotlight",
        action="store_true",
        help="Include Accept (Spotlight) papers",
    )
    parser.add_argument(
        "--accept-poster",
        action="store_true",
        help="Include Accept (Poster) papers",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers (for quick tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="guided_oral_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between Gemini calls (default: 0.5)",
    )

    args = parser.parse_args()

    api_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is required. Provide --gemini-key or set GEMINI_API_KEY.")

    skills, skills_prompt = load_skills(args.encyclopedia)
    model = GeminiClient(api_key=api_key, model_name=args.gemini_model)

    papers = fetch_accept_tracks(
        args.year,
        max_papers=args.max_papers,
        accept_oral=args.accept_oral,
        accept_spotlight=args.accept_spotlight,
        accept_poster=args.accept_poster,
    )
    if not papers:
        print("No Accept (Oral) papers found.")
        return

    results = []
    guided_count = 0

    for idx, paper in enumerate(papers, 1):
        track_label = paper.get("track", "")
        print(
            f"[{idx}/{len(papers)}] Checking ({track_label}): {paper.get('title', '')[:80]}"
        )
        try:
            verdict = score_paper(model, skills_prompt, paper)
            guided = bool(verdict.get("guided"))
            matched = verdict.get("matched_skills") or []
        except Exception as exc:
            print(f"  Gemini error: {exc}")
            guided = False
            matched = []
            verdict = {"guided": guided, "matched_skills": matched}

        if guided:
            guided_count += 1

        results.append(
            {
                "id": paper.get("id"),
                "forum": paper.get("forum"),
                "title": paper.get("title", ""),
                "guided": guided,
                "matched_skills": matched,
                "track": paper.get("track", ""),
                "venue": paper.get("venue", ""),
                "venueid": paper.get("venueid", ""),
            }
        )

        time.sleep(max(args.sleep, 0))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(papers)
    print(f"\nGuided papers: {guided_count} out of {total}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
