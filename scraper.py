"""
Scraper for ICLR 2023 Notable Top 5% Papers from OpenReview
Downloads PDFs and metadata for papers from the OpenReview website.
"""

import argparse
import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class OpenReviewScraper:
    def __init__(self, output_dir="data/papers/iclr23_top5"):
        self.output_dir = output_dir
        self.base_url = "https://openreview.net"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def _search_papers_by_title(self, title_filter):
        """
        Search for papers by title keyword using OpenReview API.
        This method searches across ALL papers by fetching all submissions and filtering client-side.
        This ensures we get all matching papers, not just the first 1000.

        Args:
            title_filter (str): Keyword to search for in paper titles (case-insensitive).

        Returns:
            list: List of papers matching the title filter.
        """
        papers = []
        api_url = "https://api.openreview.net/notes"

        # Search across all ICLR 2023 submissions
        # Use pagination to get all results
        offset = 0
        limit = 1000
        total_searched = 0
        total_found = 0

        print(
            f"Searching for papers with '{title_filter}' in title (case-insensitive)..."
        )
        print("This may take a while as we search through all ICLR 2023 submissions...")

        while True:
            try:
                params = {
                    "invitation": "ICLR.cc/2023/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = self.session.get(api_url, params=params, timeout=60)
                response.raise_for_status()

                if response.status_code == 200:
                    data = response.json()
                    notes = data.get("notes", [])

                    if not notes:
                        break  # No more papers

                    # Filter by title (case-insensitive)
                    batch_found = 0
                    for note in notes:
                        content = note.get("content", {})
                        title = content.get("title", "")

                        # Case-insensitive search - check if keyword appears anywhere in title
                        if title and title_filter.lower() in title.lower():
                            papers.append(note)
                            total_found += 1
                            batch_found += 1

                    total_searched += len(notes)

                    # Progress update
                    if batch_found > 0:
                        print(
                            f"  Searched {total_searched} papers, found {total_found} matching (latest batch: {batch_found})..."
                        )
                    elif total_searched % 5000 == 0:
                        print(
                            f"  Searched {total_searched} papers, found {total_found} matching so far..."
                        )

                    # Check if we got fewer results than limit (last page)
                    if len(notes) < limit:
                        break

                    offset += limit
                    time.sleep(0.5)  # Rate limiting between API calls
                else:
                    print(
                        f"API returned status {response.status_code}, stopping pagination"
                    )
                    break

            except requests.exceptions.RequestException as e:
                print(f"Error during search pagination: {e}")
                print(f"  Found {total_found} papers so far before error")
                break
            except Exception as e:
                print(f"Unexpected error during search: {e}")
                print(f"  Found {total_found} papers so far before error")
                break

        print(
            f"\nSearch complete: Found {len(papers)} papers matching '{title_filter}' in title (searched {total_searched} total papers)"
        )
        return papers

    def get_paper_list(self, title_filter=None, top5=False, top25=False, poster=False):
        """
        Fetch the list of papers from ICLR 2023

        Args:
            title_filter (str, optional): Filter papers by keyword in title (case-insensitive).
                                         If provided, searches across ALL papers, not just notable ones.
            top5 (bool): If True, only get notable top 5% papers
            top25 (bool): If True, only get notable top 25% papers
            poster (bool): If True, only get poster papers
        """
        # If title filter is provided, use search API to get ALL matching papers
        if title_filter:
            papers = self._search_papers_by_title(title_filter)
            if papers:
                return papers
            # If search API doesn't work, fall back to web scraping
            print("Search API didn't return results, trying web scraping...")
            return self._scrape_web_page(
                title_filter=title_filter, top5=top5, top25=top25, poster=poster
            )

        # If no title filter, use the original method
        # Try to get papers via API endpoint with pagination
        api_url = "https://api.openreview.net/notes"
        papers = []
        
        # Try with smaller limit first to avoid 400 errors
        offset = 0
        limit = 1000  # Reduced from 50000 to avoid API errors
        max_papers_to_fetch = 50000  # Maximum total papers to fetch
        
        try:
            print("Fetching papers from OpenReview API (this may take a while)...")
            while offset < max_papers_to_fetch:
                params = {
                    "invitation": "ICLR.cc/2023/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = self.session.get(api_url, params=params, timeout=60)
                
                if response.status_code != 200:
                    print(f"API returned status {response.status_code}, trying web scraping...")
                    break
                
                data = response.json()
                notes = data.get("notes", [])
                
                if not notes:
                    break  # No more papers
                
                # Filter for notable papers based on flags
                batch_count = 0
                for note in notes:
                    content = note.get("content", {})
                    venue = content.get("venue", "")
                    title = content.get("title", "")

                    # Check venue based on flags
                    should_include = False
                    if top5 and (
                        "Notable Top 5%" in venue
                        or ("Notable" in venue and "Top 5%" in venue)
                    ):
                        should_include = True
                    elif top25 and (
                        "Notable Top 25%" in venue
                        or ("Notable" in venue and "Top 25%" in venue)
                        or "notable top 25%" in venue.lower()
                    ):
                        should_include = True
                    elif poster and ("Poster" in venue or "ICLR 2023" in venue):
                        should_include = True
                    elif not top5 and not top25 and not poster:
                        # Default: get notable top 5%
                        if "Notable Top 5%" in venue or (
                            "Notable" in venue and "Top 5%" in venue
                        ):
                            should_include = True

                    if should_include:
                        papers.append(note)
                        batch_count += 1
                
                if batch_count > 0:
                    print(f"  Fetched {len(notes)} papers, found {batch_count} notable papers (total: {len(papers)})")
                
                if len(notes) < limit:
                    break  # Last batch
                
                offset += limit
                time.sleep(0.5)  # Rate limiting

            if papers:
                print(f"Found {len(papers)} papers via API")
        except requests.exceptions.RequestException as e:
            print(f"API method failed: {e}")
            print("Trying web scraping method...")
            papers = self._scrape_web_page(
                title_filter=title_filter, top5=top5, top25=top25, poster=poster
            )
        except Exception as e:
            print(f"Unexpected error in API call: {e}")
            print("Trying web scraping method...")
            papers = self._scrape_web_page(
                title_filter=title_filter, top5=top5, top25=top25, poster=poster
            )

        return papers

    def _scrape_web_page(
        self, title_filter=None, top5=False, top25=False, poster=False
    ):
        """
        Fallback method: scrape papers from multiple web pages

        Args:
            title_filter (str, optional): Filter papers by keyword in title (case-insensitive).
            top5 (bool): If True, scrape from notable top 5% papers
            top25 (bool): If True, scrape from notable top 25% papers
            poster (bool): If True, scrape from poster papers
        """
        # Build list of URLs based on flags
        # If no flags are set, use all URLs (backward compatibility)
        urls = []
        if top5 or (not top5 and not top25 and not poster):
            urls.append(
                (
                    "Notable Top 5%",
                    "https://openreview.net/group?id=ICLR.cc/2023/Conference#notable-top-5-",
                )
            )
        if top25 or (not top5 and not top25 and not poster):
            urls.append(
                (
                    "Notable Top 25%",
                    "https://openreview.net/group?id=ICLR.cc%2F2023%2FConference#notable-top-25-",
                )
            )
        if poster or (not top5 and not top25 and not poster):
            urls.append(
                (
                    "Poster",
                    "https://openreview.net/group?id=ICLR.cc%2F2023%2FConference#poster",
                )
            )

        all_papers = []
        seen_paper_ids = set()  # Track seen papers to avoid duplicates

        for url_name, url in urls:
            try:
                print(f"Scraping from {url_name}: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                # Try multiple selectors to find paper links
                # OpenReview uses various structures, so we'll try several
                paper_links = []
                
                # Method 1: Look for links with /forum?id=
                links1 = soup.find_all("a", href=re.compile(r"/forum\?id="))
                paper_links.extend(links1)
                
                # Method 2: Look for links with href containing paper IDs (forum pattern)
                links2 = soup.find_all("a", href=re.compile(r"forum.*id="))
                paper_links.extend(links2)
                
                # Method 3: Look for data attributes or IDs that might contain paper IDs
                # Some pages use data-note-id or similar attributes
                links3 = soup.find_all(attrs={"data-note-id": True})
                for elem in links3:
                    paper_id = elem.get("data-note-id")
                    if paper_id:
                        # Create a pseudo-link object
                        class PseudoLink:
                            def __init__(self, paper_id, title):
                                self.paper_id = paper_id
                                self.title = title
                            def get(self, key, default=None):
                                if key == "href":
                                    return f"/forum?id={self.paper_id}"
                                return default
                            def get_text(self, strip=False):
                                return self.title
                        title = elem.get_text(strip=True) or f"Paper_{paper_id}"
                        paper_links.append(PseudoLink(paper_id, title))
                
                # Method 4: Look for script tags that might contain JSON data with paper info
                scripts = soup.find_all("script")
                for script in scripts:
                    script_text = script.string
                    if script_text and "forum" in script_text and "id" in script_text:
                        # Try to extract paper IDs from JavaScript/JSON in script tags
                        # Look for patterns like "id": "..." or id="..."
                        ids = re.findall(r'["\']id["\']\s*:\s*["\']([^"\']+)["\']', script_text)
                        for paper_id in ids:
                            if len(paper_id) > 5:  # Filter out short IDs that are likely not paper IDs
                                class PseudoLink:
                                    def __init__(self, paper_id):
                                        self.paper_id = paper_id
                                    def get(self, key, default=None):
                                        if key == "href":
                                            return f"/forum?id={self.paper_id}"
                                        return default
                                    def get_text(self, strip=False):
                                        return f"Paper_{self.paper_id}"
                                paper_links.append(PseudoLink(paper_id))

                # Deduplicate paper_links by href
                seen_hrefs = set()
                unique_links = []
                for link in paper_links:
                    href = link.get("href", "")
                    if href and href not in seen_hrefs:
                        seen_hrefs.add(href)
                        unique_links.append(link)
                paper_links = unique_links

                for link in paper_links:
                    href = link.get("href", "")
                    paper_id = None
                    
                    # Extract paper ID from href
                    if "id=" in href:
                        paper_id = href.split("id=")[-1].split("&")[0].split("#")[0]
                    elif hasattr(link, 'paper_id'):
                        paper_id = link.paper_id
                    
                    if paper_id and paper_id not in seen_paper_ids:
                        title = link.get_text(strip=True) if hasattr(link, 'get_text') else (getattr(link, 'title', None) or f"Paper_{paper_id}")
                        # Apply title filter if specified
                        if title_filter:
                            if title_filter.lower() in title.lower():
                                all_papers.append(
                                    {
                                        "id": paper_id,
                                        "title": title,
                                        "url": urljoin(
                                            self.base_url, href if href.startswith("/") else f"/{href}"
                                        ),
                                    }
                                )
                                seen_paper_ids.add(paper_id)
                        else:
                            all_papers.append(
                                {
                                    "id": paper_id,
                                    "title": title,
                                    "url": urljoin(self.base_url, href if href.startswith("/") else f"/{href}"),
                                }
                            )
                            seen_paper_ids.add(paper_id)

                print(f"  Found {len(paper_links)} papers from this page")
                time.sleep(1)  # Rate limiting between pages

            except Exception as e:
                print(f"  Warning: Failed to scrape {url_name} ({url}): {e}")
                continue

        print(f"Total unique papers found: {len(all_papers)}")
        return all_papers

    def download_paper(self, paper_info):
        """Download a paper PDF given paper information"""
        # Try to get paper ID from various possible fields
        # The forum ID is what's used in OpenReview URLs
        paper_id = (
            paper_info.get("forum") or paper_info.get("id") or paper_info.get("number")
        )
        if not paper_id:
            print(
                f"Error: No paper ID found for paper: {paper_info.get('title', 'Unknown')}"
            )
            return None

        title = paper_info.get("content", {}).get(
            "title", paper_info.get("title", f"paper_{paper_id}")
        )

        # Clean title for filename
        safe_title = re.sub(r"[^\w\s-]", "", title)[:100]
        safe_title = re.sub(r"[-\s]+", "-", safe_title)

        # Use the standard OpenReview PDF URL format: https://openreview.net/pdf?id={paper_id}
        # This is the most reliable method as shown in the example: https://openreview.net/pdf?id=4-k7kUavAj
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"

        # Download PDF
        pdf_path = os.path.join(self.output_dir, f"{safe_title}_{paper_id}.pdf")

        try:
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            # Check if response is actually a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not content_type.startswith(
                "application/octet-stream"
            ):
                # Try alternative: check if we got HTML (error page) instead of PDF
                if response.headers.get("content-type", "").startswith("text/html"):
                    print(
                        f"Warning: Received HTML instead of PDF for {title}. The paper might not be publicly available."
                    )
                    return None

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file was written and has content
            if os.path.getsize(pdf_path) > 0:
                print(f"Downloaded: {title}")
                return pdf_path
            else:
                print(f"Error: Downloaded file is empty for {title}")
                os.remove(pdf_path)
                return None

        except requests.exceptions.HTTPError as e:
            print(f"Failed to download {title}: HTTP {e.response.status_code} - {e}")
            return None
        except Exception as e:
            print(f"Error downloading {title}: {e}")
            return None

    def save_metadata(self, papers):
        """Save paper metadata to JSON file"""
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {metadata_path}")

    def scrape_all(
        self, max_papers=None, title_filter=None, top5=False, top25=False, poster=False
    ):
        """
        Main method to scrape papers from OpenReview

        Args:
            max_papers (int, optional): Maximum number of papers to scrape.
                                       If None, scrapes all available papers.
            title_filter (str, optional): Filter papers by keyword in title (case-insensitive).
                                         If provided, only papers with this keyword in title will be included.
            top5 (bool): If True, scrape from notable top 5% papers
            top25 (bool): If True, scrape from notable top 25% papers
            poster (bool): If True, scrape from poster papers
        """
        print("Fetching paper list...")
        if title_filter:
            print(
                f"Searching for papers with '{title_filter}' in title (case-insensitive)..."
            )

        # Try API first, then fall back to web scraping with specified flags
        papers = self.get_paper_list(
            title_filter=title_filter, top5=top5, top25=top25, poster=poster
        )

        # If API doesn't work or returns no papers, try web scraping
        if not papers:
            print("No papers from API, trying web scraping...")
            papers = self._scrape_web_page(
                title_filter=title_filter, top5=top5, top25=top25, poster=poster
            )

        if not papers:
            print("No papers found. Please check the URL or API access.")
            return

        total_papers = len(papers)
        print(f"Found {total_papers} papers")

        # Save metadata (save all found papers before limiting)
        self.save_metadata(papers)

        # Limit papers if max_papers is specified
        if max_papers is not None and max_papers > 0:
            papers = papers[:max_papers]
            print(f"Limiting to {len(papers)} papers (requested: {max_papers})")

        # Download PDFs
        print(f"\nDownloading {len(papers)} papers...")
        downloaded = 0
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing paper...")
            result = self.download_paper(paper)
            if result:
                downloaded += 1
            time.sleep(1)  # Be respectful with rate limiting

        print(
            f"\nCompleted! Downloaded {downloaded}/{len(papers)} papers to {self.output_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape ICLR 2023 Notable Top 5% Papers from OpenReview"
    )
    parser.add_argument(
        "-n",
        "--num-papers",
        type=int,
        default=None,
        help="Number of papers to scrape (default: all available papers)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/papers/iclr23_top5",
        help="Output directory for downloaded papers (default: data/papers/iclr23_top5)",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default=None,
        help="Filter papers by keyword in title (case-insensitive). Example: -f 'diffusion'",
    )
    parser.add_argument(
        "--top5",
        action="store_true",
        help="Scrape from notable top 5% papers",
    )
    parser.add_argument(
        "--top25",
        action="store_true",
        help="Scrape from notable top 25% papers",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="Scrape from poster papers",
    )

    args = parser.parse_args()

    scraper = OpenReviewScraper(output_dir=args.output_dir)
    scraper.scrape_all(
        max_papers=args.num_papers,
        title_filter=args.filter,
        top5=args.top5,
        top25=args.top25,
        poster=args.poster,
    )
