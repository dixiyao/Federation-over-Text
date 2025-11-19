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

    def get_paper_list(self):
        """Fetch the list of notable top 5% papers from ICLR 2023"""
        # Try to get papers via API endpoint
        api_url = "https://api.openreview.net/notes"
        params = {
            "invitation": "ICLR.cc/2023/Conference/-/Blind_Submission",
            "details": "replyCount,invitation,original",
            "offset": 0,
            "limit": 1000,
            "sort": "number:asc",
        }

        papers = []
        try:
            response = self.session.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                notes = data.get("notes", [])

                # Filter for notable top 5% papers
                for note in notes:
                    # Check if paper is in notable top 5%
                    # This might need adjustment based on actual API response structure
                    if (
                        note.get("content", {}).get("venue")
                        == "ICLR 2023 Notable Top 5%"
                    ):
                        papers.append(note)
                    # Alternative: check for specific tags or fields
                    elif "notable" in str(note.get("content", {})).lower():
                        papers.append(note)

                # If no notable papers found, get all and filter by acceptance
                if not papers:
                    print(
                        "No notable papers found via API, trying alternative method..."
                    )
                    # Get all accepted papers
                    for note in notes:
                        content = note.get("content", {})
                        venue = content.get("venue", "")
                        if "ICLR 2023" in venue and (
                            "Notable" in venue or "Top 5%" in venue
                        ):
                            papers.append(note)
        except Exception as e:
            print(f"API method failed: {e}")
            print("Trying web scraping method...")
            papers = self._scrape_web_page()

        return papers

    def _scrape_web_page(self):
        """Fallback method: scrape papers from the web page"""
        url = "https://openreview.net/group?id=ICLR.cc/2023/Conference#notable-top-5-"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            papers = []
            # Find paper links - this selector may need adjustment
            paper_links = soup.find_all("a", href=re.compile(r"/forum\?id="))

            for link in paper_links:
                paper_id = (
                    link.get("href", "").split("id=")[-1]
                    if "id=" in link.get("href", "")
                    else None
                )
                if paper_id:
                    papers.append(
                        {
                            "id": paper_id,
                            "title": link.get_text(strip=True),
                            "url": urljoin(self.base_url, link.get("href", "")),
                        }
                    )

            return papers
        except Exception as e:
            print(f"Web scraping failed: {e}")
            return []

    def download_paper(self, paper_info):
        """Download a paper PDF given paper information"""
        paper_id = paper_info.get("id") or paper_info.get("number")
        title = paper_info.get("content", {}).get(
            "title", paper_info.get("title", f"paper_{paper_id}")
        )

        # Clean title for filename
        safe_title = re.sub(r"[^\w\s-]", "", title)[:100]
        safe_title = re.sub(r"[-\s]+", "-", safe_title)

        # Try to get PDF URL
        pdf_url = None
        if "pdf" in paper_info.get("content", {}):
            pdf_url = paper_info["content"]["pdf"]
        else:
            # Construct PDF URL
            pdf_url = f"https://openreview.net/pdf?id={paper_id}"

        # Download PDF
        pdf_path = os.path.join(self.output_dir, f"{safe_title}_{paper_id}.pdf")

        try:
            response = self.session.get(pdf_url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {title}")
                return pdf_path
            else:
                print(f"Failed to download {title}: HTTP {response.status_code}")
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

    def scrape_all(self, max_papers=None):
        """
        Main method to scrape notable top 5% papers

        Args:
            max_papers (int, optional): Maximum number of papers to scrape.
                                       If None, scrapes all available papers.
        """
        print("Fetching paper list...")
        papers = self.get_paper_list()

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

    args = parser.parse_args()

    scraper = OpenReviewScraper(output_dir=args.output_dir)
    scraper.scrape_all(max_papers=args.num_papers)
