#!/usr/bin/env python3
"""
ReadtheDocs Scraper

A module for scraping content from ReadtheDocs documentation sites.
This module extracts all textual content from a ReadtheDocs site and saves it
to text files for use in downstream RAG (Retrieval Augmented Generation) pipelines.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)


class ReadTheDocsParser:
    """Parser for ReadtheDocs HTML content."""

    def __init__(self):
        """Initialize the parser."""
        self.content_selectors = [
            "div.document",  # Standard ReadtheDocs content div
            "div[role='main']",  # Alternative content div
            "main",  # HTML5 main element
            "article",  # HTML5 article element
            "div.body",  # Another common content div
        ]
        
        self.ignore_selectors = [
            "div.sphinxsidebar",  # Sidebar
            "footer",  # Footer
            "nav",  # Navigation
            "div.header",  # Header
            "div.related",  # Related links
            "div.breadcrumbs",  # Breadcrumbs
            "div.sourcelink",  # Source link
            "div.highlight-default",  # Code blocks (we'll handle these separately)
            "div.admonition",  # Admonitions (notes, warnings, etc.)
        ]

    def extract_content(self, html_content: str, url: str) -> Dict[str, str]:
        """
        Extract the main content from a ReadtheDocs HTML page.
        
        Args:
            html_content: The HTML content of the page
            url: The URL of the page
            
        Returns:
            A dictionary containing the title and content of the page
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract the title
        title = soup.title.string if soup.title else ""
        title = title.replace(" — ", " - ").strip()
        
        # Try to find a more specific title
        if soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        
        # Find the main content
        content_element = None
        for selector in self.content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            logger.warning(f"Could not find main content in {url}")
            return {"title": title, "content": "", "url": url}
        
        # Remove elements to ignore
        for selector in self.ignore_selectors:
            for element in content_element.select(selector):
                element.decompose()
        
        # Extract text content
        content = self._extract_text_with_structure(content_element)
        
        return {"title": title, "content": content, "url": url}
    
    def _extract_text_with_structure(self, element) -> str:
        """
        Extract text from an element while preserving some structure.
        
        Args:
            element: The BeautifulSoup element to extract text from
            
        Returns:
            The extracted text with some structure preserved
        """
        if element.name in ["pre", "code"]:
            # For code blocks, preserve formatting
            return f"\n```\n{element.get_text()}\n```\n"
        
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # For headings, add appropriate markdown
            level = int(element.name[1])
            return f"\n{'#' * level} {element.get_text().strip()}\n"
        
        if element.name == "p":
            # For paragraphs, ensure they're separated by newlines
            return f"\n{element.get_text().strip()}\n"
        
        if element.name == "li":
            # For list items, add a bullet point
            return f"- {element.get_text().strip()}\n"
        
        if element.name == "table":
            # For tables, we'll just extract the text for now
            return f"\n{element.get_text().strip()}\n"
        
        # Recursively process child elements
        if hasattr(element, "children"):
            result = ""
            for child in element.children:
                if hasattr(child, "name"):
                    result += self._extract_text_with_structure(child)
                elif child.string and child.string.strip():
                    result += child.string
            return result
        
        # If it's just a string, return it
        return element.string if element.string else ""


class ReadTheDocsScraper:
    """Scraper for ReadtheDocs documentation sites."""

    def __init__(
        self, 
        base_url: str, 
        output_dir: str, 
        target_version: Optional[str] = None,
        delay: float = 0.5,
        max_pages: Optional[int] = None
    ):
        """
        Initialize the scraper.
        
        Args:
            base_url: The base URL of the ReadtheDocs site
            output_dir: The directory to save the scraped content to
            target_version: The specific version to scrape (e.g., 'en/latest')
            delay: The delay between requests in seconds
            max_pages: The maximum number of pages to scrape (None for unlimited)
        """
        self.base_url = self._sanitize_url(base_url)
        self.output_dir = Path(output_dir)
        self.target_version = target_version
        self.delay = delay
        self.max_pages = max_pages
        
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[str] = [self.base_url]
        self.parser = ReadTheDocsParser()
        
        # Create the output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract the domain for URL filtering
        self.domain = urlparse(self.base_url).netloc
        
        logger.info(f"Initialized scraper for {self.base_url}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.target_version:
            logger.info(f"Target version: {self.target_version}")
    
    def _sanitize_url(self, url: str) -> str:
        """
        Ensure the URL has the correct format.
        
        Args:
            url: The URL to sanitize
            
        Returns:
            The sanitized URL
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        if not url.endswith("/"):
            url = url + "/"
        
        return url
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid for scraping.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is valid, False otherwise
        """
        # Skip URLs that are not on the same domain
        if self.domain not in url:
            return False
        
        # Skip URLs that are not HTML pages
        if any(url.endswith(ext) for ext in [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg"]):
            return False
        
        # Skip URLs that contain anchors
        if "#" in url:
            url = url.split("#")[0]
        
        # Skip URLs that we've already visited
        if url in self.visited_urls:
            return False
        
        # If a target version is specified, only visit URLs that contain it
        if self.target_version and self.target_version not in url:
            return False
        
        return True
    
    def _save_content(self, content: Dict[str, str]) -> None:
        """
        Save the extracted content to a file.
        
        Args:
            content: A dictionary containing the title, content, and URL of the page
        """
        # Create a filename from the URL
        url_path = urlparse(content["url"]).path
        if url_path.endswith("/"):
            url_path = url_path + "index"
        
        # Replace slashes with underscores and remove the extension
        filename = url_path.strip("/").replace("/", "_")
        if not filename:
            filename = "index"
        
        # Add the .txt extension
        filename = f"{filename}.txt"
        
        # Create the full path
        filepath = self.output_dir / filename
        
        # Write the content to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {content['title']}\n")
            f.write(f"URL: {content['url']}\n")
            f.write(f"---\n\n")
            f.write(content["content"])
        
        logger.info(f"Saved content to {filepath}")
    
    def scrape(self) -> None:
        """
        Scrape the ReadtheDocs site.
        """
        logger.info(f"Starting to scrape {self.base_url}")
        
        pages_scraped = 0
        
        while self.urls_to_visit and (self.max_pages is None or pages_scraped < self.max_pages):
            # Get the next URL to visit
            url = self.urls_to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if url in self.visited_urls:
                continue
            
            logger.info(f"Scraping {url}")
            
            try:
                # Make the request
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Mark the URL as visited
                self.visited_urls.add(url)
                
                # Extract the content
                content = self.parser.extract_content(response.text, url)
                
                # Save the content
                self._save_content(content)
                
                # Find links to other pages
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    
                    # Skip empty links
                    if not href:
                        continue
                    
                    # Convert relative URLs to absolute URLs
                    if not href.startswith(("http://", "https://")):
                        href = urljoin(url, href)
                    
                    # Check if the URL is valid
                    if self._is_valid_url(href):
                        self.urls_to_visit.append(href)
                
                # Increment the counter
                pages_scraped += 1
                
                # Delay to be nice to the server
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        logger.info(f"Finished scraping. Visited {len(self.visited_urls)} pages.")
