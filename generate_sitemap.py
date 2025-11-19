#!/usr/bin/env python3
"""
Generate sitemap.xml for resources.iphysics.sa
"""
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

BASE_URL = "https://resources.iphysics.sa"
RESOURCES_DIR = Path(__file__).parent

# File extensions to include in sitemap
EXTENSIONS = {'.pdf', '.PDF', '.docx', '.DOCX', '.doc', '.DOC', '.pptx', '.PPTX'}

def get_all_files():
    """Get all resource files recursively."""
    files = []
    for root, dirs, filenames in os.walk(RESOURCES_DIR):
        # Skip hidden directories and git directory
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in filenames:
            if any(filename.endswith(ext) for ext in EXTENSIONS):
                filepath = Path(root) / filename
                files.append(filepath)
    return files

def get_url_from_path(filepath):
    """Convert file path to URL."""
    relative_path = filepath.relative_to(RESOURCES_DIR)
    # URL encode the path to handle spaces and special characters
    url_path = '/'.join(quote(part, safe='') for part in relative_path.parts)
    return f"{BASE_URL}/{url_path}"

def get_lastmod(filepath):
    """Get last modified date in ISO 8601 format."""
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

def generate_sitemap():
    """Generate sitemap.xml file."""
    files = get_all_files()

    # Sort files by path for consistency
    files.sort(key=lambda f: str(f))

    # Generate XML
    xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    for filepath in files:
        url = get_url_from_path(filepath)
        lastmod = get_lastmod(filepath)

        xml_lines.append('  <url>')
        xml_lines.append(f'    <loc>{url}</loc>')
        xml_lines.append(f'    <lastmod>{lastmod}</lastmod>')
        xml_lines.append('  </url>')

    xml_lines.append('</urlset>')

    # Write to sitemap.xml
    sitemap_path = RESOURCES_DIR / 'sitemap.xml'
    with open(sitemap_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))

    print(f"Sitemap generated successfully!")
    print(f"Total URLs: {len(files)}")
    print(f"Output: {sitemap_path}")

if __name__ == '__main__':
    generate_sitemap()
