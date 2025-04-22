import time
import json
import random
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from googlesearch import search # pip install googlesearch-python

# --- Constants ---
INPUT_CSV = "top_1000_twitch.csv"
OUTPUT_JSON = "wikitest.json"
BAD_URL_SUBSTRINGS = [
    "facebook.com", "twitter.com", "youtube.com", "instagram.com",
    "twitch.tv", "reddit.com", "tiktok.com", "discord.com",
    "google.com", "amazon.com", "wikidata.org", "spotify.com",
    "https://en.wikipedia.org/wiki/Twitch_(service)",
    "https://en.wikipedia.org/wiki/List_of_most-followed_Twitch_channels",
    "https://en.wikipedia.org/wiki/The_Streamer_Awards","worldoftanks", 
    "https://es.wikipedia.org/wiki/Twitch"
]
MIN_CONTENT_LENGTH_WIKI = 100   # Min chars for wiki content
MIN_CONTENT_LENGTH_GENERIC = 200 # Min chars for other content

# --- Helper Functions ---
def is_bad_url(url):
    if url in BAD_URL_SUBSTRINGS:
        return True
    """Skip non-encyclopedic, social domains, or known bad pages/paths."""
    url_lower = url.lower()
    for bad in BAD_URL_SUBSTRINGS:
        if bad in url_lower:
            return True
    # Additional check for x.com (Twitter)
    if "x.com" in url_lower:
        return True
    return False

def format_streamer_name(name):
    """Format streamer name for display."""
    return name.strip().replace("_", " ").title()

def fetch_page_content(url, timeout=15):
    """Fetch HTML with retries and basic checks."""
    headers = {'User-Agent': random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
    ])}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            if is_bad_url(r.url) or 'text/html' not in r.headers.get('content-type', '').lower():
                return ""
            return r.text
        except requests.exceptions.RequestException:
            if attempt < 2: 
                time.sleep(random.uniform(1, 3))
    return ""

def clean_content(text):
    """General text cleanup."""
    if not text: 
        return ""
    
    # Remove wiki artifacts
    text = re.sub(r'\[edit\]|\[ソースを編集\]|\[изменить\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]|\[citation needed\]|\[\w+\?\]', '', text)
    
    # Remove duplicates
    lines = text.split('\n')
    seen_lines = set()
    unique_lines = [line for line in lines if not (line.strip() and line.strip().lower() in seen_lines) 
                   or not seen_lines.add(line.strip().lower())]
    text = '\n'.join(unique_lines)
    
    # Remove footers
    wiki_bottom_markers = ["categories:", "category:", "see also", "external links", 
                          "references", "notes", "gallery", "bibliography", "further reading"]
    lines = text.split('\n')
    filtered_lines = []
    in_footer_section = False
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        line_strip = line.strip()
        line_lower = line_strip.lower()
        is_potential_header = line_strip and (line_strip.isupper() or line_strip.startswith("==")) and len(line_strip) < 50
        
        if is_potential_header and any(marker in line_lower for marker in wiki_bottom_markers):
            in_footer_section = True
            continue
            
        if in_footer_section:
            if is_potential_header and not any(marker in line_lower for marker in wiki_bottom_markers):
                in_footer_section = False
                filtered_lines.append(line)
            else:
                continue
        else:
            filtered_lines.append(line)
    
    text = '\n'.join(reversed(filtered_lines))
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*[-•]\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

# --- Content Cleaners ---
def clean_wiki_content(html):
    """Extract and clean text from a Wikipedia page."""
    if not html: 
        return ""
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove non-content elements
    selectors_to_remove = [
        'sup', 'table', '.infobox', '.navbox', '.metadata',
        '.mw-editsection', 'style', 'script', '.references', 
        '#toc', '.toc', '.catlinks', '#siteSub', '.printfooter'
    ]
    for selector in selectors_to_remove:
        for tag in soup.select(selector): 
            tag.decompose()
    
    # Find main content
    main = soup.find('div', id='mw-content-text')
    if main: 
        main = main.find('div', class_='mw-parser-output') or main
    if not main: 
        main = soup.body
    if not main: 
        return ""
    
    # Extract text
    parts = []
    for el in main.find_all(['h2', 'h3', 'h4', 'p', 'li'], recursive=True):
        if el.find_parent(['table', '.infobox', '.navbox']): 
            continue
            
        txt = re.sub(r'\s+', ' ', el.get_text()).strip()
        if not txt or len(txt) < 5: 
            continue
        if txt.lower().startswith(("file:", "image:")): 
            continue
            
        if el.name in ('h2', 'h3', 'h4'):
            clean_heading = txt.replace('[edit]', '').strip()
            if clean_heading: 
                parts.append(f"\n\n{clean_heading.upper()}\n{'-'*len(clean_heading)}")
        elif el.name == 'li':
            if txt: 
                parts.append(f"• {txt}")
        else:
            if txt: 
                parts.append(txt)
    
    return clean_content("\n".join(parts).strip())

def clean_liquipedia_content(html):
    """Extract and clean text from a Liquipedia page."""
    if not html: 
        return ""
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove non-content elements (wiki + liquipedia specific)
    selectors_to_remove = [
        'sup', 'table', '.infobox', '.navbox', '.metadata',
        '.mw-editsection', 'style', 'script', '.references', 
        '#toc', '.toc', '.catlinks', '#siteSub', '.printfooter',
        '.fo-ntt-esports-infobox-wrapper', '.tabs', '.bracket', 
        '.matchlist', '.teamcard'
    ]
    for selector in selectors_to_remove:
        for tag in soup.select(selector): 
            tag.decompose()
    
    # Find main content
    main = soup.find('div', id='mw-content-text')
    if main: 
        main = main.find('div', class_='mw-parser-output') or main
    if not main: 
        main = soup.body
    if not main: 
        return ""
    
    # Extract text
    parts = []
    for el in main.find_all(['h2', 'h3', 'h4', 'p', 'li'], recursive=True):
        if el.find_parent(['table', '.bracket', '.matchlist', '.teamcard']): 
            continue
            
        txt = re.sub(r'\s+', ' ', el.get_text()).strip()
        if not txt or len(txt) < 5: 
            continue
        if txt.lower().startswith(("file:", "image:", "settings")): 
            continue
            
        if el.name in ('h2', 'h3', 'h4'):
            clean_heading = txt.replace('[edit]', '').strip()
            if clean_heading: 
                parts.append(f"\n\n{clean_heading.upper()}\n{'-'*len(clean_heading)}")
        elif el.name == 'li':
            if txt: 
                parts.append(f"• {txt}")
        else:
            if txt: 
                parts.append(txt)
    
    return clean_content("\n".join(parts).strip())

def clean_fandom_content(html):
    """Extract and clean text from a Fandom/Wikia page."""
    if not html: 
        return ""
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove non-content elements
    selectors_to_remove = [
        'sup', 'table', 'aside', 'nav', 'footer', 'style', 'script',
        '.fandom-sticky-header', '.page-header__actions', '.page-footer',
        '.wiki-rail', '.ads-container', '.portable-infobox', 
        '#toc', '.toc', '.gallery'
    ]
    for selector in selectors_to_remove:
        for tag in soup.select(selector): 
            tag.decompose()
    
    # Find main content
    main = (soup.find('div', class_='mw-parser-output') or 
            soup.find('article', id='wiki-content') or 
            soup.find('div', id='content') or 
            soup.body)
    if not main: 
        return ""
    
    # Extract text
    parts = []
    page_title_tag = soup.find('h1', class_='page-header__title')
    page_title_text = page_title_tag.get_text().strip() if page_title_tag else ""
    
    for el in main.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li'], recursive=True):
        if el.find_parent(['table', 'aside', '.portable-infobox']): 
            continue
            
        txt = re.sub(r'\s+', ' ', el.get_text()).strip()
        if not txt or len(txt) < 5: 
            continue
        if txt.lower().startswith(("file:", "image:", "gallery:")): 
            continue
        if txt.lower() in ["edit", "view source", "history"]: 
            continue
            
        if el.name.startswith('h'):
            clean_heading = txt.replace('[edit]', '').strip()
            if clean_heading:
                if el.name == 'h1' and clean_heading == page_title_text:
                    parts.append(f"{clean_heading.upper()}\n{'='*len(clean_heading)}")
                else:
                    parts.append(f"\n\n{clean_heading.upper()}\n{'-'*len(clean_heading)}")
        elif el.name == 'li':
            if txt: 
                parts.append(f"• {txt}")
        else:
            if txt: 
                parts.append(txt)
    
    return clean_content("\n".join(parts).strip())

def extract_generic_content(html):
    """Extract meaningful text from any webpage."""
    if not html: 
        return ""
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove non-content elements
    selectors_to_remove = [
        'script', 'style', 'nav', 'header', 'footer', 'iframe',
        'aside', '.sidebar', 'form', 'button', '.menu',
        '.social-links', '.comments', '.pagination',
        'div[class*="cookie"]', 'div[class*="ad"]'
    ]
    for selector in selectors_to_remove:
        for tag in soup.select(selector): 
            tag.decompose()
    
    # Find main content
    main_content = (soup.find('main') or 
                   soup.find('article') or 
                   soup.find('div', role='main') or 
                   soup.find('div', id='content') or 
                   soup.find('div', class_='content') or 
                   soup.body)
    if not main_content: 
        return ""
    
    # Extract text
    content_parts = []
    skip_strings = ['subscribe', 'log in', 'sign up', 'cookie policy', 
                   'terms of service', 'advertisement', 'share this']
    
    for el in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'], recursive=True):
        if el.find_parent(['nav', 'header', 'footer', 'aside', 'form', '.menu']): 
            continue
            
        text = re.sub(r'\s+', ' ', el.get_text()).strip()
        if len(text) < MIN_CONTENT_LENGTH_GENERIC / 8: 
            continue
            
        text_lower = text.lower()
        if any(kw in text_lower for kw in skip_strings): 
            continue
            
        if el.name.startswith('h'):
            content_parts.append(f"\n\n{text.upper()}\n{'-'*len(text)}")
        elif el.name == 'li':
            content_parts.append(f"• {text}")
        else:
            content_parts.append(text)
    
    # Check length requirement
    joined_content = "\n".join(content_parts).strip()
    final_content = clean_content(joined_content)
    return final_content if len(final_content) >= MIN_CONTENT_LENGTH_GENERIC else ""

# --- CSV Reader ---
def read_streamers_from_csv():
    """Reads streamer names from the 'Name' column of the input CSV."""
    try:
        df = pd.read_csv(INPUT_CSV)
        if 'Name' in df.columns:
            names = df['Name'].dropna().astype(str).unique().tolist()
            print(f"Read {len(names)} unique names from 'Name' column in {INPUT_CSV}")
            return names
        else:
            print(f"ERROR: 'Name' column not found in {INPUT_CSV}")
            return []
    except Exception as e:
        print(f"Error reading CSV file '{INPUT_CSV}': {e}")
        return []

# --- Core Processing Function ---
def process_streamer(streamer_name, max_results=5):
    """Finds and processes the best wiki/web page for a streamer."""
    print(f"\nProcessing {streamer_name}...")
    wiki_data = None
    tried_urls = set()

    # 1. Initial Wikipedia-focused search (only search)
    initial_query = f"{streamer_name} twitch wikipedia"
    print(f" → Search: {initial_query}")
    try:
        initial_results = list(search(initial_query, num_results=max_results, lang='en', sleep_interval=1))
    except Exception as e:
        print(f" → Search Error: {e}")
        initial_results = []
    
    # 2. Check if first result is Wikipedia
    if initial_results and 'wikipedia.org/wiki/' in initial_results[0] and not is_bad_url(initial_results[0]):
        url = initial_results[0]
        print(f" → First result is Wikipedia: {url}")
        html = fetch_page_content(url)
        tried_urls.add(url)
        
        content = clean_wiki_content(html) if html else ""
        if content and len(content) >= MIN_CONTENT_LENGTH_WIKI:
            wiki_lang = url.split('://')[1].split('.wikipedia')[0]
            wiki_data = {
                'content': content, 
                'url': url, 
                'language': wiki_lang, 
                'source': 'Wikipedia'
            }
            print(f" → Success (Wikipedia)")

    # 3. Check remaining results for Liquipedia/Fandom if no Wikipedia
    if not wiki_data and initial_results:
        start_index = 0 if initial_results and initial_results[0] in tried_urls else 0
        print(f" → No Wikipedia match. Scanning remaining results for other sources...")
        
        for i in range(start_index, min(len(initial_results), 5)):
            url = initial_results[i]
            if url in tried_urls or is_bad_url(url):
                continue

            source_type = None
            cleaner_func = None
            lang = 'en'
            
            # Determine source type and cleaner
            if 'liquipedia.net' in url:
                source_type = 'Liquipedia'
                cleaner_func = clean_liquipedia_content
                print(f" → Found potential Liquipedia: {url}")
            elif ('fandom.com' in url or 'wikia.com' in url or 'wikitubia.com' in url):
                source_type = 'Fandom/Wikia'
                cleaner_func = clean_fandom_content
                print(f" → Found potential Fandom/Wikia: {url}")
            elif 'famousbirthdays.com' in url:
                source_type = 'Famous Birthdays'
                cleaner_func = extract_generic_content
                print(f" → Found Famous Birthdays: {url}")
            elif 'gamepedia.com' in url:
                source_type = 'Gamepedia'
                cleaner_func = clean_fandom_content  # Use same cleaner as Fandom
                print(f" → Found potential Gamepedia: {url}")
            elif 'streamerwiki.com' in url:
                source_type = 'StreamerWiki'
                cleaner_func = extract_generic_content
                print(f" → Found potential StreamerWiki: {url}")
            elif 'twitchpedia.com' in url:
                source_type = 'TwitchPedia'
                cleaner_func = extract_generic_content
                print(f" → Found potential TwitchPedia: {url}")
            else:
                # Generic handler for other sites
                source_type = 'Generic'
                cleaner_func = extract_generic_content
                print(f" → Trying generic extraction: {url}")

            if source_type and cleaner_func:
                html = fetch_page_content(url)
                tried_urls.add(url)
                content = cleaner_func(html) if html else ""
                
                min_length = MIN_CONTENT_LENGTH_WIKI if source_type in ['Liquipedia', 'Fandom/Wikia'] else MIN_CONTENT_LENGTH_GENERIC
                
                if content and len(content) >= min_length:
                    wiki_data = {
                        'content': content, 
                        'url': url, 
                        'language': lang, 
                        'source': source_type
                    }
                    print(f" → Success ({source_type})")
                    break

    # Final result
    if not wiki_data:
        print(" → FAILED to retrieve any suitable page.")
        return {
            'streamer': streamer_name,
            'formatted_name': format_streamer_name(streamer_name),
            'wikipedia_summary': "Failed to retrieve page.",
            'link': "",
            'source': ""
        }

    return {
        'streamer': streamer_name,
        'formatted_name': format_streamer_name(streamer_name),
        'content': wiki_data['content'],
        'url': wiki_data['url'],
        'language': wiki_data['language'],
        'source': wiki_data['source'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

# --- Main Execution ---
def main():
    streamers = read_streamers_from_csv()
    if not streamers:
        print("No streamers to process. Exiting.")
        return

    # Load existing data if available
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing entries from {OUTPUT_JSON}")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No valid existing data found at {OUTPUT_JSON}. Starting fresh.")
        existing = {}

    processed_in_session = 0
    total_streamers = len(streamers)
    start_time = time.time()

    # Process each streamer
    for i, name in enumerate(streamers, 1):
        if name in existing:
            continue

        processed_in_session += 1
        result = process_streamer(name)

        if result:
            existing[name] = result
            try:
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                print(f" ✔️ {i}/{total_streamers} Saved {name} ({result.get('source', 'No Source')})")
            except IOError as e:
                print(f" ❌ ERROR saving data for {name}: {e}")
        else:
            print(f" ❌ {i}/{total_streamers} Failed for {name}")

        # Delay between requests to avoid rate limiting
        delay = random.uniform(2.0, 5.0)
        if processed_in_session > 0 and processed_in_session % 10 == 0:
            delay += random.uniform(2.0, 4.0)
            print(f" → Adding extra delay...")
        print(f" → Waiting {delay:.1f}s...")
        time.sleep(delay)

    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--------------------")
    print("Processing Complete.")
    print(f"Total streamers in CSV: {total_streamers}")
    print(f"Total entries in JSON: {len(existing)}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Data saved to: {OUTPUT_JSON}")
    print("--------------------")

if __name__ == "__main__":
    try:
        pd.read_csv(INPUT_CSV, nrows=1)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input file '{INPUT_CSV}' not found.")
        exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read input file '{INPUT_CSV}': {e}")
        exit(1)
    main()