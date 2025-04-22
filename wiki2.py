import time
import json
import requests
import random
import urllib.parse
import pandas as pd
from bs4 import BeautifulSoup
from googlesearch import search

# --- Constants ---
INPUT_CSV = "top_1000_streamers.csv"
OUTPUT_JSON = "wikitest.json"
BAD_URL_SUBSTRINGS = [
    "fandom.com", "wikia.com", "facebook.com", "twitter.com",
    "youtube.com", "instagram.com", "twitch.tv", "reddit.com",
    "tiktok.com", "discord.com", "google.com", "amazon.com",
    "/song)", "(song)", "/album)", "(album)", "/film)", "(film)"
]

# --- Helper Functions ---
def is_bad_url(url):
    """Check if URL should be skipped."""
    url_lower = url.lower()
    return any(bad in url_lower for bad in BAD_URL_SUBSTRINGS)

def format_streamer_name(name):
    """Clean streamer name formatting."""
    return name.strip().replace("_", " ").title()

def fetch_page_content(url, timeout=15):
    """Fetch raw HTML from a URL with retries."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    for _ in range(3):  # Retry up to 3 times
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Retry failed for {url}: {str(e)}")
            time.sleep(random.uniform(1, 3))
    return f"Error: Failed to fetch {url}"

def clean_wiki_content(html):
    """Extract clean text from Wikipedia HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove Wikipedia-specific junk
    for element in soup(['sup', 'table', 'div.hatnote', 'div.infobox',
                        'span.mw-editsection', 'div.navbox', 'style',
                        'script', 'link', 'meta', 'img', 'footer']):
        element.decompose()
    
    # Get main content
    content = soup.find('div', {'id': 'mw-content-text'}) or soup.body
    if not content:
        return "No content found"
    
    # Clean and structure text
    paragraphs = []
    for element in content.find_all(['p', 'h2', 'h3']):
        text = element.get_text().strip()
        if element.name in ['h2', 'h3']:
            paragraphs.append(f"\n\n{text.upper()}\n{'-'*len(text)}")
        elif text:
            paragraphs.append(text)
    
    return "\n".join(paragraphs) or "No readable content extracted"

# --- Wikipedia API Functions ---
def fetch_wikipedia_content(page_title, lang='en'):
    """Get complete Wikipedia article content through API."""
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'prop': 'extracts|info',
        'explaintext': True,
        'exsectionformat': 'plain',
        'inprop': 'url',
        'titles': page_title,
        'format': 'json'
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        
        for page in pages.values():
            return {
                'content': page.get('extract', ''),
                'url': page.get('fullurl', ''),
                'language': lang
            }
    except Exception as e:
        print(f"Wikipedia API error ({lang}): {e}")
    return None

# --- Streamer Verification ---
def is_streamer_page(url, streamer_name):
    """Verify the page is about the streamer and not something else."""
    # Skip obvious non-streamer pages
    if any(x in url.lower() for x in ['/song', '/album', '/film', '/music']):
        return False
        
    # Check title matches streamer name
    page_title = urllib.parse.unquote(url.split('/wiki/')[-1].replace('_', ' '))
    return (streamer_name.lower() in page_title.lower() and
            'disambiguation' not in page_title.lower())

# --- Core Processing ---
def read_streamers_from_csv():
    """Read streamers from specified CSV file."""
    try:
        df = pd.read_csv(INPUT_CSV)
        # Handle different CSV formats
        if 'Name' in df.columns:
            return df['Name'].dropna().unique().tolist()
        elif 'streamer' in df.columns:
            return df['streamer'].dropna().unique().tolist()
        else:
            return df.iloc[:, 1].dropna().unique().tolist()  # Assume 2nd column
    except Exception as e:
        print(f"CSV reading error: {e}")
        return []

def process_streamer(streamer_name):
    """Full processing pipeline for one streamer."""
    print(f"\nProcessing {streamer_name}...")
    
    # Step 1: Find Wikipedia page using optimized query
    query = f"{streamer_name} twitch wikipedia"
    wiki_url = None
    lang = 'en'
    
    for url in search(query, num_results=3, lang='en'):
        if is_bad_url(url) or not is_streamer_page(url, streamer_name):
            continue
            
        if '.wikipedia.org/wiki/' in url:
            lang = url.split('.wikipedia.org')[0].split('//')[-1]
            wiki_url = url
            break
        time.sleep(random.uniform(1, 2))
    
    if not wiki_url:
        print(f"No valid Wikipedia page found for {streamer_name}")
        return None
    
    # Step 2: Fetch content
    page_title = urllib.parse.unquote(wiki_url.split('/wiki/')[-1])
    wiki_data = fetch_wikipedia_content(page_title, lang)
    
    if not wiki_data or not wiki_data['content']:
        print(f"API failed, scraping directly: {wiki_url}")
        html = fetch_page_content(wiki_url)
        if not html.startswith("Error"):
            wiki_data = {
                'content': clean_wiki_content(html),
                'url': wiki_url,
                'language': lang
            }
        else:
            return None
    
    return {
        'streamer': streamer_name,
        'formatted_name': format_streamer_name(streamer_name),
        'content': wiki_data['content'],
        'url': wiki_data['url'],
        'language': lang,
        'source': 'Wikipedia',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    """Main execution function."""
    print(f"Reading streamers from {INPUT_CSV}...")
    streamers = read_streamers_from_csv()
    if not streamers:
        print("No streamers found in CSV!")
        return
    
    print(f"Found {len(streamers)} streamers to process")
    
    # Load existing data
    try:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"Resuming with {len(existing_data)} existing records")
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    
    # Process streamers in order
    for i, streamer in enumerate(streamers, 1):
        if streamer in existing_data:
            print(f"{i}/{len(streamers)}: Skipping {streamer} (already processed)")
            continue
            
        result = process_streamer(streamer)
        if result:
            existing_data[streamer] = result
            # Save after each successful processing
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            print(f"{i}/{len(streamers)}: Saved {streamer} to {OUTPUT_JSON}")
        
        # Respectful delay with progress tracking
        delay = random.uniform(3, 8)
        print(f"Waiting {delay:.1f}s...")
        time.sleep(delay)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()