#!/usr/bin/env python3
"""Attempt to fetch a Sofifa player page using a headless browser (Selenium).

This script uses webdriver-manager to install a ChromeDriver, then
launches Chrome in headless mode to access the page and dump the HTML or
status. It is exploratory; failures (no browser binary, Cloudflare, etc.)
will be printed.
"""
import sys

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("selenium or webdriver-manager not installed")
    sys.exit(1)

url = "https://sofifa.com/player/215616/david-jason-remeseiro-salgueiro/230040"

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=1920,1080")
# mimic a normal user agent
opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

print("Starting Chrome driver...")
from selenium.webdriver.chrome.service import Service
try:
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
except Exception as e:
    print("Failed to start Chrome driver:", e)
    sys.exit(1)

try:
    driver.get(url)
    print("HTTP status code via Selenium:", driver.execute_script("return document.readyState"))
    html = driver.page_source
    print("Page source length:", len(html))
    # print first 1k for context
    print(html[:1000])
    # extract stats via regex
    import re
    print('== regex stats ==')
    for m in re.finditer(r"(pace|shooting|passing|dribbling|defending|physic)[^\d]*(\d+)", html, flags=re.IGNORECASE):
        print(m.group(1), m.group(2))
    # attempt DOM extraction
    print('== DOM stats ==')
    try:
        elems = driver.find_elements('css selector', 'li.bp3-menu-item')
        for e in elems:
            text = e.text.strip()
            if any(k in text.lower() for k in ['pace','shooting','passing','dribbling','defending','physic']):
                print(text)
    except Exception as e:
        print('DOM query failed', e)
    # inspect script tags for embedded JS variables
    print('== inspect JS scripts ==')
    scripts = driver.find_elements('tag name','script')
    for s in scripts:
        txt = s.get_attribute('innerHTML')
        if 'OVERALL_RATING' in txt or 'pace' in txt.lower():
            print('script snippet:', txt[:1000])
            break
finally:
    driver.quit()
