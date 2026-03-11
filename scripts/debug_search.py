#!/usr/bin/env python3
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time, urllib.parse, re

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--disable-gpu')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')
opts.add_argument('--window-size=1920,1080')
opts.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=opts)

name = 'Harry Kane'
q = urllib.parse.quote(name)
url = f'https://sofifa.com/players?search={q}'
print('url', url)
driver.get(url)

for i in range(5):
    time.sleep(1)
    print('sleep', i)

html = driver.page_source
print('len html', len(html))
print(html[:2000])

links = driver.find_elements('css selector', 'a')
print('links count', len(links))
for a in links:
    href = a.get_attribute('href') or ''
    if '/player/' in href:
        print('player link', href, a.text)

driver.quit()
