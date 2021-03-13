python -m pip install requests

import requests
from bs4 import BeautifulSoup

url = "https://nytimes.com"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, 'html.parser')

# Retrieve all of the anchor tags
tags = soup.find_all('a')
for tag in tags:
    print(tag.get('href', None))