from bs4 import BeautifulSoup
import requests

from duckduckgo_search import ddg


def paragraphs_from_search(question, max_results=2):
    results = ddg(question, region='wt-wt', max_results=max_results)
    paragraphs = list()
    for result in results:
        paragraphs.append(' '.join(scrape_url(result['href'])))
    return paragraphs


def scrape_url(url):
    result = requests.get(url)

    soup = BeautifulSoup(result.text, "html.parser")
    paragraphs = soup.find_all("p")
    paragraphs_ = list()
    for paragraph in paragraphs:
        paragraphs_.append(paragraph.text.strip())
    return paragraphs_
