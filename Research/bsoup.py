from bs4 import BeautifulSoup
import requests

"""
    Pull info from downloaded html file
"""
# with open('lxml - Processing XML and HTML with Python.html') as html_file:
#     soup = BeautifulSoup(html_file, 'lxml')

# print(soup.prettify())

# print(soup.title.text)

# print(soup.body.text)


"""
    Pull websource using url
"""
# webSource = requests.get('https://lxml.de/3.7/index.html#introduction').text
# soup = BeautifulSoup(webSource, 'lxml')
# print(soup.prettify()) 

"""
    Scrape for text
"""
# for article in soup.find_all('div'):
#     print("*****************************")
#     # print(article.prettify())
#     try:
#         print(article.find('p').text)
#     except Exception as e:
#         continue
#     print("*****************************")

