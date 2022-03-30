#!/usr/bin/env python
# coding: utf-8

# Write a python program to display all the header tags from wikipedia.org.

# In[1]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen('https://en.wikipedia.org')
bs = BeautifulSoup(html, "html.parser")
titles = bs.find_all(['h1', 'h2','h3','h4','h5','h6'])
print('List all the header tags :', *titles, sep='\n\n')


# In[3]:


from bs4 import BeautifulSoup
import requests
import re

# Download IMDB's Top 100 data
url = 'http://www.imdb.com/chart/top'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')

movies = soup.select('td.titleColumn')
ratings = [b.attrs.get('data-value') for b in soup.select('td.posterColumn span[name=ir]')]
year = [b.attrs.get('data-value') for b in soup.select('td.ratingColumn strong')]

imdb = []

# Store each item into dictionary (data), then put those into a list (imdb)
for index in range(0, len(movies)):
    # Seperate movie into: 'rating', 'title', 'year'
    movie_string = movies[index].get_text()
    movie = (' '.join(movie_string.split()).replace('.', ''))
    movie_title = movie[len(str(index))+1:-7]
    year = re.search('\((.*?)\)', movie_string).group(1)
    place = movie[:len(str(index))-(len(movie))]
    data = {"movie_title": movie_title,
            "year": year,
            "rating": ratings[index]}

    imdb.append(data)

for item in imdb:
    print(item['year'], '-', item['movie_title'],'Starring:', item['rating'])


# In[5]:


from bs4 import BeautifulSoup
import requests
import re

# Download IMDB's Top 100 indian data
url = 'https://www.imdb.com/india/top-rated-indian-movies/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')

movies = soup.select('td.titleColumn')
ratings = [b.attrs.get('data-value') for b in soup.select('td.posterColumn span[name=ir]')]
year = [b.attrs.get('data-value') for b in soup.select('td.ratingColumn strong')]

imdb = []

# Store each item into dictionary (data), then put those into a list (imdb)
for index in range(0, len(movies)):
    # Seperate movie into: 'rating', 'title', 'year'
    movie_string = movies[index].get_text()
    movie = (' '.join(movie_string.split()).replace('.', ''))
    movie_title = movie[len(str(index))+1:-7]
    year = re.search('\((.*?)\)', movie_string).group(1)
    place = movie[:len(str(index))-(len(movie))]
    data = {"movie_title": movie_title,
            "year": year,
            "rating": ratings[index]}

    imdb.append(data)

for item in imdb:
    print(item['year'], '-', item['movie_title'],'Starring:', item['rating'])


# In[ ]:




