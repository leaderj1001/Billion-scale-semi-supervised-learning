from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
import requests
import os
import time

chrome_driver = 'chromedriver_win.exe'
keyword = 'cat'
save_dir = './test'

if os.path.isdir(save_dir) is False:
    os.makedirs(save_dir)

browser = webdriver.Chrome(chrome_driver)
browser.get('https://www.bing.com/images/search?q=' + keyword)

body = browser.find_element_by_tag_name('body')
for i in range(100):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

keys = browser.find_elements_by_class_name('mimg')
links = []
for key in keys:
    links.append(key.get_attribute('src'))

browser.close()

for i, link in enumerate(links):
    if link.find('https') < 0:
        print(link)
        continue
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    img.save(save_dir + '/' + str(i) + '.jpg')
