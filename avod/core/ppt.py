# 首先爬取类别链接并创建大类文件夹
import requests
import os
from bs4 import BeautifulSoup as bs

URL = 'http://www.1ppt.com/'
FILE = 'f:\\python program\\ppt'


# 获取网页信息
def get_html(url):
    html = requests.get(url)
    html.encoding = 'gb2312'
    soup = bs(html.text, 'lxml')
    return soup


# 创建新的文件夹
def creatFile(element):
    path = FILE
    title = element
    new_path = os.path.join(path, title)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    return new_path


def main():
    content = get_html(URL)
    div = content.find('div', {'id': 'navMenu'})  # 定位到链接所在位置
    li_all = div.find_all('li')
    with open('f:\\python program\\ppt\\url.txt', 'w') as f:
        for li in li_all:
            li_a = li.find('a')
            link = URL + li_a['href']
            name = li_a.text
            creatFile(name)
            f.write(name + ';' + link + '\n')
    print('结束！')


if __name__ == "__main__":
    main()
