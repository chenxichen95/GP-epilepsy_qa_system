import numpy as np
import requests
from bs4 import BeautifulSoup

def getDepartmentUrl(headers):
    url1 = 'http://ask.familydoctor.com.cn/jbk/1'
    response = requests.get(url1, headers=headers, allow_redirects=False, timeout=2)
    soup = BeautifulSoup(response.text, 'html5lib')
    re = soup.find('dd', id='parent-cate').find_all('a')
    typeLinkDict = []
    for item in re:
        keyName = item.text
        curLink = item.get('href')
        typeLinkDict.append([keyName, curLink])

    departmentLinks = np.array(typeLinkDict)
    np.save('./departmentLinks.npy', departmentLinks)
