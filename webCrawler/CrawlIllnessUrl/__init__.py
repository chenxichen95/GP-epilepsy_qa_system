import numpy as np
import time
import requests
from bs4 import BeautifulSoup
import json

def getIllnessUrl(headers):
    departmentLinks = np.load('./departmentLinks.npy')
    illnessLinks = []
    for curDepartment in departmentLinks:
        name = curDepartment[0]
        link = curDepartment[1]
        curIllnessLinks = []
        while True:
            print('scarpy link:{}'.format(link))
            while True:
                try:
                    response = requests.get(link, headers=headers, timeout=2)
                except:
                    print('{} response Error'.format(name))
                    time.sleep(3)
                else:
                    if response.status_code == 200:
                        break
                    else:
                        time.sleep(3)
            time.sleep(2)
            soup = BeautifulSoup(response.text, 'html5lib')
            re = soup.find_all('div', class_='cont ly-list-href')[0].find_all('a')
            for items in re:
                illnessName = items.text
                curLink = items.get('href')
                curIllnessLinks.append([illnessName, curLink])
            try:
                nextPage = soup.find_all('div', id='anpSelectData_Settings')[0].find_all('a')[-2]
            except IndexError:
                break
            if nextPage.get('disabled'):
                break
            else:
                link = nextPage.get('href')
        illnessLinks.append([name, curIllnessLinks])
    illnessLinksJson = json.dumps(illnessLinks)
    with open('./linksTemp/illnessLinks.json', 'w', encoding='utf-8') as fp:
        fp.write(illnessLinksJson)
