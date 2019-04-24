import time
import requests
from bs4 import BeautifulSoup
import json
import re
import threading
import pymongo

proxyPool = []
getProxyFlag = False
finishScrapyFlag = False

class myWorm(threading.Thread):
    def __init__(
            self, id=0, maxPage=100, type='all', department=None, curTable=None, useProxy=False, DB=None,
            headers=None, departmentDict=None
        ):
        super(myWorm, self).__init__()  # 调用父类的构造函数
        self.id = id
        self.department = department
        self.curTable = curTable
        self.type = type
        self.maxPage = maxPage
        self.useProxy = useProxy
        self.curIllness = None
        self.DB = DB
        self.headers = headers
        self.departmentDict = departmentDict
    def requestUrl(self, url):
        global sleepTime
        FailCount = 0
        FailCount2 = 0
        timeout = 2
        while True:
            try:
                response = requests.get(url, headers=self.headers, timeout=2.5)
            except:
                print('\twormID:{}, department:{} can not request. url:{}'.format(self.id, self.department, url))
                FailCount += 1
                time.sleep(sleepTime)
            else:
                if response.status_code == 200:
                    time.sleep(1)
                    return response
                else:
                    print('\twormID:{}, department:{} request fail, status_code:{}'.format(
                        self.id, self.department, response.status_code))
                    FailCount += 1
                    time.sleep(sleepTime)
            if FailCount == 20:
                print('\t\twormID:{},department:{} sleep 180s'.format(self.id, self.department))
                time.sleep(60)
                FailCount = 0
                FailCount2 += 1
                timeout = timeout + 0.5*FailCount2
            if FailCount2 == 15:
                FailCount2 = 0
                FailCount = 0
                return []
    def clearErrorLogOl(self):
        with open('./log_V2/{}_errorLog'.format(self.id), 'w', encoding='utf-8') as fp:
            fp.write('\n')
    def writeLog(self, str):
        with open('./log_V2/myLog_{}_V2.log'.format(self.department), 'a', encoding='utf-8') as fp:
            fp.write(str)
            fp.write('\n')
    def writeErrorLogOl(self, str):
        with open('./log_V2/{}_errorLog'.format(self.id), 'a', encoding='utf-8') as fp:
            fp.write(str)
            fp.write('\n')
    def requestUrlWithProxy(self, url, proxy):
        global proxyPool
        FailCount = 0
        self.clearErrorLogOl()
        while True:
            try:
                response = requests.get(url, headers=headers, proxies=proxy, timeout=10)
            except:
                proxy = proxyPool.pop(0)  # change proxy
                self.writeErrorLogOl('wormID:{} crawling {}-{} current url:{}:'
                                     'can not request'.format(self.id, self.department, self.curIllness, url))
                FailCount += 1
            else:
                if response.status_code == 200:
                    return response, proxy
                else:
                    proxy = proxyPool.pop(0)  # change proxy
                    FailCount += 1
                    self.writeErrorLogOl('wormID:{} crawling {}-{} current url:{}:'
                                         'request fail, status_code:{}'.format(
                        self.id, self.department, self.curIllness,url, response.status_code))
            if FailCount == 200:
                return []
    def getNextPageLink(self, soup):
        try:
            nextPage = soup.find_all('div', id='anpSelectData_Settings')[0].find_all('a')[-2]
        except IndexError:
            # only one page
            return []
        if nextPage.get('disabled'):
            return []
        else:
            # have next page , give up current page, turn to scrapy next page
            link = nextPage.get('href')
            return link
    def crawQALinks(self):
            # in department
            global illnessLink
            global proxyPool
            department = self.department
            while illnessLink:
                item2 = illnessLink.pop(0)
                # in illness
                illness = item2[0]
                self.curIllness = illness
                link = item2[1]
                proxy = proxyPool.pop(0)
                while True:
                    # get all QALinks in current illness
                    if self.useProxy:
                        response, proxy = self.requestUrlWithProxy(link, proxy)
                    else:
                        response = self.requestUrl(link)

                    if response:
                        soup = BeautifulSoup(response.text, 'html5lib')
                    else:
                        # can't get current link response, go to next page
                        print('wormID:{} crawling {}-{} current url:{}:'
                              'can not get current link response'.format(self.id, department, illness, link))
                        self.writeLog('wormID:{} crawling {}-{} current url:{}:'
                                 'can not get current link response'.format(self.id, department, illness, link))
                        break

                    try:
                        faq_list = soup.find_all('div', class_='cont faq-list')[0].find_all('dl')
                    except:
                        print('wormID:{} crawling {}-{} current url:{}:'
                              'can not find cont faq-list'.format(self.id, department, illness, link))
                        self.writeLog('wormID:{} crawling {}-{} current url:{}:'
                                 'can not find cont faq-list'.format(self.id, department, illness, link))
                        link = self.getNextPageLink(soup)
                        if link:
                            # have next page
                            continue
                        else:
                            # don't have next page
                            break

                    for item3 in faq_list:
                        try:
                            curData ={
                                'illness': illness,
                                'link': item3.find('a').get('href'),
                            }
                            self.curTable.insert(curData)
                        except:
                            print('wormID:{} crawling {}-{} current url:{}:'
                                  'can not write current data'.format(self.id, department, illness, link))
                            self.writeLog('wormID:{} crawling {}-{} current url:{}:'
                                     'can not write current data'.format(self.id, department, illness, link))
                            continue
                    print('wormID:{} crawling {}-{} current url:{} successfully'.format(
                        self.id, department, illness,link))
                    link = self.getNextPageLink(soup)
                    if link:
                        pass
                    else:
                        # don't have next page
                        print('wormID:{} crawling {}-{} current url:{}:'
                              'can not arrive maxPage:{}'.format(self.id, department, illness, link, self.maxPage))
                        self.writeLog('wormID:{} crawling {}-{} current url:{}:'
                                'can not arrive maxPage:{}'.format(self.id, department, illness, link, self.maxPage))
                        break
                    # only scrapy maxPage
                    curPageNum = re.search('page=[0-9]+', link)
                    curPageNum = curPageNum.group().split('=')[-1]
                    if int(curPageNum) >= self.maxPage:
                        print('wormID:{} crawling {}-{} current url:{}:'
                              'arrive maxPage:{}'.format(self.id, department, illness, link, self.maxPage))
                        self.writeLog('wormID:{} crawling {}-{} current url:{}:'
                              'arrive maxPage:{}'.format(self.id, department, illness, link, self.maxPage))
                        break
                # save the rest of illnessLinks
                restIllnessLinks = illnessLink
                restIllnessLinksJson = json.dumps(restIllnessLinks)
                with open('./linksTempV2/{}_restIllnessLinks.json'.format(department), 'w', encoding='utf-8') as fp:
                    fp.write(restIllnessLinksJson)
    def run(self):
        global illnessLink
        while True:
            if illnessLink:
                self.curTable = self.DB[self.departmentDict[self.department]]

                if self.type == 'all':
                    self.crawQALinks()
                elif self.type == 'rest':
                    # load rest illness links
                    pass
            else:
                break

class myProxy(threading.Thread):

    def __init__(self, proxyNum):
        super(myProxy, self).__init__()  # 调用父类的构造函数
        self.proxyNum = proxyNum
    def requestUrl(self, url):
        while True:
            try:
                response = requests.get(url)
            except:
                time.sleep(2)
            else:
                if response.status_code == 200:
                    time.sleep(2)
                    return response
                else:
                    time.sleep(2)
    def getProxies(self):
        global getProxyFlag
        global proxyPool
        while True:
            if finishScrapyFlag:
                break
            print('*' * 60 + '\n\tthe rest of proxies in the proxyPool:{}\n'.format(len(proxyPool)) + '*' * 60)
            url = 'http://api3.xiguadaili.com/ip/?tid=556466288353255&num={}&category=2'.format(self.proxyNum)
            while True:
                response = self.requestUrl(url)
                if response:
                    getProxyFlag = True
                    break
            proxies = response.text.split('\r\n')
            proxyPooltemp=[]
            for curIp in proxies:
                proxyPooltemp.append({'http': 'http://{}'.format(curIp), 'https': 'http://{}'.format(curIp)})
            proxyPool = proxyPooltemp.copy()
            print('*'*60+'\n\tupdate proxyPool sucessfully\n'+'*'*60)
            time.sleep(30)
    def run(self):
        self.getProxies()

def getQAUrl(DBname, proxyNum, wormNum, maxPage=20, useProxy=True, headers=None, departmentDict=None):
    global finishScrapyFlag
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017')
    except:
        print('can not connet Mongodb')
        quit()
    print('conneted Mongodb successfully')
    global illnessLinks
    with open('./illnessLinks.json', 'r', encoding='utf-8') as fp:
        illnessLinks = fp.read()
    illnessLinks = json.loads(illnessLinks)
    DB = client[DBname] #Link DB
    startTime = time.time()
    proxies = myProxy(proxyNum=proxyNum)
    proxies.start()     # start  get proxies
    while not getProxyFlag:
        #  waiting build proxyPool
        pass
    for item in illnessLinks[2:]:
        wormPool = []
        for i in range(wormNum):
            # create worm , add it in wormPool
            curWorm = myWorm(
                id=i+1,
                maxPage=maxPage,
                type=type,
                useProxy=useProxy,
                department=item[0],
                DB=DB,
                headers=headers,
                departmentDict=departmentDict,
            )
            wormPool.append(curWorm)
        for curWorm in wormPool:
            # active worms
            curWorm.start()
            time.sleep(2)
        for curWorm in wormPool:
            # waiting worms to finish crawling
            curWorm.join()
    finishScrapyFlag = True
    proxies.join()
    endTime = time.time()
    print('total Time:{}s'.format(endTime - startTime))