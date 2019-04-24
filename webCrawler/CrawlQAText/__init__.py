import time
import requests
from bs4 import BeautifulSoup
import threading
import pymongo

proxyPool = []
getProxyFlag = False
finishScrapyFlag = False
class myWorm(threading.Thread):

    def __init__(self, id=0, type='all', department=None, curTable=None, headers=None):
        super(myWorm, self).__init__()  # 调用父类的构造函数
        self.id = id
        self.department = department
        self.curTable = curTable
        self.type = type
        self.curIllness = None
        self.headers = headers
    def clearErrorLogOl(self):
        with open('./QA_log_V2/{}_errorLog'.format(self.id), 'w', encoding='utf-8') as fp:
            fp.write('\n')
    def writeLog(self, str):
        with open('./QA_log_V2/myLog_{}_V2.log'.format(self.department), 'a', encoding='utf-8') as fp:
            fp.write(str)
            fp.write('\n')
    def writeErrorLogOl(self, str):
        with open('./QA_log_V2/{}_errorLog'.format(self.id), 'a', encoding='utf-8') as fp:
            fp.write(str)
            fp.write('\n')
    def requestUrlWithProxy(self, url, proxy):
        global proxyPool
        FailCount = 0
        self.clearErrorLogOl()
        while True:
            try:
                response = requests.get(url, headers=self.headers, proxies=proxy, timeout=10)
            except:
                proxy = proxyPool.pop(0)  # change proxy
                self.writeErrorLogOl('wormID:{} crawling {} current url:{}:'
                                     'can not request'.format(self.id, self.department,  url))
                FailCount += 1
            else:
                if response.status_code == 200:
                    return response, proxy
                else:
                    proxy = proxyPool.pop(0)  # change proxy
                    FailCount += 1
                    self.writeErrorLogOl('wormID:{} crawling {} current url:{}:'
                                         'request fail, status_code:{}'.format(self.id, self.department,url, response.status_code))
            if FailCount == 200:
                return []
    def getQAandWriteDB(self,soup):
        try:
            Q = soup.find('h3', class_='quest-title').text.strip()
            Q_detailed = soup.find('div', class_='illness-pics').find('p').text.strip()
            illness = soup.find('p', class_='illness-type').find('a').text.strip()
            A = []
            for item in soup.find_all('dl', class_='answer-info-cont'):
                A.append(item.find('p', class_='answer-words').text.strip())
        except:
            return False
        else:
            if (not Q) or (not A):
                # don't exist Q or A , bad data
                return False
            else:
                try:
                    curData = {}
                    curData['illnessType'] = illness
                    curData['Q'] = Q
                    curData['Q_detailed'] = Q_detailed
                    for i in range(len(A)):
                        curData['A{}'.format(i+1)] = A[i]
                    self.curTable.insert(curData)
                except:
                    return False
                else:
                    return True
    def crawQALinks(self):
            # in department
            global curQALinks
            global proxyPool
            global curQALinksNum0
            department = self.department
            proxy = proxyPool.pop(0)
            while curQALinks:
                link = curQALinks.pop(0)
                response, proxy = self.requestUrlWithProxy(link, proxy)
                if response:
                    soup = BeautifulSoup(response.text, 'html5lib')
                else:
                    # can't get current link response, go to next page
                    print('wormID:{} crawling department:{} current url:{}'
                          '\n\tcan not get current link response\tthe rest number of cuQALinks:{:.2%}/{}'
                          .format(self.id, department, link, len(curQALinks)/curQALinksNum0, len(curQALinks)))
                    self.writeLog('wormID:{} crawling department:{} current url:{}:'
                             'can not get current link response'.format(self.id, department, link))
                    continue
                if self.getQAandWriteDB(soup):
                    print('wormID:{} crawling department:{} current url:{}'
                          '\n\tsuccessfully\tthe rest number of cuQALinks:{:.2%}/{}'
                          .format(self.id, self.department, link, len(curQALinks)/curQALinksNum0, len(curQALinks)))
                else:
                    print('wormID:{} crawling department:{} current url:{}:'
                          '\n\tfail\tthe rest number of cuQALinks:{:.2%}/{}'
                          .format(self.id, self.department, link, len(curQALinks)/curQALinksNum0, len(curQALinks)))
    def run(self):
        global curQALinks
        while True:
            if curQALinks:
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

def getQAText(DBname, DBname2, proxyNum, departments, departmentDict, wormNum):
    global finishScrapyFlag
    global curQALinks
    global curQALinksNum0
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017')
    except:
        print('can not connet Mongodb')
        quit()
    print('conneted Mongodb successfully')

    DB = client[DBname] #Link DB which saved QALinks
    DB2 = client[DBname2] # save QA


    startTime = time.time()

    proxies = myProxy(proxyNum=proxyNum)
    proxies.start()     # start  get proxies

    while not getProxyFlag:
        #  waiting build proxyPool
        pass
    for item in departments:  # zyk error
        department = departmentDict[item]
        curTable = DB[department]
        curQALinks = []
        for item2 in curTable.find({}):
            curQALinks.append(item2['link'])
        curQALinks = list(set(curQALinks)) # delete repeat links
        curQALinksNum0 = len(curQALinks)
        wormPool = []
        for i in range(wormNum):
            # create worm , add it in wormPool
            curWorm = myWorm(
                id=i+1,
                type=type,
                department=department,
                curTable=DB2[department],
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


