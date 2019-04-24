from CrawlDepartmentUrl import getDepartmentUrl
from CrawlIllnessUrl import getIllnessUrl
from CrawlQAUrl import getQAUrl
from CrawlQAText import getQAText

if __name__ == "__main__":
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'max-age=0',
        #   'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    departmentDict = {
        '内科': 'neike',
        '外科': 'waike',
        '妇产科': 'fck',
        '男科': 'nanke',
        '生殖健康': 'szjk',
        '儿科': 'erke',
        '五官科': 'wgk',
        '肿瘤科': 'zlk',
        '皮肤性病科': 'pfxpk',
        '精神心理科': 'jsxlk',
        '感染科': 'grk',
        '老年病科': 'lnbk',
        '肝病科': 'gbk',
        '急诊科': 'jzk',
        '中医科': 'zyk',
        '体检保健科': 'tjbjk',
        '营养科': 'yyk',
        '成瘾医学科': 'cyyxk',
        '职业病科': 'zybk',
        '整形美容科': 'zxmrk',
    }
    departments = [
        '内科',
        '外科',
        '妇产科',
        '男科',
        '生殖健康',
        '儿科',
        '五官科',
        '肿瘤科',
        '皮肤性病科',
        '精神心理科',
        '感染科',
        '老年病科',
        '肝病科',
        '急诊科',
        '中医科',
        '体检保健科',
        '营养科',
        '成瘾医学科',
        '职业病科',
        '整形美容科',
    ]
    getDepartmentUrl(headers)
    getIllnessUrl(headers)
    getQAUrl(
        DBname='db_familyDoctor_QALink_V2',
        proxyNum=200,
        wormNum=30,
        maxPage=20,
        useProxy=True,
        headers=headers,
        departmentDict=departmentDict,
    )
    getQAText(
        DBname='db_familyDoctor_QALink_V2',
        DBname2='db_familyDoctor_QA_V2',
        proxyNum=500,
        departments=departments,
        departmentDict=departmentDict,
        wormNum=50
    )



