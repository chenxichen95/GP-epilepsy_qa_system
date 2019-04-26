import jieba
import numpy as np
import re
import copy
from tqdm import tqdm
import pandas as pd
from bson import ObjectId
import json
import os


def getData(illnessName, collectionNames, db):
    '''
        extract data from db based on collectionNames
    '''
    illnessSet = []
    for curCollection in collectionNames:
        print(f'searching collection: {curCollection}...')
        for item in db[curCollection].find():
            if item['illnessType'] == illnessName:
                illnessSet.append(item)
    print('done.')
    return illnessSet

def delRepetition(dataset):
    '''
        delete repetitions in dataset based on 'Q'(key in dictionary)
    '''
    dataset0 = copy.deepcopy(dataset)
    questionUnique = {} # 使用 python 的 字典结构 加速检索过程
    illnessSet_unique = []
    repetitionNum = 0
    repetitionList = []
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            if questionUnique.get(item['Q'], 0) == 0:
                questionUnique.setdefault(item['Q'], 1)
                illnessSet_unique.append(item)
            else:
                repetitionList.append(item)
                repetitionNum += 1
            pbar.update(1)

    print(f'有 {repetitionNum} 个样本重复。')
    print(f'剩下 {len(illnessSet_unique)} 个“癫痫”疾病的样本')
    return illnessSet_unique, repetitionList

def keepChinaEngPunc(dataset):
    '''
        保留中文、英文、标点符号
        keep chinese, english, punctuations in dataset.
    '''
    dataset0 = copy.deepcopy(dataset)
    keepChar = u'[^a-zA-Z\u4e00-\u9fa5,.:;''""?|!\^%()-\[\]{}/\`~，。：；‘’”“？|！……%（）【】{}·~～、]'
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            dataset0[index]['Q'] = re.sub(f'{keepChar}', '', item['Q'])
            dataset0[index]['Q_detailed'] = re.sub(f'{keepChar}', '', item['Q_detailed'])
            dataset0[index]['A1'] = re.sub(f'{keepChar}', '', item['A1'])
            pbar.update(1)
    # 检查多少样本被处理
    subNum = 0
    subSample = []
    for index, item in enumerate(dataset):
        if item['Q'] != dataset0[index]['Q']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        elif item['Q_detailed'] != dataset0[index]['Q_detailed']:
            subNum +=1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        elif item['A1'] != dataset0[index]['A1']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        else:
            continue
    print(f'在“保留中文、英文和标点符号”的过程中，样本被处理的数量： {subNum}({subNum/len(dataset0):.2%})')
    return dataset0, subSample

def chinaPuncSubEngPunc(dataset):
    '''
        english punctuations convert to chinese punctuations
    '''
    usualChinaPunc = ['，', '。', '：', '；', '？', '！', '（', '）', '【', '】']
    usualEngPunc = [',', '.', ':', ';', '?', '!', '(', ')', '[', ']']
    dataset0 = copy.deepcopy(dataset)
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            for curEngIndex, curEngPunc in enumerate(usualEngPunc):
                item['Q'] = re.sub(f'[{curEngPunc}]', f'{usualChinaPunc[curEngIndex]}', item['Q'])
                item['Q_detailed'] = re.sub(f'[{curEngPunc}]', f'{usualChinaPunc[curEngIndex]}', item['Q_detailed'])
                item['A1'] = re.sub(f'[{curEngPunc}]', f'{usualChinaPunc[curEngIndex]}', item['A1'])
            dataset0[index]['Q'] = item['Q']
            dataset0[index]['Q_detailed'] = item['Q_detailed']
            dataset0[index]['A1'] = item['A1']
            pbar.update(1)
    subNum = 0
    subSample = []
    for index, item in enumerate(dataset):
        if item['Q'] != dataset0[index]['Q']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        elif item['Q_detailed'] != dataset0[index]['Q_detailed']:
            subNum +=1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        elif item['A1'] != dataset0[index]['A1']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        else:
            continue
    print(f'在“英文标点符号转中文标点符号”的过程中，样本被处理的数量： {subNum}({subNum/len(dataset0):.2%})')
    return dataset0, subSample

def upper2Lower(dataset):
    '''
        upper english char convert to lower english char
    '''
    dataset0 = copy.deepcopy(dataset)
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            dataset0[index]['Q'] = item['Q'].lower()
            dataset0[index]['Q_detailed'] = item['Q_detailed'].lower()
            dataset0[index]['A1'] = item['A1'].lower()
            pbar.update(1)
    # 检查多少样本被处理
    subNum = 0
    subSample = []
    for index, item in enumerate(dataset):
        if item['Q'] != dataset0[index]['Q']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        elif item['Q_detailed'] != dataset0[index]['Q_detailed']:
            subNum +=1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        elif item['A1'] != dataset0[index]['A1']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
            continue
        else:
            continue
    print(f'在“大写英文字母转小写英文字母”的过程中，样本被处理的数量： {subNum}({subNum/len(dataset0):.2%})')
    return dataset0, subSample

def detectQuestionMark(dataset):
    '''
        check if the sentence end with '？'
    '''
    dataset0 = copy.deepcopy(dataset)
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            lastChar = item['Q'][-1]
            if lastChar != '？':
                dataset0[index]['Q'] = item['Q'] + '？'
            pbar.update(1)
    # 检查多少样本被处理
    subNum = 0
    subSample = []
    for index, item in enumerate(dataset):
        if item['Q'] != dataset0[index]['Q']:
            subNum += 1
            subSample.append({'old': item, 'new': dataset0[index]})
    print(f'在“问句检查”的过程中，样本被处理的数量： {subNum}({subNum/len(dataset0):.2%})')
    return dataset0, subSample

def filterQ(dataset):
    '''
        filter dataset based on the keywords
    '''
    filterKeyWords = ['吗', '什么', '怎么', '哪些', '呢', '怎么办', '如何', '是不是', '为什么',
                      '怎样', '请问', '怎么样', '多少', '怎么回事', '哪里', '好不好', '有没有',
                      '可不可以', '几年', '几天', '哪个', '多久', '是否', '有用吗']
    dataset_filter = []
    for index, item in enumerate(dataset):
        curQ = item['Q']
        curQ_cut = list(jieba.cut(curQ))
        QSentenceNum = re.split('[，。？！……]',curQ)[:-1]
        if len(QSentenceNum) < 3:
            for curFilterWord in filterKeyWords:
                if curFilterWord in curQ_cut:
                    dataset_filter.append(item)
                    break
    subNum = len(dataset_filter)
    print(f'在“问句筛选”的过程中，剩余样本个数： {subNum}({subNum/len(dataset):.2%})')
    return dataset_filter

def delAText(dataset):
    '''
        process answer text in dataset
    '''
    dataset0 = copy.deepcopy(dataset)
    dataNew = []
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            curA = item['A1']
            curA_sub = re.sub('病情分析：|指导意见：|医生建议：', '', curA)
            dataNew.append({
                '_id': item['_id'],
                'illnessType': item['illnessType'],
                'Q': item['Q'],
                'Q_detailed': item['Q_detailed'],
                'A': curA_sub,
            }
            )
            pbar.update(1)
    # 检查多少样本被处理
    subNum = 0
    subSample = []
    for index, item in enumerate(dataset):
        if item['A1'] != dataNew[index]['A']:
            subNum += 1
            subSample.append({'old': item, 'new': dataNew[index]})
            continue
    print(f'在“答案文本处理”的过程中，样本被处理的数量： {subNum}({subNum/len(dataset):.2%})')
    return dataNew, subSample


def calcMaxLen(dataset):
    '''
        statistic the len of text in dataset
    '''
    maxLen_Q = 0
    maxLen_Q_cut = 0
    maxLen_Q_detailed = 0
    maxLen_Q_detailed_cut = 0
    maxLen_A = 0
    maxLen_A_cut = 0
    with tqdm(total=len(dataset)) as pbar:
        for index, item in enumerate(dataset):
            curQ = item['Q']
            curQ_cut = list(jieba.cut(curQ))
            curQ_detailed = item['Q_detailed']
            curQ_detailed_cut = list(jieba.cut(curQ_detailed))
            curA = item['A']
            curA_cut = list(jieba.cut(curA))

            if maxLen_Q < len(curQ):
                maxLen_Q = len(curQ)
            if maxLen_Q_cut < len(curQ_cut):
                maxLen_Q_cut = len(curQ_cut)
            if maxLen_Q_detailed < len(curQ_detailed):
                maxLen_Q_detailed = len(curQ_detailed)
            if maxLen_Q_detailed_cut < len(curQ_detailed_cut):
                maxLen_Q_detailed_cut = len(curQ_detailed_cut)
            if maxLen_A < len(curA):
                maxLen_A = len(curA)
            if maxLen_A_cut < len(curA_cut):
                maxLen_A_cut = len(curA_cut)
            pbar.update(1)
    print(f'“Q”的最大字数是：{maxLen_Q},最大词数是：{maxLen_Q_cut}')
    print(f'“Q_detailed”的最大字数是：{maxLen_Q_detailed},最大词数是：{maxLen_Q_detailed_cut}')
    print(f'“A”的最大字数是：{maxLen_A},最大词数是：{maxLen_A_cut}')

def statisticWord(dataset):
    '''
        statistic the distribution of word count in dataset
    '''
    len_Q = []
    len_Q_detailed = []
    len_A = []
    with tqdm(total=len(dataset)) as pbar:
        for index, item in enumerate(dataset):
            curQ = item['Q']
            len_Q.append(len(list(jieba.cut(curQ))))
            curQ_detailed = item['Q_detailed']
            len_Q_detailed.append(len(list(jieba.cut(curQ_detailed))))
            curA = item['A']
            len_A.append(len(list(jieba.cut(curA))))
            pbar.update(1)
    len_Q = pd.DataFrame(len_Q)
    len_Q_detailed = pd.DataFrame(len_Q_detailed)
    len_A = pd.DataFrame(len_A)
    print('='*20)
    print('“Q_”的词量统计情况如下：')
    print(len_Q.describe(percentiles=[0.5,0.8,0.9,0.99,0.995]))
    print('='*20)
    print('='*20)
    print('“Q_detailed”的词量统计情况如下：')
    print(len_Q_detailed.describe(percentiles=[0.5,0.8,0.9,0.99,0.999]))
    print('='*20)
    print('='*20)
    print('“A”的词量统计情况如下：')
    print(len_A.describe(percentiles=[0.5,0.8,0.9,0.99,0.991]))
    print('='*20)
    return len_Q, len_Q_detailed, len_A

def delToLongSample(dataset, maxLen=300):
    '''
        delete the data which word count over 300 in 'Q_detailed' or 'A'
    '''
    dataset0 = copy.deepcopy(dataset)
    dataNew = []
    with tqdm(total=len(dataset0)) as pbar:
        for index, item in enumerate(dataset0):
            curQ_detailed_Len = len(list(jieba.cut(item['Q_detailed'])))
            curA_Len = len(list(jieba.cut(item['A'])))

            if curQ_detailed_Len <= maxLen:
                if curA_Len <= maxLen:
                    item['label'] = 1
                    dataNew.append(item)
            pbar.update(1)
    print(f'经过“文本长度处理”后，剩余样本个数：{len(dataNew)}({len(dataNew)/len(dataset):.2%})')
    return dataNew

def checkNone(dataset):
    '''
        check if none in dataset, if exist, delete the bad data.
    '''
    goodDataset = []
    with tqdm(total=len(dataset)) as pbar:
        for index, item in enumerate(dataset):
            if (item['Q'] == '') or (item['Q'] == []):
                pbar.update(1)
                continue
            elif (item['Q_detailed'] == '') or (item['Q_detailed'] == []):
                pbar.update(1)
                continue
            elif (item['A'] == '') or (item['A'] == []):
                pbar.update(1)
                continue
            else:
                goodDataset.append(item)
                pbar.update(1)

    print(f'删除含有空值的样本个数：{len(dataset)-len(goodDataset)}，剩余样本个数：{len(goodDataset)}({len(goodDataset)/len(dataset):.2%})')
    return goodDataset

def getNoEpilepsyData(collectionNames, db):
    '''
        get no-epilepsy data
    '''
    illnessSet = []
    for curCollection in collectionNames:
        print(f'searching collection: {curCollection}...')
        for item in db[curCollection].find():
            if (item['illnessType'] != '癫痫') and (item['illnessType'] != ''):
                illnessSet.append(item)
    print('done.')
    print(f'共获取样本个数：{len(illnessSet)}')
    return illnessSet


def processingData(isEpilepsy, collectionNames, db):
    illnessSet = getData('癫痫', collectionNames, db)
    illnessSet_unique, _ = delRepetition(illnessSet)
    illnessSet_unique_keep, _ = keepChinaEngPunc(illnessSet_unique)
    illnessSet_unique_keep_subPunc, _ = chinaPuncSubEngPunc(illnessSet_unique_keep)
    illnessSet_unique_keep_subPunc_lower, _ = upper2Lower(illnessSet_unique_keep_subPunc)
    illnessSet_unique_keep_subPunc_lower_qCheck, _ = detectQuestionMark(
        illnessSet_unique_keep_subPunc_lower)
    illnessSet_unique_keep_subPunc_lower_qCheck_filter = filterQ(
        illnessSet_unique_keep_subPunc_lower_qCheck, )
    illnessSet_unique_keep_subPunc_lower_qCheck_filter_dealA, _ = delAText(
        illnessSet_unique_keep_subPunc_lower_qCheck_filter)
    if isEpilepsy:
        calcMaxLen(illnessSet_unique_keep_subPunc_lower_qCheck_filter_dealA)
        statisticWord(illnessSet_unique_keep_subPunc_lower_qCheck_filter_dealA)
    illnessSet_unique_keep_subPunc_lower_qCheck_filter_dealA_dealT = delToLongSample(
        illnessSet_unique_keep_subPunc_lower_qCheck_filter_dealA, maxLen=300)
    illnessSet_after_process = checkNone(illnessSet_unique_keep_subPunc_lower_qCheck_filter_dealA_dealT)
    return illnessSet_after_process

def buildNegative(dataset_epilepsy, dataset_noEpilepsy, rate=1, seed=1000):
    '''
        build negative data
    '''
    dataset_epilepsy0 = copy.deepcopy(dataset_epilepsy)
    dataset_noEpilepsy0 = copy.deepcopy(dataset_noEpilepsy)
    epilepsyNum = len(dataset_epilepsy)
    noEpilepsyNum = int(epilepsyNum * rate)
    index_epilepsy = 0
    index_noEpilepsy = 0
    seedCount = 0
    np.random.seed(seed)
    randIndex_noEpilepsy = np.random.permutation(noEpilepsyNum)
    np.random.seed(seed)
    randID =np.random.permutation(10**5)
    np.random.seed(seed)
    randIndex = np.random.permutation(epilepsyNum)
    dataset_epilepsy_negative = []
    while True:
        if index_epilepsy % epilepsyNum == 0:
            index_epilepsy = 0
            np.random.seed(seed + seedCount)
            seedCount += 1
            randIndex = np.random.permutation(epilepsyNum)

        if index_noEpilepsy == noEpilepsyNum:
            break
        curItem = dataset_epilepsy0[randIndex[index_epilepsy]]
        curFalseItem = dataset_noEpilepsy0[randIndex_noEpilepsy[index_noEpilepsy]]
        index_epilepsy += 1
        index_noEpilepsy += 1
        curItem['A'] = curFalseItem['A']
        curItem['_id'] = ObjectId(f'{randID[index_noEpilepsy]:024d}')
        curItem['label'] = 0
        dataset_epilepsy_negative.append(curItem)
    print(f'总共构建了 {len(dataset_epilepsy_negative)} 反例样本')
    return dataset_epilepsy_negative


def deal_concat_random_cut(dataset_positive0, dataset_negative0, seed=1000, testRate=0.3, splitWord=True):
    '''
        generate train set and test set from dataset_positive and dataset_negative
    '''
    dataset_positive = copy.deepcopy(dataset_positive0)
    dataset_negative = copy.deepcopy(dataset_negative0)
    trainNum_positive = int(len(dataset_positive) * (1 - testRate))
    trainNum_negative = int(len(dataset_negative) * (1 - testRate))

    np.random.seed(seed)
    randIndex_positive = np.random.permutation(len(dataset_positive))
    np.random.seed(seed + 1)
    randIndex_negative = np.random.permutation(len(dataset_negative))

    dataset_trainAndDev = []
    dataset_test = []

    dataset_trainAndDev.extend([dataset_positive[i] for i in randIndex_positive[:trainNum_positive]])
    dataset_trainAndDev.extend([dataset_negative[i] for i in randIndex_negative[:trainNum_negative]])

    dataset_test.extend([dataset_positive[i] for i in randIndex_positive[trainNum_positive:]])
    dataset_test.extend([dataset_negative[i] for i in randIndex_negative[trainNum_negative:]])

    np.random.seed(seed)
    randIndex_trainAndDev = np.random.permutation(len(dataset_trainAndDev))
    np.random.seed(seed + 1)
    randIndex_test = np.random.permutation(len(dataset_test))

    dataset_trainAndDev2 = []
    with tqdm(total=len(dataset_trainAndDev)) as pbar:
        pbar.set_description('build train and dev set')
        for index in randIndex_trainAndDev:
            item = dataset_trainAndDev[index]
            if splitWord:
                item['Q'] = jieba.lcut(item['Q'])
                item['Q_detailed'] = jieba.lcut(item['Q_detailed'])
                item['A'] = jieba.lcut(item['A'])
            dataset_trainAndDev2.append(item)
            pbar.update(1)

    dataset_test2 = []
    with tqdm(total=len(dataset_test)) as pbar:
        pbar.set_description('build test set')
        for index in randIndex_test:
            item = dataset_test[index]
            if splitWord:
                item['Q'] = jieba.lcut(item['Q'])
                item['Q_detailed'] = jieba.lcut(item['Q_detailed'])
                item['A'] = jieba.lcut(item['A'])
            dataset_test2.append(item)
            pbar.update(1)

    return dataset_trainAndDev2, dataset_test2

def jsonSave(data, path):
    '''
        将 data 中 “_id” 的 ObjectId 对象 转换为 str。
        使用 json 保存
    '''
    data2 = copy.deepcopy(data)
    for index, item in enumerate(data2):
        data2[index]['_id'] = str(item['_id'].binary)
    dataJson = json.dumps(data2)
    with open(f'{path}', 'w', encoding='utf-8') as fp:
        fp.write(dataJson)
    print(f'{os.path.abspath(path)} save done')
