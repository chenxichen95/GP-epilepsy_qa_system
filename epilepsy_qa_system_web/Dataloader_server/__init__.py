import numpy as np
import json
import os
from tqdm import tqdm
import time
import tensorflow as tf
import gc
import re
import jieba
class DataLoader():
    def __init__(
            self,
            dataPath = './data',
            ansPoolPath='',
            w2vFilePath='',
            maxA_Len=300, maxQ_Len=35, maxQ_detailed_Len=300, embeddingSize=150, silence=False,
    ):

        self.ansNum = 0
        self.ansPool_id = ''
        self.ansPool_A_vector = ''
        self.ansPool_sentence_EndPos_A = ''

        self.dataPath = dataPath
        if ansPoolPath:
            self.ansPoolPath = ansPoolPath
            self.ansPool = self.buildAnsPool(ansPoolPath, fenci=False)
        else:
            raise Exception('can not find the ans pool')

        if w2vFilePath:
            self.w2vFilePath = w2vFilePath
            self.w2v = self.loadJson(w2vFilePath)
            # build OOV word vector
            w2v_allVector = np.stack([value for value in self.w2v.values()])
            w2v_mean, w2v_std = np.mean(w2v_allVector, axis=0), np.std(w2v_allVector, axis=0)
            self.OOV_vector = np.random.normal(w2v_mean, w2v_std, (embeddingSize,)).tolist()
            if not silence: print(f'pretrain word2vec: "{w2vFilePath}" loaded')
        else:
            self.w2v = []

        self.maxA_Len = maxA_Len
        self.maxQ_Len = maxQ_Len
        self.maxQ_detailed_Len = maxQ_detailed_Len
        self.embeddingSize = embeddingSize
        self.silence = silence

        self.padding_vector = np.zeros((self.embeddingSize,))
        self.sentenceEndPunct = '，。？！'
        self.sentencePad = 99999

        self.max_sentenceEndPosNum_A = 52  # first 0
        self.max_sentenceEndPosNum_Q_detailed = 41  # first 0

        self.usualChinaPunc = ['，', '。', '：', '；', '？', '！', '（', '）', '【', '】']
        self.usualEngPunc = [',', '.', ':', ';', '?', '!', '(', ')', '[', ']']



        if os.path.exists(f'{dataPath}/ansPool_id.npy') and os.path.exists(
                f'{dataPath}/ansPool_A_vector.npy') and os.path.exists(f'{dataPath}/ansPool_sentence_EndPos_A.npy'):
            print('ansPool_id.npy ansPool_A_vector.npy ansPool_sentence_EndPos_A.npy exist')
            self.ansPool_id = np.load(f'{dataPath}/ansPool_id.npy')
            self.ansPool_A_vector = np.load(f'{dataPath}/ansPool_A_vector.npy')
            self.ansPool_sentence_EndPos_A = np.load(f'{dataPath}/ansPool_sentence_EndPos_A.npy')
            pass
        else:
            print('ansPool_id.npy ansPool_A_vector.npy ansPool_sentence_EndPos_A.npy dont not exist')
            print('building... ')
            self.ansPool_id, self.ansPool_A_vector, self.ansPool_sentence_EndPos_A = self.dealAnsPool()

    def loadJson(self, file):
        '''
            加载 json 文件
        '''
        with open(file, 'r', encoding='utf-8') as fp:
            dataJson = fp.read()
        return json.loads(dataJson)

    def buildAnsPool(self, file, fenci=True):
        '''
            fenci
        '''
        ansPool = []
        data = self.loadJson(file)
        self.ansNum = len(data)
        with tqdm(total=len(data)) as pbar:
            pbar.set_description_str('buildAnsPool')
            for index, item in enumerate(data):
                if fenci:
                    ansPool.append({'A': jieba.lcut(item['A']), 'id': item['_id']})
                else:
                    ansPool.append({'A': item['A'], 'id': item['_id']})
                pbar.update(1)

        return ansPool

    def wordProcess(self, word):
        '''
            作用：
                将 word 转换为词向量，如果词向量不存在，则使用某种分布填充。
                将 word 添加 字典 vocab 中
            return：
                word_vector
        '''
        word_vector = self.w2v.get(word)
        if not word_vector:
            if not self.silence: print(f'have OOV: {word}')
            word_vector = self.OOV_vector


        return word_vector

    def word2vec(self, sentence):
        return [self.wordProcess(word) for word in sentence]

    def padding(self, mode, vector):
        if mode == 'A':
            return vector + [self.padding_vector for i in range(self.maxA_Len - len(vector))]
        elif mode == 'Q':
            return vector + [self.padding_vector for i in range(self.maxQ_Len - len(vector))]
        elif mode == 'Q_detailed':
            return vector + [self.padding_vector for i in range(self.maxQ_detailed_Len - len(vector))]
        else:
            raise Exception('mode: {mode} don not exist')

    def dealAnsPool(self):
        '''
            word => word2vec ,padding, cal sentencenEndPos
        '''
        ID = []
        A_vector = []
        sentence_EndPos_A = []
        ansPool_fenci = self.buildAnsPool(self.ansPoolPath, fenci=True)
        for index, item in enumerate(ansPool_fenci):
            ID.append(item['id'])
            curA = item['A']

            # word2vec
            curA_vector = self.word2vec(curA)

            #padding
            curA_vector = self.padding(mode='A', vector=curA_vector)

            A_vector.append(curA_vector)

            # calc sentence position in A
            curSentence_EndPos_A = [index for index, word in enumerate(curA) if word in self.sentenceEndPunct]
            if curSentence_EndPos_A == []:
                '''
                    a litle of A don't have any punct.
                '''
                curSentence_EndPos_A = [len(item['A']) - 1]
            sentence_EndPos_A.append(curSentence_EndPos_A)
            if len(curSentence_EndPos_A) > self.max_sentenceEndPosNum_A:
                self.max_sentenceEndPosNum_A = len(curSentence_EndPos_A)

        # sentence_pos padding
        for index, item in enumerate(sentence_EndPos_A):
            sentence_EndPos_A[index] = item + [self.sentencePad for i in range(self.max_sentenceEndPosNum_A - len(item))]

        id = np.array(ID)
        A_vector = np.array(A_vector, dtype=np.float32)
        sentence_EndPos_A = np.array(sentence_EndPos_A, dtype=np.int32)

        np.save(f'{self.dataPath}/ansPool_id.npy', id)
        np.save(f'{self.dataPath}/ansPool_A_vector.npy', A_vector)
        np.save(f'{self.dataPath}/ansPool_sentence_EndPos_A.npy', sentence_EndPos_A)

        return id, A_vector, sentence_EndPos_A

    def keepChinaEngPunc(self, text):
        '''
            作用：
                删除文本中的乱码，无用的标点符号，多余的空格
                只保留规定的字符

            return dataset, subSample
        '''
        keepChar = u'[^a-zA-Z\u4e00-\u9fa5,.:;''""?|!\^%()-\[\]{}/\`~，。：；‘’”“？|！……%（）【】{}·~～、]'
        text = re.sub(f'{keepChar}', '', text)

        return text

    def chinaPuncSubEngPunc(self, text):
        '''
            常见的英文标点符号 替换为 对应的中文标点符号

            return text

            Bug:
                小数点的 '.' 应该不可以被替换为 '。'

        '''
        for curEngIndex, curEngPunc in enumerate(self.usualEngPunc):
            text = re.sub(f'[{curEngPunc}]', f'{self.usualChinaPunc[curEngIndex]}', text)

        return text

    def upper2Lower(self, text):
        '''
            将文本中的 大写英文字母 转 小写英文字母

            return dataset, subSample
        '''

        text = text.lower()

        return text

    def textPreProcessing(self, text):
        text = self.keepChinaEngPunc(text)
        text = self.chinaPuncSubEngPunc(text)
        text = self.upper2Lower(text)

        return text


    def getModelServerInputData(self, q, q_detailed='', testSpeed=False):
        '''

        :param Q:
            a raw question text
        :param Q_detailed:
            a raw q_detailed text
        :return:
            inputDataSet, include ans, id sentencePos ..
        '''
        startTime = time.time()
        q = self.textPreProcessing(q)
        if q_detailed == '':
            q_detailed = q
        else:
            q_detailed = self.textPreProcessing(q_detailed)

        # split word
        q = jieba.lcut(q)
        q_detailed = jieba.lcut(q_detailed)

        # word2vec
        q_vector = self.word2vec(q)
        q_detailed_vector = self.word2vec(q_detailed)

        # padding
        q_vector = self.padding(mode='Q', vector=q_vector)
        q_detailed_vector = self.padding(mode='Q_detailed', vector=q_detailed_vector)

        # calc sentence position in Q_detailed
        curSentence_EndPos_Q_detailed = [index for index, word in enumerate(q_detailed) if
                                         word in self.sentenceEndPunct]
        if curSentence_EndPos_Q_detailed == []:
            '''
                a litle of Q_detailed don't have any punct.
            '''
            curSentence_EndPos_Q_detailed = [len(q_detailed) - 1]

        # padding
        sentence_EndPos_Q_detailed = curSentence_EndPos_Q_detailed + [self.sentencePad for i in range(self.max_sentenceEndPosNum_Q_detailed - len(curSentence_EndPos_Q_detailed))]

        '''
        q_vector = [q_vector for i in range(self.ansNum)]
        q_vector = np.array(q_vector, dtype=np.float32)
        q_detailed_vector = [q_detailed_vector for i in range(self.ansNum)]
        q_detailed_vector = np.array(q_detailed_vector, dtype=np.float32)
        sentence_EndPos_Q_detailed = [sentence_EndPos_Q_detailed for i in range(self.ansNum)]
        sentence_EndPos_Q_detailed = np.array(sentence_EndPos_Q_detailed, dtype=np.int32)
        '''
        q_vector = np.array(q_vector, dtype=np.float32)[np.newaxis, :]
        q_vector = np.repeat(q_vector, self.ansNum, axis=0)

        q_detailed_vector = np.array(q_detailed_vector, dtype=np.float32)[np.newaxis, :]
        q_detailed_vector = np.repeat(q_detailed_vector, self.ansNum, axis=0)

        sentence_EndPos_Q_detailed = np.array(sentence_EndPos_Q_detailed, dtype=np.int32)[np.newaxis, :]
        sentence_EndPos_Q_detailed = np.repeat(sentence_EndPos_Q_detailed, self.ansNum, axis=0)



        endTime = time.time()
        if testSpeed:
            print(f'getModelServerInputData() total time:{endTime- startTime}')

        return {
            'id': self.ansPool_id,
            'Q': q_vector,
            'Q_detailed': q_detailed_vector,
            'A': self.ansPool_A_vector,
            'sentencePos_A': self.ansPool_sentence_EndPos_A,
            'sentencePos_Q_detailed': sentence_EndPos_Q_detailed,
        }



if __name__ == '__main__':
    dataloader = dataLoader(dataPath='../data')
    q = '癫痫平时饮食要注意什么？'
    q_detailed = '我有个亲戚因为出了交通事故导致了癫痫病，这个病每次发作起来总是非常的突然，他总是好好的就会摔倒手脚抽搐，现在每天按照医生要求服药治疗，想知道癫痫平时饮食要注意什么呢？'
    inputData = dataloader.getModelServerInputData(q, q_detailed, testSpeed=True)
    print('end')