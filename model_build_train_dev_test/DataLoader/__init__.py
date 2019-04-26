import numpy as np
import json

class DataLoader():
    '''
        dataLoader
        作用：
            将预处理后的数据集，转化为可用于模型训练的格式。
            包括：
                词 转化 词向量
                句子结束符（，。？！）位置记录
                向量 padding
    '''

    def __init__(self, trainAndDev_dataPath=None, test_dataPath=None, w2vFilePath=None,
                 maxA_Len=300, maxQ_Len=35, maxQ_detailed_Len=300, embeddingSize=150,
                 silence=False
                 ):
        if trainAndDev_dataPath:
            self.trainAndDev_dataPath = trainAndDev_dataPath
            self.data_train_val = self.loadJson(trainAndDev_dataPath)
            if not silence: print(f'dataset: "{trainAndDev_dataPath}" loaded')
        else:
            self.data_train_val = []

        if test_dataPath:
            self.test_dataPath = test_dataPath
            self.data_test = self.loadJson(test_dataPath)
            if not silence: print(f'dataset: "{test_dataPath}" loaded')
        else:
            self.data_test = []

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

        self.vocab = {}
        self.allData = {}
        self.max_sentenceEndPosNum_A = 52  # first 0
        self.max_sentenceEndPosNum_Q_detailed = 41  # first 0
        self.train_val_Num = 0
        self.test_Num = 0

    def loadJson(self, file):
        '''
            加载 json 文件
        '''
        with open(file, 'r', encoding='utf-8') as fp:
            dataJson = fp.read()
        return json.loads(dataJson)

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

        if not self.vocab.get(word):
            self.vocab[word] = len(self.vocab)

        return word_vector

    def dataProcess(self, data):
        '''
            对 data 进行词向量转换，padding，句子结束位置标志
        '''
        ID = []
        Q_vector = []
        A_vector = []
        Q_detailed_vector = []
        sentence_EndPos_A = []
        sentence_EndPos_Q_detailed = []
        label = []

        # 词向量转换
        for index, item in enumerate(data):

            curID = item['_id']
            ID.append(curID)

            # word to vector
            curQ_vector = [self.wordProcess(word) for word in item['Q']]
            curQ_detailed_vector = [self.wordProcess(word) for word in item['Q_detailed']]
            curA_vector = [self.wordProcess(word) for word in item['A']]

            # padding
            curQ_vector += [self.padding_vector for i in range(self.maxQ_Len - len(curQ_vector))]
            curQ_detailed_vector += [self.padding_vector for i in
                                     range(self.maxQ_detailed_Len - len(curQ_detailed_vector))]
            curA_vector += [self.padding_vector for i in range(self.maxA_Len - len(curA_vector))]

            Q_vector.append(curQ_vector)
            Q_detailed_vector.append(curQ_detailed_vector)
            A_vector.append(curA_vector)
            label.append(item['label'])

            # calc sentence position in A
            curSentence_EndPos_A = [index for index, word in enumerate(item['A']) if word in self.sentenceEndPunct]
            if curSentence_EndPos_A == []:
                '''
                    a litle of A don't have any punct.
                '''
                curSentence_EndPos_A = [len(item['A']) - 1]
            sentence_EndPos_A.append(curSentence_EndPos_A)
            if len(curSentence_EndPos_A) > self.max_sentenceEndPosNum_A:
                self.max_sentenceEndPosNum_A = len(curSentence_EndPos_A)

            # calc sentence position in Q_detailed
            curSentence_EndPos_Q_detailed = [index for index, word in enumerate(item['Q_detailed']) if
                                             word in self.sentenceEndPunct]
            if curSentence_EndPos_Q_detailed == []:
                '''
                    a litle of Q_detailed don't have any punct.
                '''
                curSentence_EndPos_Q_detailed = [len(item['Q_detailed']) - 1]
            sentence_EndPos_Q_detailed.append(curSentence_EndPos_Q_detailed)
            if len(curSentence_EndPos_Q_detailed) > self.max_sentenceEndPosNum_Q_detailed:
                self.max_sentenceEndPosNum_Q_detailed = len(curSentence_EndPos_Q_detailed)

        # sentence_pos padding
        for index, item in enumerate(sentence_EndPos_A):
            sentence_EndPos_A[index] = item + [self.sentencePad for i in
                                               range(self.max_sentenceEndPosNum_A - len(item))]
        for index, item in enumerate(sentence_EndPos_Q_detailed):
            sentence_EndPos_Q_detailed[index] = item + [self.sentencePad for i in
                                                        range(self.max_sentenceEndPosNum_Q_detailed - len(item))]

        return (
            np.array(ID),
            np.array(Q_vector, dtype=np.float32),
            np.array(Q_detailed_vector, dtype=np.float32),
            np.array(A_vector, dtype=np.float32),
            np.array(sentence_EndPos_A, dtype=np.int32),
            np.array(sentence_EndPos_Q_detailed, dtype=np.int32),
            np.array(label, dtype=np.float32),
        )

    def getAllData(self):
        '''
            处理 data_train_val 数据集
            处理 data_test 数据集
        '''
        self.allData['train_dev'] = {}
        self.allData['train_dev']['id'], \
        self.allData['train_dev']['Q'], self.allData['train_dev']['Q_detailed'], \
        self.allData['train_dev']['A'], self.allData['train_dev']['sentencePos_A'], \
        self.allData['train_dev']['sentencePos_Q_detailed'], self.allData['train_dev']['label'] \
            = self.dataProcess(self.data_train_val)
        if not self.silence: print('data_train_dev process done')

        print(f'max_sentenceEndPosNum_A: {self.max_sentenceEndPosNum_A}')
        print(f'max_sentenceEndPosNum_Q_detailed: {self.max_sentenceEndPosNum_Q_detailed}')
        self.train_val_Num = self.allData['train_dev']['Q'].shape[0]

        self.allData['test'] = {}
        self.allData['test']['id'], \
        self.allData['test']['Q'], self.allData['test']['Q_detailed'], \
        self.allData['test']['A'], self.allData['test']['sentencePos_A'], \
        self.allData['test']['sentencePos_Q_detailed'], self.allData['test']['label'] \
            = self.dataProcess(self.data_test)
        if not self.silence: print('data_test process done')
        print(f'max_sentenceEndPosNum_A: {self.max_sentenceEndPosNum_A}')
        print(f'max_sentenceEndPosNum_Q_detailed: {self.max_sentenceEndPosNum_Q_detailed}')
        self.test_Num = self.allData['test']['Q'].shape[0]




