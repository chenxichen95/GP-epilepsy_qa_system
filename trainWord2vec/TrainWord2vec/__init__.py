import json
from gensim.models import word2vec
import logging
from tqdm import tqdm
import copy
import jieba
import os

def loadJsonData(path):
    with open(f'{path}', 'r', encoding='utf-8') as fp:
        jsonData = fp.read()
    data = json.loads(jsonData)
    return data

def buildCorpusForW2V(dataset0, saveFilePath):
    '''
        使用 dataset 构建用于训练 word2vec 的 corpus
    '''
    corpus = []
    dataset = copy.deepcopy(dataset0)
    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description('building corpus')
        for index, item in enumerate(dataset):
            curQ = item['Q']
            curQ_detail = item['Q_detailed']
            curA = item['A']

            if curA[-1] != '。':
                curA = curA + '。'

            corpus.extend(jieba.lcut(curQ))
            corpus.extend(jieba.lcut(curQ_detail))
            corpus.extend(jieba.lcut(curA))
            pbar.update(1)

    corpusWordNum = len(set(corpus))
    print(f'该语料一共有 {corpusWordNum} 个词，总词量：{len(corpus)}')
    with open(f'{saveFilePath}', 'w', encoding='utf-8') as fp:
        for item in corpus:
            fp.write(f'{item} ')
    print(f'save {os.path.abspath(saveFilePath)} done')


def trainW2V(path, embeddingSize=150, trainIter=5):
    Word2VecParam = {
        'sg': 1,
        'size': embeddingSize,
        'window': 5,
        'min_count': 1,
        'iter': trainIter,  # 迭代次数
        # 'negative': 3,
        # 'sample': 0.001,
        # 'hs': 1,
        'workers': 4,
    }
    basePath = os.path.split(path)[0]
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(f'{path}')
    model = word2vec.Word2Vec(sentences, **Word2VecParam)
    print('word2vec model training done')
    print(f'word2vec model：{model}')
    savePath = f'{basePath}/word2vec_{embeddingSize}dim_{trainIter}iters.w2v.model'
    model.save(savePath)
    print(f'{os.path.abspath(savePath)} save done')
    vocabulary = {}
    for vocab in model.wv.vocab:
        vector = model[vocab]
        vocabulary[vocab] = vector.tolist()
    savePath2 = f'{basePath}/word2vec_{embeddingSize}dim_{trainIter}iters.w2v.vocab.json'
    with open(f'{savePath2}', 'w',
              encoding='utf-8') as fp:
        vocabularyJson = json.dumps(vocabulary)
        fp.write(vocabularyJson)
    print(f'{savePath2} save done')
    return model