import TrainWord2vec

if __name__ == "__main__":
    data = TrainWord2vec.loadJsonData('../data/datasetForW2V.json')
    TrainWord2vec.buildCorpusForW2V(data, '../data/corpus_forW2V.txt')
    TrainWord2vec.trainW2V(
        '../data/corpus_forW2V.txt',
        embeddingSize=150,
        trainIter=10 ** 3,
    )
    pass