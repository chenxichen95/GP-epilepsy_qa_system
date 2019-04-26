import TextPreprocessing
import pymongo
import os

if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client['epilepsy_qa_system_demo_data']
    collectionNames = db.collection_names()
    SEED = 1000

    # get epilepsy dataset from demo_data
    illnessSet_epilepsy = TextPreprocessing.processingData(isEpilepsy=True, collectionNames=collectionNames, db=db)

    # get no-epilepsy dataset from demo_data
    illnessSet_noEpilepsy = TextPreprocessing.processingData(isEpilepsy=False, collectionNames=collectionNames, db=db)

    # build negative data
    illnessSet_epilepsy_negative = TextPreprocessing.buildNegative(
        illnessSet_epilepsy,
        illnessSet_noEpilepsy,
        rate=1,
        seed=SEED,
    )

    dataset_trainAndDev, dataset_test = TextPreprocessing.deal_concat_random_cut(illnessSet_epilepsy, illnessSet_noEpilepsy)

    # save dataset
    dataPath = '../data'
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)

    TextPreprocessing.jsonSave(dataset_trainAndDev, f'{dataPath}/dataset_trainAndDev.json')
    TextPreprocessing.jsonSave(dataset_test, f'{dataPath}/dataset_test.json')
    TextPreprocessing.jsonSave(illnessSet_noEpilepsy + illnessSet_epilepsy, f'{dataPath}/datasetForW2V.json')
    TextPreprocessing.jsonSave(illnessSet_epilepsy, f'{dataPath}/epilepsyTrueAns_pool.json')
