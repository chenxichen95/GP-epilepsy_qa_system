import pickle
import pymongo
from pymongo.errors import ServerSelectionTimeoutError
import sys

def loadData2Mongo():
    with open('./demo_data.pkl', 'rb') as fp:
        data_b = fp.read()
    data = pickle.loads(data_b)
    collectionName = []
    for item in data:
        collectionName.append(item['type_id'])
    collectionName = set(collectionName)

    try:
        client = pymongo.MongoClient("mongodb://localhost:27017")
        db = client['epilepsy_qa_system_demo_data']

        if db.list_collection_names():
            print('database epilepsy_qa_system_demo_data exist!')
            sys.exit()
        collection = {}
        for name in collectionName:
            collection[name] = db[name]
        for item in data:
            department = item['type_id']
            del item['type_id']
            del item['_id']
            collection[department].insert(item)
            pass
    except ServerSelectionTimeoutError as error:
        print('\33[0;31;0mcan not connect local mongo server(mongodb://localhost:27017), please start mongo server\033[0m')
        sys.exit()
    else:
        print('dumps demo_data in mongodb successfully')

if __name__ == "__main__":
    loadData2Mongo()


