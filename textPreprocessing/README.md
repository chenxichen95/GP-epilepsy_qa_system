# Text Preprocess
do text preprocess for the dataset in mongo database "epilepsy_qa_system_demo_data", generate files as follow:
+ dataset_trainAndDev.json and dataset_test.json  
    used for train and test model
+ datasetForW2V.json  
    used for train word2vec model
+ epilepsyTrueAns_pool.json  
    used for answer pool in qa system
    
# Environment
+ jieba                  0.39     
+ numpy                  1.14.5   
+ pandas                 0.23.3   
+ pymongo                3.7.1    
+ tqdm                   4.25.0  

# How to run
~~~
    python3 ./main_run.py 
~~~

# Notice
you should run demo_data/load_demo_data.py before running this code.