# model_build_train_dev_test
build q&a model, use data/dataset_trainAndDev.json to train model, use data/dataset_test.json to test model

# Environment
+ numpy                  1.14.5   
+ tensorflow             1.9.0   

# How to run
~~~
    python3 ./main_run.py --batch=20 --epoch=10 --hidden_units=80 --hops=3 --embeddingSize=150 --dropout=0.0 --lr=0.001 --display_steps=1
~~~

# Model illustration
1. DataLoader  
    processing raw data:
    + data/dataset_trainAndDev.json
    + data/dataset_test.json
    + data/word2vec_150dim_1000iters.w2v.vocab.json
    
2. Dataset_loader  
    use tensorflow.data.Dataset to load dataset. When train, dev or test model, Dataset can generate batch data
    
3. Metric_calc_summary
    define some metrics to evaluate model, and define tensorflow.summary to save metrics in eventFile, eventFile can use for tensorboard
    
4. Model
    define Q&A model by tensorflow
    
# Notice
before running this code, you should run these code firstly: 
1. demo_data/load_demo_data.py 
2. textPreprocessing/main_run.py
3. trainWord2vec/main_run.py
