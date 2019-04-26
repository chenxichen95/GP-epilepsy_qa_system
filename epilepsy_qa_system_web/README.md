# epilepsy_qa_system_web
use Flask build background server, use js, html, and css to write front-end page, and deploy the mode in the background server

# Environment
+ Flask                  1.0.2    
+ numpy                  1.14.5   
+ tensorflow             1.9.0   

# how to run
~~~
    # model is the trained model which save at the model_train_dev_test/logs/
    # e.g. --model ../model_build_train_dev_test/logs/2019-04-26@12_34_08-batchSize20-epoch10-embeddingSize150-dropout0.0-hidden_units80-hops3-lr0.001-randSeed1000/model/model-5
    python ./myServer --model modelPath 
~~~