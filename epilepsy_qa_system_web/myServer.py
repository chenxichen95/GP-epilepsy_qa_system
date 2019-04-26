import sys
sys.path.append('../model_build_train_dev_test')

from flask import (
    Flask,
    request,
    send_file,
)
import tensorflow as tf
import numpy as np
from Dataloader_server import DataLoader
from Model import Model
from Dataset_loader import Dataset_loader
import os
import time
import re
import argparse



app = Flask(__name__)

@app.route('/sendData', methods=['POST'])
def sendData():
    q = request.form['q']
    q_detailed = request.form['q_detailed']

    inputData = dataloader.getModelServerInputData(q, q_detailed, testSpeed=False)

    logits = getLogits(inputData, model1, sess_model1, dataset_loader, sess_dataset, batchSize=150)
    ans = getAns(logits)

    if q:
        return ans
    else:
        return 'error'

@app.route('/')
def mainRoot():
    return send_file("./index.html")

@app.route('/vendor/bootstrap/css/bootstrap.min.css')
def bootstrap():
    return send_file('./vendor/bootstrap/css/bootstrap.min.css')

@app.route('/css/util.css')
def util():
    return send_file('./css/util.css')

@app.route('/css/main.css')
def main_css():
    return send_file('./css/main.css')

@app.route('/vendor/jquery/jquery-3.2.1.min.js')
def js():
    return send_file('./vendor/jquery/jquery-3.2.1.min.js')

@app.route('/images/bg-01.jpg')
def bg():
    return send_file('./images/bg-01.jpg')

@app.route('/favicon.ico')
def icon():
    return ('./images/icons/favicon.ico')

@app.route('/fonts/poppins/Poppins-Regular.ttf')
def Poppins_Regular():
    return send_file('./fonts/poppins/Poppins-Regular.ttf')

@app.route('/fonts/montserrat/Montserrat-Bold.ttf')
def Montserrat_Bold():
    return send_file('./fonts/montserrat/Montserrat-Bold.ttf')


def getLogits(inputData, model, sess_model, dataset_loader, sess_dataset, batchSize=150):
    dataset_loader.initiate(
        sess=sess_dataset,
        q_embed_vector=inputData['Q'],
        q_detailed_embed_vector=inputData['Q_detailed'],
        a_embed_vector=inputData['A'],
        sentence_EndPos_A=inputData['sentencePos_A'],
        sentence_EndPos_Q_detailed=inputData['sentencePos_Q_detailed'],
        label=np.zeros((dataloader.ansNum), dtype=np.int32),
    )

    logits = np.zeros((1, 2), dtype=np.float32)

    while True:
        '''
            calc all label and logits base on testData
        '''
        re =dataset_loader.getBatchData(sess=sess_dataset)
        if re:
            q_embed_vector, q_detailed_embed_vector, a_embed_vector, \
            sentence_EndPos_A, sentence_EndPos_Q_detailed, label = re
            logits_step = model.test_step(
                sess=sess_model,
                q_embed_vector=q_embed_vector,
                q_detailed_embed_vector=q_detailed_embed_vector,
                a_embed_vector=a_embed_vector,
                sentence_EndPos_A=sentence_EndPos_A,
                sentence_EndPos_Q_detailed=sentence_EndPos_Q_detailed,
            )
            logits = np.concatenate([logits, logits_step], axis=0)
            break
        else:
            break


    return logits[1:][:, 1]

def getAns(logits):
    maxLogitIndex = np.argmax(logits)
    ans = dataloader.ansPool[maxLogitIndex]

    return ans['A']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='the trained model used for qa system web server')
    args = parser.parse_args()

    dataloader = DataLoader(
        dataPath='../data',
        ansPoolPath='../data/epilepsyTrueAns_pool.json',
        w2vFilePath='../data/word2vec_150dim_1000iters.w2v.vocab.json',
    )
    bestModelPath = args.model
    if not bestModelPath:
        raise Exception("can\'t find the model")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph_model1 = tf.Graph()
    sess_model1 = tf.Session(graph=graph_model1, config=config)

    graph_dataset = tf.Graph()
    sess_dataset = tf.Session(graph=graph_dataset, config=config)

    batchSize = int(re.findall('[0-9]+',re.findall('batchSize[0-9]+-', bestModelPath)[0])[0])
    embeddingSize = int(re.findall('[0-9]+',re.findall('embeddingSize[0-9]+-', bestModelPath)[0])[0])
    dropout = float(re.findall('[0-9.]+', re.findall('dropout[0-9.]+-', bestModelPath)[0])[0])
    hidden_units = int(re.findall('[0-9]+', re.findall('hidden_units[0-9]+-', bestModelPath)[0])[0])
    hops = int(re.findall('[0-9]+', re.findall('hops[0-9]+-', bestModelPath)[0])[0])
    lr = float(re.findall('[0-9.]+', re.findall('lr[0-9.]+-', bestModelPath)[0])[0])

    with graph_model1.as_default():
        model1 = Model(
            batchSize=batchSize,
            embeddingSize=embeddingSize,
            dropout=dropout,
            hidden_units=hidden_units,
            hops=hops,
            learning_rate=lr
        )
        model1.build()
        saver_model1 = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    saver_model1.restore(sess_model1, bestModelPath)
    print(f'restore from {bestModelPath} successfully!')

    with graph_dataset.as_default():
        with tf.device('/cpu:0'):
            dataset_loader = Dataset_loader(mode='test')

    time.sleep(2)

    config = dict(
        debug=False,
        host='0.0.0.0',
        port=8080,
    )
    app.run(**config)
    print('server started!!')
