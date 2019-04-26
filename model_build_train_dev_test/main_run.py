import tensorflow as tf
import os
from DataLoader import DataLoader
import gc
import numpy as np
import time
from Dataset_loader import Dataset_loader
from Metric_calc_summary import Metric_calc_summary
from Model import Model
import re
import argparse

def getTrainAndTest(dataloader, val_precent=0.3, randSeed=1):
    num = dataloader.train_val_Num
    trainNum = int(num * (1 - val_precent))
    np.random.seed(randSeed)
    randIndex = np.random.permutation(num)
    trainIndex = randIndex[:trainNum]
    devIndex = randIndex[trainNum:]

    trainData = {}
    trainData['id'], trainData['Q'], trainData['Q_detailed'], \
    trainData['A'], trainData['sentencePos_A'], \
    trainData['sentencePos_Q_detailed'], trainData['label'], \
        = dataloader.allData['train_dev']['id'][trainIndex], \
          dataloader.allData['train_dev']['Q'][trainIndex, :], \
          dataloader.allData['train_dev']['Q_detailed'][trainIndex, :], \
          dataloader.allData['train_dev']['A'][trainIndex, :], \
          dataloader.allData['train_dev']['sentencePos_A'][trainIndex, :], \
          dataloader.allData['train_dev']['sentencePos_Q_detailed'][trainIndex, :], \
          dataloader.allData['train_dev']['label'][trainIndex]

    devData = {}
    devData['id'], devData['Q'], devData['Q_detailed'], \
    devData['A'], devData['sentencePos_A'], \
    devData['sentencePos_Q_detailed'], devData['label'] \
        = dataloader.allData['train_dev']['id'][devIndex], \
          dataloader.allData['train_dev']['Q'][devIndex, :], \
          dataloader.allData['train_dev']['Q_detailed'][devIndex, :], \
          dataloader.allData['train_dev']['A'][devIndex, :], \
          dataloader.allData['train_dev']['sentencePos_A'][devIndex, :], \
          dataloader.allData['train_dev']['sentencePos_Q_detailed'][devIndex, :], \
          dataloader.allData['train_dev']['label'][devIndex]

    dataloader.allData = {}
    gc.collect()

    return trainData, devData

def trainFunc():
    curEpoch = 0
    trainDataNum = trainData['Q'].shape[0]
    graph_train_dev = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    best_dev_F1 = 0
    with tf.Session(graph=graph_train_dev, config=config) as sess:
        # create dataset_loader
        train_dataset_loader = Dataset_loader(batchSize=batchSize, seed=0, mode='train')
        dev_dataset_loader = Dataset_loader(batchSize=batchSize, seed=0, mode='dev')

        # initiate dataset_loader
        train_dataset_loader.initiate(
            sess=sess,
            q_embed_vector=trainData['Q'],
            q_detailed_embed_vector=trainData['Q_detailed'],
            a_embed_vector=trainData['A'],
            sentence_EndPos_A=trainData['sentencePos_A'],
            sentence_EndPos_Q_detailed=trainData['sentencePos_Q_detailed'],
            label=trainData['label'],
        )

        # create metric_calc_summary
        metric = Metric_calc_summary(sess=sess, saveDir=saveSummaryDir)

        # build main model
        model = Model(
            batchSize=batchSize,
            embeddingSize=embeddingSize,
            dropout=dropout,
            hidden_units=hidden_units,
            hops=hops,
            learning_rate=lr
        )
        model.build()

        # create Saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        sess.run(tf.global_variables_initializer())
        # start Train
        while True:
            if curEpoch == epoch:
                break
            q_embed_vector, q_detailed_embed_vector, a_embed_vector,\
            sentence_EndPos_A, sentence_EndPos_Q_detailed, label = train_dataset_loader.getBatchData(sess=sess)
            train_labels, train_logits, step, train_loss = model.train_step(
                sess=sess,
                q_embed_vector=q_embed_vector,
                q_detailed_embed_vector=q_detailed_embed_vector,
                a_embed_vector=a_embed_vector,
                sentence_EndPos_A=sentence_EndPos_A,
                sentence_EndPos_Q_detailed=sentence_EndPos_Q_detailed,
                label=label,
            )
            train_F1, train_accuracy, train_precision, train_recall = metric.write_summaries(
                loss=train_loss,
                labels=train_labels,
                logits=train_logits,
                step=step,
                mode='train'
            )

            if step * batchSize >= trainDataNum * (curEpoch + 1):
                '''
                    finish one epoch, do dev, save model.
                '''
                curEpoch += 1
                print('\033[0;33m'+'='*40 +'dev'+'='*40+'\033[0m')
                # dev
                dev_dataset_loader.initiate(
                    sess=sess,
                    q_embed_vector=devData['Q'],
                    q_detailed_embed_vector=devData['Q_detailed'],
                    a_embed_vector=devData['A'],
                    sentence_EndPos_A=devData['sentencePos_A'],
                    sentence_EndPos_Q_detailed=devData['sentencePos_Q_detailed'],
                    label=devData['label'],
                )
                labels2 = []
                logits2 = []
                loss2 = []
                while True:
                    '''
                        calc all label and logits base on devData
                    '''
                    re = dev_dataset_loader.getBatchData(sess=sess)
                    if re:
                        q_embed_vector, q_detailed_embed_vector, a_embed_vector,\
                        sentence_EndPos_A, sentence_EndPos_Q_detailed, label = re
                        labels, logits, step, loss = model.dev_step(
                            sess=sess,
                            q_embed_vector=q_embed_vector,
                            q_detailed_embed_vector=q_detailed_embed_vector,
                            a_embed_vector=a_embed_vector,
                            sentence_EndPos_A=sentence_EndPos_A,
                            sentence_EndPos_Q_detailed=sentence_EndPos_Q_detailed,
                            label=label,
                        )
                        labels2.extend(labels.tolist())
                        logits2.extend(logits.tolist())
                        loss2.append(loss)
                    else:
                        break
                dev_loss = np.mean(np.array(loss2))

                dev_F1, dev_accuracy, dev_precision, dev_recall = metric.write_summaries(
                    loss=dev_loss,
                    labels=np.array(labels2),
                    logits=np.array(logits2),
                    step=step,
                    mode='dev'
                )

                print(f'\033[0;33mepoch: {curEpoch}, dev=> loss: {dev_loss:.4f}, F1: {dev_F1:.2%}, acc: {dev_accuracy:.2%}, precision: {dev_precision:.2%}, recall: {dev_recall:.2%}\033[0m')

                if best_dev_F1 < dev_F1:
                    '''
                        保存最优的模型
                    '''
                    saver.save(sess=sess, save_path=saveModelDir, global_step=step)
                    best_dev_F1 = dev_F1
                    print('save best model done')

                print('\033[0;33m'+'='*83+'\033[0m')
            if step % display_steps == 0:
                print(f'\033[0;31mepoch: {curEpoch}, eclipse: {step*batchSize%trainDataNum*1.0/trainDataNum:.2%}, loss: {train_loss:.4f}, F1: {train_F1:.2%}, acc: {train_accuracy:.2%}, precision: {train_precision:.2%}, recall: {train_recall:.2%}\033[0m')

def getTestData(dataloader):
    testData = {}
    testData['id'], testData['Q'], testData['Q_detailed'], \
    testData['A'], testData['sentencePos_A'], \
    testData['sentencePos_Q_detailed'], testData['label'], \
        = dataloader.allData['test']['id'], \
          dataloader.allData['test']['Q'], \
          dataloader.allData['test']['Q_detailed'], \
          dataloader.allData['test']['A'], \
          dataloader.allData['test']['sentencePos_A'], \
          dataloader.allData['test']['sentencePos_Q_detailed'], \
          dataloader.allData['test']['label']

    dataloader.allData = {}
    gc.collect()

    return testData

def getBestModelPath(modelTime):
    logs = os.listdir('./logs')
    matchLogs = []
    bestModelPath = []
    for time in modelTime:
            matchLogs.extend([log for index, log in enumerate(logs) if time in log])
    for index, log in enumerate(matchLogs):
        path = f'./logs/{log}/model'
        bestStep = max([int(re.findall('-[0-9]+.', file)[0][1:-1]) for file in os.listdir(path) if '.meta' in file])
        bestModelPath.append(f'{path}/model-{bestStep}')
    return bestModelPath

def testFunc():
    graph_test = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph_test, config=config) as sess:
        test_dataset_loader = Dataset_loader(batchSize=batchSize, seed=0, mode='test')

        # create metric_calc_summary
        metric = Metric_calc_summary(sess=sess, saveDir=None)

        # build main model
        model = Model(
            batchSize=batchSize,
            embeddingSize=embeddingSize,
            dropout=dropout,
            hidden_units=hidden_units,
            hops=hops,
            learning_rate=lr
        )
        model.build()

        # create Saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        sess.run(tf.global_variables_initializer())

        # model restore
        print(f'{bestModelPath}/n model restore successfully!')
        saver.restore(sess, bestModelPath)
        step = sess.run(model.step)

        print('\033[0;33m'+'='*40 +'dev'+'='*40+'\033[0m')
        # test
        test_dataset_loader.initiate(
            sess=sess,
            q_embed_vector=testData['Q'],
            q_detailed_embed_vector=testData['Q_detailed'],
            a_embed_vector=testData['A'],
            sentence_EndPos_A=testData['sentencePos_A'],
            sentence_EndPos_Q_detailed=testData['sentencePos_Q_detailed'],
            label=testData['label'],
        )
        labels2 = []
        logits2 = []
        loss2 = []
        while True:
            '''
                calc all label and logits base on testData
            '''
            re = test_dataset_loader.getBatchData(sess=sess)
            if re:
                q_embed_vector, q_detailed_embed_vector, a_embed_vector,\
                sentence_EndPos_A, sentence_EndPos_Q_detailed, label = re
                labels, logits, step, loss = model.dev_step(
                    sess=sess,
                    q_embed_vector=q_embed_vector,
                    q_detailed_embed_vector=q_detailed_embed_vector,
                    a_embed_vector=a_embed_vector,
                    sentence_EndPos_A=sentence_EndPos_A,
                    sentence_EndPos_Q_detailed=sentence_EndPos_Q_detailed,
                    label=label,
                )
                labels2.extend(labels.tolist())
                logits2.extend(logits.tolist())
                loss2.append(loss)
            else:
                break
        test_loss = np.mean(np.array(loss2))

        test_F1, test_accuracy, test_precision, test_recall = metric.write_summaries(
            loss=test_loss,
            labels=np.array(labels2),
            logits=np.array(logits2),
            step=step,
            mode='test'
        )

        print(f'\033[0;33m test=> loss: {test_loss:.4f}, acc: {test_accuracy:.2%}, precision: {test_precision:.2%}, recall: {test_recall:.2%}\033[0m')
        print('\033[0;33m'+'='*83+'\033[0m')
        return test_loss, test_F1, test_accuracy, test_precision, test_recall

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='set model parameters')
    parser.add_argument('--batch', type=str, help='batch size', default='10')
    parser.add_argument('--epoch', type=str, help='epoch', default='5')
    parser.add_argument('--hidden_units', type=str, help='hidden units in model', default='150')
    parser.add_argument('--hops', type=str, help='memory hos in model', default='3')
    parser.add_argument('--embeddingSize', type=str, help='embedding size of word', default='150')
    parser.add_argument('--dropout', type=str, help='dropout', default='0.0')
    parser.add_argument('--lr', type=str, help='learning rate', default='0.001')
    parser.add_argument('--display_steps', type=str, help='display steps when training model', default='1')
    args = parser.parse_args()

    dataPath = '../data'
    maxA_Len = 300
    maxQ_Len = 35
    maxQ_detailed_Len = 300

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    randSeed = 1000
    hops = int(args.hops)
    hidden_units = int(args.hidden_units)

    batchSize = int(args.batch)
    epoch = int(args.epoch)
    embeddingSize = int(args.embeddingSize)
    dropout = float(args.dropout)
    lr = float(args.lr) # learning rate
    display_steps = int(args.display_steps)

    trainAndDev_dataPath = '../data/dataset_trainAndDev.json'
    test_dataPath = '../data/dataset_test.json'
    w2vFilePath = '../data/word2vec_150dim_1000iters.w2v.vocab.json'

    dataloader = DataLoader(
        trainAndDev_dataPath=trainAndDev_dataPath,
        test_dataPath=test_dataPath,
        w2vFilePath = w2vFilePath,
        silence=False,
        maxA_Len=maxA_Len,
        maxQ_Len=maxQ_Len,
        maxQ_detailed_Len=maxQ_detailed_Len,
    )

    dataloader.getAllData()
    trainData, devData = getTrainAndTest(dataloader, randSeed=randSeed)

    Dir = (
        f'./logs/{time.strftime("%Y-%m-%d@%H_%M_%S", time.localtime())}'
        f'-batchSize{batchSize}'
        f'-epoch{epoch}'
        f'-embeddingSize{embeddingSize}'
        f'-dropout{dropout}'
        f'-hidden_units{hidden_units}'
        f'-hops{hops}'
        f'-lr{lr}'
        f'-randSeed{randSeed}'
    )

    saveSummaryDir = f'{Dir}/summary'
    saveModelDir = f'{Dir}/model/model'
    trainFunc()

    bestModelPath = getBestModelPath([os.path.split(Dir)[1][:19]])[0]
    dataloader.getAllData()
    testData = getTestData(dataloader)
    test_loss, test_F1, test_accuracy, test_precision, test_recall = testFunc()
