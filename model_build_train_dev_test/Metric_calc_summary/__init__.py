import tensorflow as tf

class Metric_calc_summary():
    def __init__(self, sess, saveDir):
        '''
            base on labels and logits, calculate metric: accuracy, precision, recall
            and add metrics to tf.summary

            sess:
                tf.Session
            saveDir:
                the save dir of tf.summary.writer
                train dir => saveDir/train
                dev dir => saveDir/dev
        '''
        self.labels = tf.placeholder(tf.int32, [None, ])
        self.logits = tf.placeholder(tf.float32, [None, 2])
        self.loss = tf.placeholder(tf.float32, [])

        self.sess = sess
        self.saveDir = saveDir

        if saveDir:
            self.train_summary_writer = tf.summary.FileWriter(f'{saveDir}/train', sess.graph)
            self.dev_summary_writer = tf.summary.FileWriter(f'{saveDir}/dev', sess.graph)

        self.F1, self.accuracy, self.precision, self.recall = self.calc_acc(self.labels, self.logits)

        accuracy_s = tf.summary.scalar('accuracy', self.accuracy)
        precision_s = tf.summary.scalar('precision', self.precision)
        recall_s = tf.summary.scalar('recall', self.recall)
        loss_s = tf.summary.scalar('loss', self.loss)
        F1_s = tf.summary.scalar('F1', self.F1)

        self.summary_op = tf.summary.merge([F1_s, loss_s, accuracy_s, precision_s, recall_s])

    def write_summaries(self, loss, labels, logits, step, mode='train'):
        '''
            labels:
                .shape = (batchSize), real label
            logits:
                .shape = (batchSize, num_classes), predict label present
            model:
                'train' or 'dev'
        '''
        if mode == 'train':
            summaries, F1, accuracy, precision, recall = self.sess.run(
                fetches=[self.summary_op, self.F1, self.accuracy, self.precision, self.recall],
                feed_dict={self.loss: loss, self.labels: labels, self.logits: logits}
            )
            self.train_summary_writer.add_summary(summaries, step)
        elif mode == 'dev':
            summaries, F1, accuracy, precision, recall = self.sess.run(
                fetches=[self.summary_op, self.F1, self.accuracy, self.precision, self.recall],
                feed_dict={self.loss: loss, self.labels: labels, self.logits: logits}
            )
            self.dev_summary_writer.add_summary(summaries, step)
        elif mode == 'test':
            F1, accuracy, precision, recall = self.sess.run(
                fetches=[self.F1, self.accuracy, self.precision, self.recall],
                feed_dict={self.loss: loss, self.labels: labels, self.logits: logits}
            )
        else:
            raise Exception('mode shoube be \'train\' or \'dev\' ')

        return F1, accuracy, precision, recall

    def calc_acc(self, labels, logits):
        '''
            labels.shape = (None, )
            logits.shape = (None, num_classes)

            calculater accuracy, precision, recall

            return
                accuracy, precision, recall
        '''
        pre_labels = tf.argmax(logits, axis=-1, output_type=tf.int32)
        num = tf.shape(labels)[0]
        i = tf.constant(0, dtype=tf.int32)
        TP = tf.constant(0, dtype=tf.float32, name='TP')
        TN = tf.constant(0, dtype=tf.float32, name='TN')
        FP = tf.constant(0, dtype=tf.float32, name='FP')
        FN = tf.constant(0, dtype=tf.float32, name='FN')
        trueFlag = tf.constant(1, dtype=tf.int32, name='trueFlag')
        falseFlag = tf.constant(0, dtype=tf.int32, name='falseFlag')

        def cond(labels, pre_labels, i, TP, TN, FP, FN):
            return tf.not_equal(i, num)

        def body(labels, pre_labels, i, TP, TN, FP, FN):
            cur_label = labels[i]
            cur_pre_label = pre_labels[i]
            TP = tf.cond(
                tf.logical_and(tf.equal(cur_label, trueFlag), tf.equal(cur_label, cur_pre_label)),
                lambda: TP + 1,
                lambda: TP,
            )
            FN = tf.cond(
                tf.logical_and(tf.equal(cur_label, trueFlag), tf.not_equal(cur_label, cur_pre_label)),
                lambda: FN + 1,
                lambda: FN,
            )
            TN = tf.cond(
                tf.logical_and(tf.equal(cur_label, falseFlag), tf.equal(cur_label, cur_pre_label)),
                lambda: TN + 1,
                lambda: TN,
            )
            FP = tf.cond(
                tf.logical_and(tf.equal(cur_label, falseFlag), tf.not_equal(cur_label, cur_pre_label)),
                lambda: FP + 1,
                lambda: FP,
            )
            i = i + 1
            return labels, pre_labels, i, TP, TN, FP, FN

        _, _, i, TP, TN, FP, FN = tf.while_loop(cond, body, [labels, pre_labels, i, TP, TN, FP, FN])

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = tf.cond(tf.not_equal(TP + FP, 0), lambda: TP / (TP + FP), lambda: 0.0)
        recall = tf.cond(tf.not_equal(TP + FN, 0), lambda: TP / (TP + FN), lambda: 0.0)
        F1 = tf.cond(tf.not_equal(recall + precision, 0), lambda: 2 * recall * precision / (recall + precision),
                     lambda: 0.0)

        return F1, accuracy, precision, recall