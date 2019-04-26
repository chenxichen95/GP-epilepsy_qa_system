import tensorflow as tf

class Dataset_loader():
    def __init__(self, batchSize=150, seed=0, maxQ_Len=35, maxQ_detailed_Len=300, maxA_Len=300,
                 max_sentenceEndPosNum_A=52, max_sentenceEndPosNum_Q_detailed=41, embeddingSize=150,
                 mode='train'):

        self.seed = seed
        self.batchSize = batchSize
        with tf.name_scope('dataset_loader_allData'):
            self.q_embed_vector = tf.placeholder(
                tf.float32,
                [None, maxQ_Len, embeddingSize],
                name='q_embed_vector',
            )
            self.q_detailed_embed_vector = tf.placeholder(
                tf.float32,
                [None, maxQ_detailed_Len, embeddingSize],
                name='q_detailed_embed_vector',
            )
            self.a_embed_vector = tf.placeholder(
                tf.float32,
                [None, maxA_Len, embeddingSize],
                name='a_embed_vector',
            )
            self.sentence_EndPos_A = tf.placeholder(
                tf.int32,
                [None, max_sentenceEndPosNum_A],
                name='sentence_EndPos_A',
            )
            self.sentence_EndPos_Q_detailed = tf.placeholder(
                tf.int32,
                [None, max_sentenceEndPosNum_Q_detailed],
                name='sentence_EndPos_Q_detailed',
            )
            self.label = tf.placeholder(
                tf.int32,
                [None, ],
                name='label',
            )

        self.dataSet = tf.data.Dataset.from_tensor_slices(
            (
                self.q_embed_vector,
                self.q_detailed_embed_vector,
                self.a_embed_vector,
                self.sentence_EndPos_A,
                self.sentence_EndPos_Q_detailed,
                self.label,
            )
        )

        if mode == 'train':
            self.dataSet = self.dataSet.repeat(None).shuffle(buffer_size=100, seed=self.seed).batch(batchSize)
            self.iterator = self.dataSet.make_initializable_iterator()
            self.next_batch_op = self.iterator.get_next()

        elif mode == 'dev':
            self.dataSet = self.dataSet.repeat(1).batch(batchSize)
            self.iterator = self.dataSet.make_initializable_iterator()
            self.next_batch_op = self.iterator.get_next()
        elif mode == 'test':
            self.dataSet = self.dataSet.repeat(1).batch(batchSize)
            self.iterator = self.dataSet.make_initializable_iterator()
            self.next_batch_op = self.iterator.get_next()
        else:
            raise Exception('mode should be \'train\' or \'dev\' or \'test\'')

    def initiate(self,
                 sess,
                 q_embed_vector,
                 q_detailed_embed_vector,
                 a_embed_vector,
                 sentence_EndPos_A,
                 sentence_EndPos_Q_detailed,
                 label):
        sess.run(self.iterator.initializer, feed_dict=
            {
                self.q_embed_vector: q_embed_vector,
                self.q_detailed_embed_vector: q_detailed_embed_vector,
                self.a_embed_vector: a_embed_vector,
                self.sentence_EndPos_A: sentence_EndPos_A,
                self.sentence_EndPos_Q_detailed: sentence_EndPos_Q_detailed,
                self.label: label,
            }
        )

    def getBatchData(self, sess):
        try:
            q_embed_vector, q_detailed_embed_vector, a_embed_vector, \
            sentence_EndPos_A, sentence_EndPos_Q_detailed, label \
                = sess.run(self.next_batch_op)
        except tf.errors.OutOfRangeError:
            # print('OutOfRangeError')
            return []

        return q_embed_vector, q_detailed_embed_vector, a_embed_vector, \
               sentence_EndPos_A, sentence_EndPos_Q_detailed, label