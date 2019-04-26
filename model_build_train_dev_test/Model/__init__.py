import numpy as np
import tensorflow as tf

class BI_GRU():
    def __init__(self, num_units=150, initial_state_fw=None, initial_state_bw=None):
        '''
            bidirectional GRU Cell
        '''
        self.num_units = num_units
        self.initial_state_fw = initial_state_fw
        self.initial_state_bw = initial_state_bw
        self.cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.num_units)
        self.cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.num_units)

    def __call__(self, inputs, sequence_length):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_bw=self.initial_state_bw,
            initial_state_fw=self.initial_state_fw,
            dtype=tf.float32,
        )
        return (outputs, output_states)


class attentionGru():
    def __init__(self, hidden_units):
        '''
            the GRU base on the attention

        '''
        self.hidden_units = hidden_units
        self.sentencePadFlag = 99999
        # attention gscore weights define
        self.W1 = tf.Variable(
            np.random.randn(hidden_units * 4 * 2, hidden_units) / np.sqrt(hidden_units * 4 * 2),
            name='W1',
            dtype=tf.float32,
        )
        self.b1 = tf.Variable(np.zeros((hidden_units,)), name='b1', dtype=tf.float32)
        self.W2 = tf.Variable(
            np.random.randn(hidden_units, 1) / np.sqrt(hidden_units),
            name='W2',
            dtype=tf.float32
        )
        self.b2 = tf.Variable(np.zeros((1,)), name='b2', dtype=tf.float32)
        self.g_score_list = []  # save the g_scores in each memory

        # gru weights define
        self.Wr = tf.Variable(
            np.random.randn(hidden_units * 2, hidden_units * 2) / np.sqrt(hidden_units * 2),
            name='attention_gru_Wr',
            dtype=tf.float32,
        )
        self.Ur = tf.Variable(
            np.random.randn(hidden_units * 2, hidden_units * 2) / np.sqrt(hidden_units),
            name='attention_gru_Ur',
            dtype=tf.float32,
        )
        self.br = tf.Variable(np.zeros((1, hidden_units * 2)), name='attention_gru_br', dtype=tf.float32)
        self.W = tf.Variable(
            np.random.randn(hidden_units * 2, hidden_units * 2) / np.sqrt(hidden_units * 2),
            name='attention_gru_W',
            dtype=tf.float32,
        )
        self.U = tf.Variable(
            np.random.randn(hidden_units * 2, hidden_units * 2) / np.sqrt(hidden_units),
            name='attention_gru_U',
            dtype=tf.float32,
        )
        self.bh = tf.Variable(np.zeros((1, hidden_units * 2)), name='attention_gru_bh', dtype=tf.float32)

        # MLP : for updating memory
        self.Wt = tf.Variable(
            np.random.randn(hidden_units * 2 * 3, hidden_units * 2) / np.sqrt(hidden_units * 2 * 3),
            name='updata_memory_Wt',
            dtype=tf.float32,
        )
        self.bt = tf.Variable(np.zeros((1, hidden_units * 2)), name='updata_memory_bt', dtype=tf.float32)

    def __call__(self, fact, q, pre_m, sentence_EndPos):
        '''
            fact.shape = (batchSize, max_sentenceEndPosNum, hidden_unis*2)
            q.shape = (batchSize, hidden_unis*2)
            pre_m.shape = (batchSize, hidden_unis*2)
            sentence_EndPos.shape = (batchSize, max_sentenceEndPosNum)
                use for dynamic calculate RNN output in dynamic length sequence input

            return:
                new memory

        '''

        def calc_gscore(inputs):
            '''
                inputs = [curFact, curQ, curPre_m]
                curFact.shape = (max_sentenceEndPosNum, hidden_unis*2)
                curQ.shape = (hidden_unis*2, )
                curPre_m.shape = (hidden_unis*2, )
            '''
            curFact, curQ, curPre_m = inputs
            z = tf.concat(
                [
                    tf.multiply(curFact, curQ),
                    tf.multiply(curFact, curPre_m),
                    tf.abs(tf.subtract(curFact, curQ)),
                    tf.abs(tf.subtract(curFact, curPre_m)),
                ],
                axis=-1,
            )
            Z = tf.add(tf.matmul(tf.nn.tanh(tf.add(tf.matmul(z, self.W1), self.b1)), self.W2), self.b2)
            g_score = tf.nn.softmax(Z, axis=-1, name='g_score')
            return g_score

        g_score = tf.squeeze(tf.map_fn(calc_gscore, [fact, q, pre_m], dtype=tf.float32), axis=-1)
        self.g_score_list.append(g_score)

        def calc_c(inputs):
            '''
                use attention GRU , base on fact and g_score , calculate the c for updating memory.
                inputs = [curFact, cur_g_score, cur_sentence_EndPos]
                curFact.shape = (max_sentenceEndPosNum, hidden_unis*2)
                cur_g_score.shape = (max_sentenceEndPosNum, )
                cur_sentence_EndPos.shape = (max_sentenceEndPosNum, )

                return
                    c
            '''
            curFact, cur_g_score, cur_sentence_EndPos = inputs
            i = tf.constant(0, dtype=tf.int32)
            c = tf.zeros((1, self.hidden_units * 2), dtype=tf.float32)
            max_sentenceEndPosNum = tf.shape(curFact)[0]

            def cond(curFact, cur_g_score, cur_sentence_EndPos, i, c):
                '''
                    tf.logical_and 支持 短路运算，即第一个为假时，第二个表达式不再进行运算。
                '''
                return tf.logical_and(tf.not_equal(i, max_sentenceEndPosNum - 1),
                                      tf.not_equal(cur_sentence_EndPos[i], self.sentencePadFlag))

            def body(curFact, cur_g_score, cur_sentence_EndPos, i, c):
                curFact_i = tf.expand_dims(curFact[i], axis=0)
                g_i = cur_g_score[i]
                ri = tf.nn.sigmoid(tf.matmul(curFact_i, self.Wr) + tf.matmul(c, self.Ur) + self.br)
                h_tilde_i = tf.nn.tanh(tf.matmul(curFact_i, self.W) + tf.multiply(tf.matmul(c, self.U), ri) + self.bh)
                c = tf.multiply(h_tilde_i, g_i) + tf.multiply(c, (1 - g_i))
                i = i + 1
                return curFact, cur_g_score, cur_sentence_EndPos, i, c

            _, _, _, _, c = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=[curFact, cur_g_score, cur_sentence_EndPos, i, c],
                shape_invariants=[curFact.shape, cur_g_score.shape, cur_sentence_EndPos.shape, i.shape, c.shape]
            )
            return tf.squeeze(c, axis=0)

        c = tf.map_fn(calc_c, [fact, g_score, sentence_EndPos], dtype=tf.float32, name='c')

        def update_memory(c, pre_m, q):
            '''
                c.shape = (batchSize, hidden_units*2)
                pre_m.shape = (batchSize, hidden_units*2)
                q.shape = (batchSize, hidden_units*2)

                return:
                    m : new_memory
                        m.shape = (batchSize, hidden_units*2)
            '''
            m = tf.nn.relu(tf.matmul(tf.concat([pre_m, c, q], axis=-1), self.Wt) + self.bt)
            return m

        return update_memory(c, pre_m, q)


class MLP():
    def __init__(self, hidden_units):
        '''
            two layers MLP
        '''
        self.hidden_units = hidden_units
        self.W1 = tf.Variable(
            np.random.randn(hidden_units * 2 * 2, hidden_units) / np.sqrt(hidden_units * 2 * 2),
            dtype=tf.float32,
            name='W1',
        )
        self.b1 = tf.Variable(np.zeros((1, hidden_units)), dtype=tf.float32, name='b1')
        self.W2 = tf.Variable(
            np.random.randn(hidden_units, 2) / np.sqrt(hidden_units),
            dtype=tf.float32,
            name='W2',
        )
        self.b2 = tf.Variable((np.zeros(2, )), dtype=tf.float32, name='b2')

    def __call__(self, inputs):
        '''
            let inputs through 2 Layers MLP
            inputs.shape = (batch, feature)

            the last layer use softmax, but define in the loss calculate

            return
                outputs : outpus.shape = (batch, )
        '''
        layers1_out = tf.sigmoid(tf.matmul(inputs, self.W1) + self.b1)
        outputs = tf.matmul(layers1_out, self.W2) + self.b2
        return outputs


class Model():
    def __init__(self, batchSize, maxQ_Len=35, maxQ_detailed_Len=300, maxA_Len=300, max_sentenceEndPosNum_A=52,
                 max_sentenceEndPosNum_Q_detailed=41, embeddingSize=150, dropout=0, hidden_units=150,
                 sentencePad=99999, hops=3, learning_rate=0.001
                 ):
        self.batchSize = batchSize
        self.maxQ_Len = maxQ_Len
        self.maxQ_detailed_Len = maxQ_detailed_Len
        self.maxA_Len = maxA_Len
        self.max_sentenceEndPosNum_A = max_sentenceEndPosNum_A
        self.max_sentenceEndPosNum_Q_detailed = max_sentenceEndPosNum_Q_detailed
        self.embeddingSize = embeddingSize
        self.dropout = dropout
        self.hidden_units = hidden_units  # the numbers of contextual layer bi-gru's hidden unit
        self.sentencePad = sentencePad
        self.hops = hops

        self.q_embed_vector = tf.placeholder(
            tf.float32,
            [None, self.maxQ_Len, self.embeddingSize],
            name='q_embed_vector',
        )
        self.q_detailed_embed_vector = tf.placeholder(
            tf.float32,
            [None, self.maxQ_detailed_Len, self.embeddingSize],
            name='q_detailed_embed_vector',
        )
        self.a_embed_vector = tf.placeholder(
            tf.float32,
            [None, self.maxA_Len, self.embeddingSize],
            name='a_embed_vector',
        )
        self.sentence_EndPos_A = tf.placeholder(
            tf.int32,
            [None, self.max_sentenceEndPosNum_A],
            name='sentence_EndPos_A',
        )
        self.sentence_EndPos_Q_detailed = tf.placeholder(
            tf.int32,
            [None, self.max_sentenceEndPosNum_Q_detailed],
            name='sentence_EndPos_Q_detailed',
        )
        self.label = tf.placeholder(
            tf.int32,
            [None, ],
            name='label',
        )

        self.step = tf.Variable(0, dtype=tf.int32, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def build(self):
        '''
            build model
        '''
        q_vector, q_detailed_vector, a_sentence_vector = self.contextualLayer(
            q_embed_vector=self.q_embed_vector,
            q_detailed_embed_vector=self.q_detailed_embed_vector,
            a_embed_vector=self.a_embed_vector,
            sentence_EndPos_Q_detailed=self.sentence_EndPos_Q_detailed,
            sentence_EndPos_A=self.sentence_EndPos_A,
        )
        final_memory = self.memoryLayer(
            q_vector=q_vector,
            q_detailed_vector=q_detailed_vector,
            a_sentence_vector=a_sentence_vector,
            sentence_EndPos_A=self.sentence_EndPos_A,
        )
        self.logits = self.answerLayer(final_memory)

        self.loss = self.calc_loss(label=self.label, logits=self.logits)

        self.apply_optimize = self.optimize()

    def train_step(self, sess, q_embed_vector, q_detailed_embed_vector, a_embed_vector,
                   sentence_EndPos_A, sentence_EndPos_Q_detailed, label
                   ):
        '''
            inputs:
                sess,
                q_embed_vector,
                q_detailed_embed_vector,
                a_embed_vector,
                sentence_EndPos_A,
                sentence_EndPos_Q_detailed,
                label
        '''
        re = sess.run(
            fetches=[self.label, self.logits, self.step, self.apply_optimize, self.loss],
            feed_dict={
                self.q_embed_vector: q_embed_vector,
                self.q_detailed_embed_vector: q_detailed_embed_vector,
                self.a_embed_vector: a_embed_vector,
                self.sentence_EndPos_A: sentence_EndPos_A,
                self.sentence_EndPos_Q_detailed: sentence_EndPos_Q_detailed,
                self.label: label
            }
        )
        labels, logits, step, _, loss = re

        return labels, logits, step, loss

    def dev_step(self, sess, q_embed_vector, q_detailed_embed_vector, a_embed_vector,
                 sentence_EndPos_A, sentence_EndPos_Q_detailed, label):
        re = sess.run(
            fetches=[self.label, self.logits, self.step, self.loss],
            feed_dict={
                self.q_embed_vector: q_embed_vector,
                self.q_detailed_embed_vector: q_detailed_embed_vector,
                self.a_embed_vector: a_embed_vector,
                self.sentence_EndPos_A: sentence_EndPos_A,
                self.sentence_EndPos_Q_detailed: sentence_EndPos_Q_detailed,
                self.label: label
            }
        )
        labels, logits, step, loss = re
        return labels, logits, step, loss

    def test_step(self, sess, q_embed_vector, q_detailed_embed_vector, a_embed_vector,
                 sentence_EndPos_A, sentence_EndPos_Q_detailed):
        logits = sess.run(
            fetches=self.logits,
            feed_dict={
                self.q_embed_vector: q_embed_vector,
                self.q_detailed_embed_vector: q_detailed_embed_vector,
                self.a_embed_vector: a_embed_vector,
                self.sentence_EndPos_A: sentence_EndPos_A,
                self.sentence_EndPos_Q_detailed: sentence_EndPos_Q_detailed,
            }
        )
        return logits

    def optimize(self):
        '''
            optimize all trainable weights
        '''
        grad_var = self.optimizer.compute_gradients(loss=self.loss)
        apply_optimize = self.optimizer.apply_gradients(grad_var, global_step=self.step)
        '''
            step 会随着每次迭代更新 + 1
        '''
        return apply_optimize

    def contextualLayer(self, q_embed_vector, q_detailed_embed_vector, a_embed_vector,
                        sentence_EndPos_Q_detailed, sentence_EndPos_A
                        ):
        '''
            使用 biGRU 进行上下文信息编码
            inputs:
                q_embed_vector
                q_detailed_embed_vector
                a_embed_vector
                sentence_EndPos_Q_detailed,
                sentence_EndPos_A
        '''
        with tf.name_scope('contextual_layer') as scope:
            # dropout
            q_embed_vector = tf.nn.dropout(
                q_embed_vector,
                keep_prob=(1 - self.dropout),
                noise_shape=(self.batchSize, self.maxQ_Len, self.embeddingSize),
            )
            q_detailed_embed_vector = tf.nn.dropout(
                q_detailed_embed_vector,
                keep_prob=(1 - self.dropout),
                noise_shape=(self.batchSize, self.maxQ_detailed_Len, self.embeddingSize),
            )
            a_embed_vector = tf.nn.dropout(
                a_embed_vector,
                keep_prob=(1 - self.dropout),
                noise_shape=(self.batchSize, self.maxA_Len, self.embeddingSize),
            )

            with tf.variable_scope('q_contextual_scope') as scope:
                q_bi_gru = BI_GRU(num_units=self.hidden_units)
                _, q_contextual_vector = q_bi_gru(
                    q_embed_vector,
                    self.calcSequenceLength(q_embed_vector),
                )
                # 提取 GRU 隐含层单元最后的输出作为问题 Q 的向量表示。
                q_vector = tf.concat(q_contextual_vector, axis=-1)

            with tf.variable_scope('q_detailed_contextual_scope') as scope:
                q_detailed_bi_gru = BI_GRU(num_units=self.hidden_units)
                q_detailed_contextual_vector, _ = q_detailed_bi_gru(
                    q_detailed_embed_vector,
                    self.calcSequenceLength(q_detailed_embed_vector),
                )
                q_detailed_contextual_vector = tf.concat(q_detailed_contextual_vector, axis=-1)
                # 提取 Q_detailed 的句子表示。
                q_detailed_sentence_vector = self.getSentenceRepresentation(
                    inputs=q_detailed_contextual_vector,
                    sentenceEndPos=sentence_EndPos_Q_detailed,
                    max_sentenceEndPosNum=self.max_sentenceEndPosNum_Q_detailed,
                )
                # mean pooling q_detailed_sentence_vector
                q_detailed_vector = tf.reduce_mean(q_detailed_sentence_vector, axis=1)

            with tf.variable_scope('a_contextual_scope') as scope:
                a_bi_gru = BI_GRU(num_units=self.hidden_units)
                a_contextual_vector, _ = a_bi_gru(
                    a_embed_vector,
                    self.calcSequenceLength(a_embed_vector),
                )
                a_contextual_vector = tf.concat(a_contextual_vector, axis=-1)
                # 提取 A 的句子表示。
                a_sentence_vector = self.getSentenceRepresentation(
                    inputs=a_contextual_vector,
                    sentenceEndPos=sentence_EndPos_A,
                    max_sentenceEndPosNum=self.max_sentenceEndPosNum_A,
                )

        return q_vector, q_detailed_vector, a_sentence_vector

    def memoryLayer(self, q_vector, q_detailed_vector, a_sentence_vector, sentence_EndPos_A):
        with tf.variable_scope('memoryLayer'):
            aGru_q = attentionGru(self.hidden_units)
            memory_q = [q_vector]
            for i in range(self.hops):
                memory_q.append(
                    aGru_q(a_sentence_vector, q_vector, memory_q[-1], sentence_EndPos_A)
                )

            aGru_q_detialed = attentionGru(self.hidden_units)
            memory_q_detailed = [q_detailed_vector]
            for i in range(self.hops):
                memory_q_detailed.append(
                    aGru_q_detialed(a_sentence_vector, q_detailed_vector, memory_q_detailed[-1], sentence_EndPos_A)
                )

        return tf.concat([memory_q[-1], memory_q_detailed[-1]], axis=-1)

    def answerLayer(self, final_memory):
        with tf.variable_scope('answerLayer'):
            mlp = MLP(self.hidden_units)
            logits = mlp(final_memory)
        return logits

    def calc_loss(self, label, logits):
        '''
            calculate model loss
        '''
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits)
        return loss

    def calcSequenceLength(self, inputs):
        '''
            计算 padding 后的不定长序列的实际长度
        '''
        return tf.reduce_sum(tf.to_int32(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0)), axis=-1)

    def getSentenceRepresentation(self, inputs, sentenceEndPos, max_sentenceEndPosNum):
        '''
            根据 sentenceEndPos ， 提出 inputs 中的句子表示。
        '''

        def extractVector(x):
            curBatchInput = x[0]
            pos = x[1]
            pos_mask = tf.not_equal(pos, self.sentencePad)
            pos_delPad = tf.boolean_mask(pos, pos_mask)
            extracted_inputs = tf.gather(curBatchInput, pos_delPad)
            padding = tf.zeros((max_sentenceEndPosNum - tf.shape(pos_delPad)[0], self.hidden_units * 2))
            extracted_inputs = tf.concat([extracted_inputs, padding], axis=0)
            return extracted_inputs

        sentence_Rep = tf.map_fn(extractVector, (inputs, sentenceEndPos), tf.float32)
        return sentence_Rep