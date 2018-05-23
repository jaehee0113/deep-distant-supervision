import logging
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

logger = logging.getLogger(__name__)


class NRE:
    """Neural Relation Extractor
    """

    def __init__(self, conf, pre_word2vec=None, activate_fn=tf.nn.tanh):
        pretrained_w2v = conf.pretrained_w2v
        max_position = conf.max_position
        pos_dim = conf.pos_dim
        num_relation = conf.num_relation
        len_sentence = conf.len_sentence
        num_hidden = conf.num_hidden
        batch_size = conf.batch_size
        reg_weight = conf.reg_weight
        network_type = conf.network_type

        # CNN-specific config settings
        num_filters = conf.num_filters
        filter_sizes = [2,3,4]
        word_embedding_dim = 50 #subject to change

        # Note that the first dimension of input_sentence and input_y are different because each row in input_y is per
        # triple, whereas each row in input_sentences corresponds to a single sentence and a triple consists of multiple
        # sentences. We use input_triple_index to align sentences with corresponding label
        # for example, label of input_sentences[input_triple_index[0]:input_triple_index[1]] is input[0]
        self.input_sentences = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_sentence')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_position1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_position2')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, num_relation], name='input_y')
        self.input_triple_index = tf.placeholder(dtype=tf.int32, shape=[None], name='input_triple_index')

        num_sentences = self.input_triple_index[-1]

        with tf.device('/gpu:1'):
            if pretrained_w2v:
                self.word2vec = tf.get_variable(initializer=pre_word2vec, name="word_embedding")
            else:
                self.word2vec = tf.get_variable(shape=[conf.voca_size, conf.word_embedding_dim], name="word_embedding")

            self.pos2vec1 = tf.get_variable(shape=[max_position, pos_dim], name="pos2vec1")
            self.pos2vec2 = tf.get_variable(shape=[max_position, pos_dim], name="pos2vec2")

        # concatenate word embedding + position embeddings
        # input_forward.shape = [num_sentence, len_sentence, w2v_dim+2*conf.pos_dim]
        input_forward = tf.concat([tf.nn.embedding_lookup(self.word2vec, self.input_sentences),
                                   tf.nn.embedding_lookup(self.pos2vec1, self.input_pos1),
                                   tf.nn.embedding_lookup(self.pos2vec2, self.input_pos2)], 2)

        if network_type == 'cnn':
            with tf.device('/gpu:1'):
                #input_forward = tf.unstack(input_forward, len_sentence, 1)
                input_forward = tf.expand_dims(input_forward, -1) #as conv2d expects 4 rank input
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        filter_shape = [filter_size, word_embedding_dim, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                        #Convolution layer
                        conv = tf.nn.conv2d(
                            input_forward,
                            W,
                            strides=[1,1,1,1],
                            padding="SAME",
                            name="conv")

                        #Activation function (ReLu) layer
                        nl = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                        #Max-pooling layer
                        pooled = tf.nn.max_pool(
                            nl,
                            ksize= [1, len_sentence - filter_size +1, 1, 1],
                            strides=[1,1,1,1],
                            padding="SAME",
                            name="pool")

                        pooled_outputs.append(pooled)
                # Combine all pooled features
                num_filters_total = num_filters * len(filter_sizes)
                self.pool = tf.concat(pooled_outputs, 3)

                # m dim : input forward (?)
                sentence_embedding = tf.reshape(self.pool, [-1, num_filters_total])
                sentence_embedding = tf.nn.dropout(sentence_embedding, 0.5)
                #num_hidden = 16 #(sentence_embedding n dimension)
                h_sentence = sentence_embedding
                num_hidden = num_filters_total
        elif network_type == 'rnn':
            with tf.variable_scope("RNN"):
                def create_rnn_cells(num_units):
                    """return list of rnn cells"""
                    cells = [rnn.GRUCell(num_units, activation=activate_fn) for _ in range(conf.num_layer)]
                    if conf.dropout and conf.is_train:
                        return [rnn.DropoutWrapper(cell) for cell in cells]
                    else:
                        return cells

                input_forward = tf.unstack(input_forward, len_sentence, 1)

                # construct rnn with high-level api
                if conf.bidirectional:
                    output_rnn, _, _ = rnn.stack_bidirectional_rnn(
                        create_rnn_cells(num_hidden), 
                        create_rnn_cells(num_hidden), 
                        input_forward,
                        dtype=tf.float32)
                    num_hidden = 2 * num_hidden  # dimension of concatenated fw-bw outputs
                    output_hidden = tf.reshape(tf.concat(output_rnn, 1), [num_sentences, len_sentence, num_hidden])
                else:
                    output_rnn, _ = rnn.static_rnn(
                        create_rnn_cells(num_hidden)[0], 
                        input_forward, 
                        dtype=tf.float32)
                    output_hidden = tf.reshape(tf.concat(output_rnn, 1), [num_sentences, len_sentence, num_hidden])

                # word-level attention layer, represent a sentence as a weighted sum of word vectors
                with tf.variable_scope("word-attn"):
                    if conf.word_attn:
                        word_attn = tf.get_variable('W', shape=[num_hidden, 1])
                        word_weight = tf.matmul(
                            tf.reshape(output_hidden, [num_sentences * len_sentence, num_hidden]),
                            word_attn)
                        word_weight = tf.reshape(word_weight, [num_sentences, len_sentence])
                        sentence_embedding = tf.matmul(
                            tf.reshape(tf.nn.softmax(word_weight), [num_sentences, 1, len_sentence]),
                            output_hidden)
                        sentence_embedding = tf.reshape(sentence_embedding, [num_sentences, num_hidden])
                    else:
                        sentence_embedding = tf.reduce_mean(output_hidden, 1)

            with tf.variable_scope("fc-hidden"):
                h_sentence = tf.layers.dense(sentence_embedding, num_hidden, activation=activate_fn, name='fc-hidden')

        # sentence-level attention layer, represent a triple as a weighted sum of sentences
        with tf.device('/gpu:1'), tf.variable_scope("sentence-attn"):
            attn_weight = tf.get_variable("W", shape=[num_hidden, 1])
            if conf.use_multiplier:
                multiplier = tf.get_variable("A", shape=[num_hidden])
            triple_embeddings = list()

            for i in range(batch_size):
                target_sentences = h_sentence[self.input_triple_index[i]:self.input_triple_index[i + 1]]

                if conf.sent_attn:
                    num_triple_sentence = self.input_triple_index[i + 1] - self.input_triple_index[i]
                    if conf.use_multiplier:
                        tmp = tf.multiply(target_sentences, multiplier)
                    else:
                        tmp = target_sentences
                    sentence_weight = tf.reshape(
                        tf.nn.softmax(tf.reshape(tf.matmul(tmp, attn_weight), [num_triple_sentence])),
                        [1, num_triple_sentence])
                    triple_embedding = tf.squeeze(tf.matmul(sentence_weight, target_sentences))  # [num_hidden]
                else:
                    # use mean vector if sentence-level attention layer is not used
                    triple_embedding = tf.squeeze(tf.reduce_mean(target_sentences, 0))
                triple_embeddings.append(triple_embedding)

            triple_embeddings = tf.reshape(triple_embeddings, [-1, num_hidden])
            triple_output = tf.layers.dense(triple_embeddings, num_relation, name='fc-output')


        # Optmization preparation step
        self.prob = tf.nn.softmax(triple_output)
        self.predictions = tf.argmax(self.prob, axis=1, name="predictions")
        self.total_loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=triple_output, labels=self.input_y), name="loss")
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, tf.argmax(self.input_y, 1)), "float"), name="accuracy")

        tf.summary.scalar("loss", self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(reg_weight),
            weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar("l2_loss", self.l2_loss)
        tf.summary.scalar("final_loss", self.final_loss)
