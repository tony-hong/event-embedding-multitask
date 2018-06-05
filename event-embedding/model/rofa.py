''' This module contains multi-task non-incremental role-filler.

    Author: Tony Hong
    Version: 4
    Ref: Ottokar Tilk, Event participant modelling with neural networks, EMNLP 2016
'''

import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Dense, Lambda, Multiply, Masking
from keras.initializers import glorot_uniform
from keras.layers.advanced_activations import PReLU
from keras.models import Model, load_model

from embeddings import factored_embedding
from layers import target_word_hidden, target_role_hidden
from generic import GenericModel


class MTRFv4(GenericModel):
    """Multi-task non-incremental role-filler

    """
    def __init__(self, n_word_vocab=50001, n_role_vocab=7, n_factors_emb=300, n_hidden=300, word_vocabulary=None, role_vocabulary=None, 
        unk_word_id=50000, unk_role_id=7, missing_word_id=50001, 
        using_dropout=False, dropout_rate=0.3, optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['accuracy'], loss_weights=[1., 1.]):
        super(MTRFv4, self).__init__(n_word_vocab, n_role_vocab, n_factors_emb, n_hidden, word_vocabulary, role_vocabulary, 
            unk_word_id, unk_role_id, missing_word_id, using_dropout, dropout_rate, optimizer, loss, metrics)

        # minus 1 here because one of the role is target role
        input_length = n_role_vocab - 1

        n_factors_cls = n_hidden

        # each input is a fixed window of frame set, each word correspond to one role
        input_words = Input(shape=(input_length, ), dtype='int32', name='input_words')
        input_roles = Input(shape=(input_length, ), dtype='int32', name='input_roles')
        target_word = Input(shape=(1, ), dtype='int32', name='target_word')
        target_role = Input(shape=(1, ), dtype='int32', name='target_role')

        # role based embedding layer
        embedding_layer = factored_embedding(input_words, input_roles, n_word_vocab, n_role_vocab, glorot_uniform(), 
            missing_word_id, input_length, n_factors_emb, n_hidden, True, using_dropout, dropout_rate)

        # non-linear layer, using 1 to initialize
        non_linearity = PReLU(alpha_initializer='ones')(embedding_layer)

        # mean on input_length direction;
        # obtaining context embedding layer, shape is (batch_size, n_hidden)
        context_embedding = Lambda(lambda x: K.mean(x, axis=1), 
            name='context_embedding',
            output_shape=(n_hidden,))(non_linearity)
            
        # target word hidden layer
        tw_hidden = target_word_hidden(context_embedding, target_role, n_word_vocab, n_role_vocab, glorot_uniform(), n_hidden, n_hidden, 
            using_dropout=using_dropout, dropout_rate=dropout_rate)

        # target role hidden layer
        tr_hidden = target_role_hidden(context_embedding, target_word, n_word_vocab, n_role_vocab, glorot_uniform(), n_hidden, n_hidden, 
            using_dropout=using_dropout, dropout_rate=dropout_rate)

        # softmax output layer
        target_word_output = Dense(n_word_vocab, 
            activation='softmax', 
            input_shape=(n_hidden, ), 
            name='softmax_word_output')(tw_hidden)

        # softmax output layer
        target_role_output = Dense(n_role_vocab, 
            activation='softmax', 
            input_shape=(n_hidden, ), 
            name='softmax_role_output')(tr_hidden)

        self.model = Model(inputs=[input_words, input_roles, target_word, target_role], outputs=[target_word_output, target_role_output])

        self.model.compile(optimizer, loss, metrics, loss_weights)


    def set_0_bias(self):
        word_output_weights = self.model.get_layer("softmax_word_output").get_weights()
        word_output_kernel = word_output_weights[0]
        word_output_bias = np.zeros(self.n_word_vocab)
        self.model.get_layer("softmax_word_output").set_weights([word_output_kernel, word_output_bias])

        role_output_weights = self.model.get_layer("softmax_role_output").get_weights()
        role_output_kernel = role_output_weights[0]
        role_output_bias = np.zeros(self.n_role_vocab)
        self.model.get_layer("softmax_role_output").set_weights([role_output_kernel, role_output_bias])

        return word_output_weights[1], role_output_weights[1]

    def set_bias(self, bias):
        word_output_weights = self.model.get_layer("softmax_word_output").get_weights()
        word_output_kernel = word_output_weights[0]
        self.model.get_layer("softmax_word_output").set_weights([word_output_kernel, bias[0]])

        role_output_weights = self.model.get_layer("softmax_role_output").get_weights()
        role_output_kernel = role_output_weights[0]
        self.model.get_layer("softmax_role_output").set_weights([role_output_kernel, bias[1]])

        return bias

    # Train and test
    # Deprecated temporarily
    def train(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c, batch_size=256, epochs=100, validation_split=0.05, verbose=0):
        train_result = self.model.fit([i_w, i_r, t_w, t_r], [t_w_c, t_r_c], batch_size, epochs, validation_split, verbose)
        return train_result

    def test(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c, batch_size=256, verbose=0):
        test_result = self.model.evaluate([i_w, i_r, t_w, t_r], [t_w_c, t_r_c], batch_size, verbose)
        return test_result

    def train_on_batch(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c):
        train_result = self.model.train_on_batch([i_w, i_r, t_w, t_r], [t_w_c, t_r_c])
        return train_result

    def test_on_batch(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c, sample_weight=None):
        test_result = self.model.test_on_batch([i_w, i_r, t_w, t_r], [t_w_c, t_r_c], sample_weight)
        return test_result


    def predict(self, i_w, i_r, t_w, t_r, batch_size=1, verbose=0):
        """ Return the output from softmax layer. """
        predict_result = self.model.predict([i_w, i_r, t_w, t_r], batch_size, verbose)
        return predict_result

    def predict_word(self, i_w, i_r, t_w, t_r, batch_size=1, verbose=0):
        """ Return predicted target word from prediction. """
        predict_result = self.predict(i_w, i_r, t_w, t_r, batch_size, verbose)
        return np.argmax(predict_result[0], axis=1)

    def predict_role(self, i_w, i_r, t_w, t_r, batch_size=1, verbose=0):
        """ Return predicted target role from prediction. """
        predict_result = self.predict(i_w, i_r, t_w, t_r, batch_size, verbose)
        return np.argmax(predict_result[1], axis=1)

    def p_words(self, i_w, i_r, t_w, t_r, batch_size=1, verbose=0):
        """ Return the output scores given target words. """
        predict_result = self.predict(i_w, i_r, t_w, t_r, batch_size, verbose)
        return predict_result[0][range(batch_size), list(t_w)]

    def p_roles(self, i_w, i_r, t_w, t_r, batch_size=1, verbose=0):
        """ Return the output scores given target roles. """
        predict_result = self.predict(i_w, i_r, t_w, t_r, batch_size, verbose)
        return predict_result[1][range(batch_size), list(t_r)]

    def top_words(self, i_w, i_r, t_w, t_r, topN=20, batch_size=1, verbose=0):
        """ Return top N target words given context. """
        predict_result = self.predict(i_w, i_r, t_w, t_r, batch_size, verbose)[0]
        rank_list = np.argsort(predict_result, axis=1)[0]
        return rank_list[-topN:][::-1]
        # return [r[-topN:][::-1] for r in rank_list]
        
    # TODO
    def list_top_words(self, i_w, i_r, t_r, topN=20, batch_size=1, verbose=0):
        """ Return a list of decoded top N target words.
            (Only for reference, can be removed.)
        """
        top_words_lists = self.top_words(i_w, i_r, t_r, topN, batch_size, verbose)
        print type(top_words_lists)
        result = []
        for i in range(batch_size):
            top_words_list = top_words_lists[i]
            result.append([self.word_decoder[w] for w in top_words_list])
        return result


    def summary(self):
        self.model.summary()


