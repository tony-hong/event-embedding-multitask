''' This module contains non-incremental role-filler.

    Author: Tony Hong
    Designer: Ottokar Tilk
    Ref: Ottokar Tilk, Event participant modelling with neural networks, EMNLP 2016
'''

import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Dense, Lambda, Multiply, Masking, Add
from keras.initializers import glorot_uniform
from keras.layers.advanced_activations import PReLU
from keras.models import Model

from embeddings import role_based_word_embedding
from layers import target_word_hidden
from generic import GenericModel


class NNRF_ROFA(GenericModel):
    """Non-incremental model role-filler

    """
    def __init__(self, n_word_vocab=50001, n_role_vocab=7, n_factors_emb=256, n_factors_cls=512, n_hidden=256, word_vocabulary={}, role_vocabulary={}, 
        unk_word_id=50000, unk_role_id=7, missing_word_id=50001, 
        using_dropout=False, dropout_rate=0.3, optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        super(NNRF_ROFA, self).__init__(n_word_vocab, n_role_vocab, n_factors_emb, n_hidden, word_vocabulary, role_vocabulary, 
            unk_word_id, unk_role_id, missing_word_id, using_dropout, dropout_rate, optimizer, loss, metrics)

        # minus 1 here because one of the role is target role
        self.input_length = n_role_vocab - 1

        # each input is a fixed window of frame set, each word correspond to one role
        input_words = Input(shape=(self.input_length, ), dtype='int32', name='input_words')
        input_roles = Input(shape=(self.input_length, ), dtype='int32', name='input_roles')
        target_role = Input(shape=(1, ), dtype='int32', name='target_role')

        # role based embedding layer
        embedding_layer = role_based_word_embedding(input_words, input_roles, n_word_vocab, n_role_vocab, glorot_uniform(), 
            missing_word_id, self.input_length, n_factors_emb, True, using_dropout, dropout_rate)
        
        # fully connected layer, output shape is (batch_size, input_length, n_hidden)
        hidden = Dense(n_hidden, 
            activation='linear', 
            input_shape=(n_factors_emb,), 
            name='projected_event_embedding')(embedding_layer)

        # non-linear layer, using 1 to initialize
        non_linearity = PReLU(
            alpha_initializer='ones',
            name='PReLU')(hidden)

        # sum on input_length direction;
        # obtaining context embedding layer, shape is (batch_size, n_factors_emb)
        event_embedding = Lambda(lambda x: K.sum(x, axis=1), 
            name='event_embedding',
            output_shape=(hidden,))(non_linearity)


        # hidden layer
        hidden_layer2 = target_word_hidden(event_embedding, target_role, n_word_vocab, n_role_vocab, glorot_uniform(), n_factors_cls, n_hidden, 
            using_dropout=using_dropout, dropout_rate=dropout_rate)

        # softmax output layer
        output_layer = Dense(n_word_vocab, 
            activation='softmax', 
            input_shape=(n_factors_cls, ), 
            name='softmax_word_output')(hidden_layer2)

        self.model = Model(inputs=[input_words, input_roles, target_role], outputs=[output_layer])

        self.model.compile(optimizer, loss, metrics)


    def set_0_bias(self):
        """ This function is used as a hack that set output bias to 0.
            According to Ottokar's advice in the paper, during the *evaluation*, the output bias needs to be 0 
            in order to replicate the best performance reported in the paper.
        """
        word_output_weights = self.model.get_layer("softmax_word_output").get_weights()
        word_output_kernel = word_output_weights[0]
        word_output_bias = np.zeros(self.n_word_vocab)
        self.model.get_layer("softmax_word_output").set_weights([word_output_kernel, word_output_bias])

        return word_output_weights[1]

    def set_bias(self, bias):
        word_output_weights = self.model.get_layer("softmax_word_output").get_weights()
        word_output_kernel = word_output_weights[0]
        self.model.get_layer("softmax_word_output").set_weights([word_output_kernel, bias])

        return bias

    # Deprecated temporarily
    def train(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c, batch_size=256, epochs=100, validation_split=0.05, verbose=0):
        train_result = self.model.fit([i_w, i_r, t_r], t_w_c, batch_size, epochs, validation_split, verbose)
        return train_result

    def test(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c, batch_size=256, verbose=0):
        test_result = self.model.evaluate([i_w, i_r, t_r], t_w_c, batch_size, verbose)
        return test_result

    def train_on_batch(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c):
        train_result = self.model.train_on_batch([i_w, i_r, t_r], t_w_c)
        return train_result

    def test_on_batch(self, i_w, i_r, t_w, t_r, t_w_c, t_r_c, sample_weight=None):
        test_result = self.model.test_on_batch([i_w, i_r, t_r], t_w_c, sample_weight)
        return test_result


    def predict(self, i_w, i_r, t_r, batch_size=1, verbose=0):
        """ Return the output from softmax layer. """
        predict_result = self.model.predict([i_w, i_r, t_r], batch_size, verbose)
        return predict_result

    def summary(self):
        self.model.summary()


    def predict_class(self, i_w, i_r, t_r, batch_size=1, verbose=0):
        """ Return predicted target word from prediction. """
        predict_result = self.predict(i_w, i_r, t_r, batch_size, verbose)
        return np.argmax(predict_result, axis=1)

    def p_words(self, i_w, i_r, t_w, t_r, batch_size=1, verbose=0):
        """ Return the output scores given target words. """
        predict_result = self.predict(i_w, i_r, t_r, batch_size, verbose)
        return predict_result[range(batch_size), list(t_w)]

    def top_words(self, i_w, i_r, t_r, topN=20, batch_size=1, verbose=0):
        """ Return top N target words given context. """
        predict_result = self.predict(i_w, i_r, t_r, batch_size, verbose)
        rank_list = np.argsort(predict_result, axis=1)
        return [r[-topN:][::-1] for r in rank_list]

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

