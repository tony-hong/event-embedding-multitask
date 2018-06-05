''' This module contains embeddings defined in non-incremental model of role-filler. 

    * Role based embedding: role_based_embedding

    Author: Tony Hong
    Designer: Ottokar Tilk
    Ref: Ottokar Tilk, Event participant modelling with neural networks, EMNLP 2016
'''


import numpy as np

from keras import backend as K
from keras.layers import Embedding, Dropout, Dense, Lambda, Multiply, Masking



def role_based_word_embedding(input_words, input_roles, n_word_vocab, n_role_vocab, emb_init, 
    missing_word_id, input_length, n_factors_emb=256, 
    mask_zero=True, using_dropout=False, dropout_rate=0.3):
    """Role-based word embedding combining word and role embedding.

    # Arguments:
        input_words:        place holder for input words, shape is (batch_size, input_length)
        input_roles:        place holder for input roles, shape is (batch_size, input_length)
        n_word_vocab:       size of word vocabulary
        n_role_vocab:       size of role vocabulary
        emb_init:           initializer of embedding
        missing_word_id:    the id used as place-holder for the role without a word appearing
        n_factors_emb:      tensor factorization number, default: 256
        n_sample:           number of samples, useful when there are negative samples
        mask_zero:          bool, zero out the weight of missing word
        using_dropout:      bool, using drop-out layer or not
        dropout_rate:       rate of drop-out layer
    """

    # word embedding; shape is (batch_size, input_length, n_factors_emb)
    word_embedding = Embedding(n_word_vocab, n_factors_emb, 
        embeddings_initializer=emb_init,
        name='org_word_embedding')(input_words)
    
    if mask_zero:
        # a hack zeros out the missing word inputs
        weights = np.ones((n_word_vocab, n_factors_emb))
        weights[missing_word_id] = 0
        mask = Embedding(n_word_vocab, n_factors_emb, 
            weights=[weights], 
            trainable=False,
            name='mask_missing')(input_words)

        # masked word embedding
        word_embedding = Multiply(
            name='word_embedding')([word_embedding, mask])

        # Alternative implementation, need missing_word_id == 0
        # self.word_embedding = Masking(mask_value=0., 
        #     input_shape=(input_length, n_factors_emb)(word_embedding)
    
    # role embedding; shape is (batch_size, input_length, n_factors_emb)
    role_embedding = Embedding(n_role_vocab, n_factors_emb, 
        embeddings_initializer=emb_init,
        name='role_embedding')(input_roles)

    if using_dropout:
        # Drop-out layer after embeddings
        word_embedding = Dropout(dropout_rate)(word_embedding)
        role_embedding = Dropout(dropout_rate)(role_embedding)

    # hidden units after combining 2 embeddings; shape is the same with embedding
    embedding = Multiply()([word_embedding, role_embedding])

    return embedding



def factored_embedding(input_words, input_roles, n_word_vocab, n_role_vocab, emb_init, 
    missing_word_id, input_length, n_factors_emb=256, n_hidden=256, 
    mask_zero=True, using_dropout=False, dropout_rate=0.3, using_bias=False):
    """Role-based word embedding combining word and role embedding.

    # Arguments:
        input_words:        place holder for input words, shape is (batch_size, input_length)
        input_roles:        place holder for input roles, shape is (batch_size, input_length)
        n_word_vocab:       size of word vocabulary
        n_role_vocab:       size of role vocabulary
        emb_init:           initializer of embedding
        missing_word_id:    the id used as place-holder for the role without a word appearing
        n_factors_emb:      tensor factorization number, default: 256
        n_sample:           number of samples, useful when there are negative samples
        mask_zero:          bool, zero out the weight of missing word
        using_dropout:      bool, using drop-out layer or not
        dropout_rate:       rate of drop-out layer
    """

    # word embedding; shape is (batch_size, input_length, n_factors_emb)
    word_embedding = Embedding(n_word_vocab, n_factors_emb, 
        embeddings_initializer=emb_init,
        name='org_word_embedding')(input_words)
    
    if mask_zero:
        # a hack zeros out the missing word inputs
        weights = np.ones((n_word_vocab, n_factors_emb))
        weights[missing_word_id] = 0
        mask = Embedding(n_word_vocab, n_factors_emb, 
            weights=[weights], 
            trainable=False,
            name='mask_missing')(input_words)

        # masked word embedding
        word_embedding = Multiply(name='word_embedding')([word_embedding, mask])

        # Alternative implementation, need missing_word_id == 0
        # self.word_embedding = Masking(mask_value=0., 
        #     input_shape=(input_length, n_factors_emb)(word_embedding)
    
    # role embedding; shape is (batch_size, input_length, n_factors_emb)
    role_embedding = Embedding(n_role_vocab, n_factors_emb, 
        embeddings_initializer=emb_init,
        name='role_embedding')(input_roles)

    if using_dropout:
        # Drop-out layer after embeddings
        word_embedding = Dropout(dropout_rate)(word_embedding)
        role_embedding = Dropout(dropout_rate)(role_embedding)

    # hidden units after combining 2 embeddings; shape is the same with embedding
    hidden = Multiply(name='multiply_composition')([word_embedding, role_embedding])

    # fully connected layer, output shape is (batch_size, input_length, n_hidden)
    embedding = Dense(n_hidden, 
        activation='linear', 
        use_bias=using_bias,
        input_shape=(n_factors_emb,), 
        name='role_based_word_embedding')(hidden)

    return embedding


