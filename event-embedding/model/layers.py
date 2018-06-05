''' This module contains layers defined in non-incremental model of role-filler. 

    * Output layer of NNRF: target_word_classifier

    Author: Tony Hong
    Designer: Ottokar Tilk
    Ref: Ottokar Tilk, Event participant modelling with neural networks, EMNLP 2016
'''


from keras import backend as K
from keras.layers import Embedding, Dropout, Dense, Lambda, Add, Multiply, Dot, Masking, LSTM, GRU

from embeddings import role_based_word_embedding



def target_word_hidden(inputs, target_role, n_word_vocab, n_role_vocab, emb_init, 
    n_factors_cls=512, n_hidden=256, using_dropout=False, dropout_rate=0.3):
    """Hidden layer of non-incremental model role-filler to predict target word given context 
    (input words, input roles, target role).

    # Args:
        inputs:         output of context embedding from the last layer, shape is (batch_size, input_length)
        target_role:    place holder for target roles, shape is (batch_size, 1)
        n_word_vocab:   size of word vocabulary
        n_role_vocab:   size of role vocabulary
        emb_init:       initializer of embedding
        n_hidden:       number of hidden units
        using_dropout:      bool, using drop-out layer or not
        dropout_rate:       rate of drop-out layer        
    # Return:
        (n_factors_cls, )
    """
    
    # target role embedding; shape is (batch_size, 1, n_factors_cls)
    target_role_embedding = Embedding(n_role_vocab, n_factors_cls, 
        embeddings_initializer=emb_init, 
        name='target_role_embedding')(target_role)

    if using_dropout:
        # Drop-out layer after embeddings
        target_role_embedding = Dropout(dropout_rate)(target_role_embedding)

    # reduce dimension of tensor from 3 to 2
    target_role_embedding = Lambda(lambda x: K.sum(x, axis=1),
        output_shape=(n_factors_cls,))(target_role_embedding)
    
    # context_emb after linear projection
    weighted_context_embedding = Dense(n_factors_cls, 
        activation='linear', 
        use_bias=False,
        input_shape=(n_hidden, ))(inputs)

    # if using_dropout:
    #     # Drop-out layer after fully connected layer
    #    weighted_context_embedding = Dropout(0.5)(weighted_context_embedding)

    # hidden units after combining 2 embeddings; shape is the same with embedding
    hidden = Multiply()([weighted_context_embedding, target_role_embedding])

    return hidden


def target_role_hidden(inputs, target_word, n_word_vocab, n_role_vocab, emb_init, 
    n_factors_cls=512, n_hidden=256, using_dropout=False, dropout_rate=0.3):
    """Hidden layer of multi-task non-incremental model role-filler to predict target role given context 
    (input words, input roles, target word).

    # Args:
        context_emb:    output of context embedding from the last layer, shape is (batch_size, input_length)
        target_word:    place holder for target word, shape is (batch_size, 1)
        n_word_vocab:   size of word vocabulary
        n_role_vocab:   size of role vocabulary
        emb_init:       initializer of embedding
        n_hidden:       number of hidden units
    # Return:
        (n_factors_cls, )
    """
    
    # target role embedding; shape is (batch_size, 1, n_factors_emb)
    target_word_embedding = Embedding(n_word_vocab, n_factors_cls, 
        embeddings_initializer=emb_init, 
        name='target_word_embedding')(target_word)

    if using_dropout:
        target_word_embedding = Dropout(dropout_rate)(target_word_embedding)

    # reduce dimension of tensor from 3 to 2
    target_word_embedding = Lambda(lambda x: K.sum(x, axis=1),
        output_shape=(n_factors_cls,))(target_word_embedding)
    
    # context_emb after linear projection
    weighted_context_embedding = Dense(n_factors_cls, 
        activation='linear', 
        use_bias=False,        
        input_shape=(n_hidden, ))(inputs)

    # if using_dropout:
    #     weighted_context_embedding = Dropout(0.5)(weighted_context_embedding)

    # hidden units after combining 2 embeddings; shape is the same with embedding
    hidden = Multiply()([weighted_context_embedding, target_word_embedding])

    return hidden



def input_hidden(input_words, input_roles, n_word_vocab, n_role_vocab, emb_init, missing_word_id, 
    n_factors_emb=256, n_hidden=256, n_sample=1, mask_zero=True, using_dropout=False, dropout_rate=0.3, 
    activation='linear', a_target=False):
    """Input layer designed by Ottokar

        Embedding layers are initialized with glorot uniform.
        batch_size is None during compile time.
        input_length is length of input_words/input_roles

    # Arguments:
        input_words:        place holder for input words, shape is (batch_size, input_length)
        input_roles:        place holder for input roles, shape is (batch_size, input_length)
        n_word_vocab:       size of word vocabulary
        n_role_vocab:       size of role vocabulary
        emb_init:           initializer of embedding
        missing_word_id:    the id used as place-holder for the role without a word appearing
        n_factors_emb:      tensor factorization number
        n_hidden:           number of hidden units
        n_sample:           number of samples, useful when there are negative samples
        mask_zero:          bool, zero out the weight of missing word
        using_dropout:      bool, using drop-out layer or not
        dropout_rate:       rate of drop-out layer
        activation:         activation function in fully connected layer
        is_target:          bool, True if this is a target embedding

    # if a_target:
    #     input_length = n_sample
    # else:
    #     input_length = n_role_vocab - 1
    """
    hidden = role_based_word_embedding(input_words, input_roles, n_word_vocab, n_role_vocab, emb_init, 
        missing_word_id, n_factors_emb, mask_zero, using_dropout, dropout_rate)

    if a_target:
        # fully connected layer, output shape is (batch_size, n_sample, n_hidden)
        output = Dense(n_hidden, 
            activation=activation, 
            use_bias=False,
            input_shape=(n_sample, n_factors_emb,),
            name='target_role_based_embedding')(hidden)

    else:
        # sum on input_length direction;
        # obtaining context embedding layer, shape is (batch_size, n_factors_emb)
        context_hidden = Lambda(lambda x: K.sum(x, axis=1), 
            name='context_hidden',
            output_shape=(n_factors_emb,))(hidden)

        # fully connected layer, output shape is (batch_size, n_hidden)
        output = Dense(n_hidden, 
            activation=activation, 
            use_bias=True,
            input_shape=(n_factors_emb,), 
            name='role_based_embedding')(context_hidden)

    # if using_dropout:
    #     # Drop-out layer after fully connected layer
    #     output = Dropout(0.5)(output)

    return output
