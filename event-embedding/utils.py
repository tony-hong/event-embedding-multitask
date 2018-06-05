# coding: utf-8

import numpy as np

def get_reverse_map(dictionary):
    return {v:k for k,v in dictionary.iteritems()}

def shuffle_arrays(*arrays):
    import numpy as np
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.set_state(rng_state)
        np.random.shuffle(array)
    
def input_word_index(vocabulary, input_word, unk_id, warn_unk=False):
    if warn_unk and input_word not in vocabulary:
        print "Warning: %s not in vocabulary" % input_word
    return vocabulary.get(input_word, unk_id)
