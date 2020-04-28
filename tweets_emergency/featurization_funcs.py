import numpy as np

def create_vocabulary(masterset):
    vocabulary = {}
    c = 0
    for i in sorted(masterset):
        vocabulary[i] = c
        c += 1
    return vocabulary


def create_multihot(train_ex, vocab):
    vec = np.zeros(len(vocab))
    for word in train_ex:
        index = vocab.get(word)
        vec[index] = 1
    return vec
        
def create_trainmat(list_texts, vocab):
    train_matrix = np.zeros((len(list_texts), len(vocab)), dtype = np.float32)
    c = 0
    for train_ex in list_texts:
        vec = create_multihot(train_ex, vocab)
        train_matrix[c,:] = vec
        c+=1
    return train_matrix
        