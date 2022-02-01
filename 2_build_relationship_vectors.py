import gensim

# Loading wordvector models
home = expanduser("~")
w2v_model = KeyedVectors.load(f"{home}/Documents/data/pretrained_word_vectors/enwiki.model")