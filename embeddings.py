"""Собраны ембеддинги на основании библиотеки Gensim"""

def doc2bow_func(model, tokenize_texts):
    return [model.doc2bow(tokenize_text) for tokenize_text in tokenize_texts]


def lda_lsi_tfidf_func(model, tokenize_texts):
    return [model[tokenize_text] for tokenize_text in tokenize_texts]


def doc2vec_func(model, tokenize_texts):
    return [model.infer_vector(tokenize_text) for tokenize_text in tokenize_texts]


def fasttexts_func(model, tokenize_texts):
    def doc2vec(tokenize_text):
        vectors = model.wv.get_vector(tokenize_text[0])
        for word in tokenize_text[1:]:
            vectors = vectors + model.wv.get_vector(word)
        return vectors / len(tokenize_text)

    return [doc2vec(tokenize_text) for tokenize_text in tokenize_texts]


def keras_text2vec_func(model, tokenize_texts):
    return model.predict([" ".join(tx) for tx in tokenize_texts])


def transformer_func(model, texts: []):
    return model.encode(texts)


class Embedding:
    """
    Abstract Factory for Classes which converting texts into vectors.
    """

    def __init__(self, model, vectorization_function, embedding=None):
        self.model = model
        self.embedding = embedding
        self.func = vectorization_function

    def texts_processing(self, texts: []) -> []:
        """Convert list of texts into list of vectors"""

        if self.embedding is None:
            embedded_texts = texts
        else:
            embedded_texts = self.embedding(texts)

        return self.func(self.model, embedded_texts)

    def __call__(self, texts, num_terms=None):
        return self.texts_processing(texts)

