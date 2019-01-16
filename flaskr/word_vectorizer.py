from sklearn.feature_extraction.text import CountVectorizer

class WordVectorizer(object):
    def __init__(self, values):
        self.vectorizer = CountVectorizer()
        self.values = values

    def get_vectorizer(self):
        return self.vectorizer

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.vectorizer.fit_transform(self.values)
