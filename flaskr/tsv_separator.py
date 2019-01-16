import pandas as pd

class TsvSeparator(object):
    def __init__(self, tsv_path):
        self.values = []
        self.results = []
        self.tsv_path = tsv_path

    def get_values(self):
        return self.values

    def get_results(self):
        return self.results

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        tsv = pd.read_table(self.tsv_path).values.tolist()
        for row in tsv:
            self.values.append(row[-1])
            self.results.append(1 if row[1] == 'p' else 0)
