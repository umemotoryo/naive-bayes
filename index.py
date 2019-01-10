import os
import types
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import sentencepiece as spm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/')
def index():
    count_vect = CountVectorizer()
    values = []
    results = []
    tsv = pd.read_table("tmp/data.tsv").values.tolist()
    for row in tsv:
        values.append(row[-1])
        results.append(1 if row[1] == 'p' else 0)
    X = count_vect.fit_transform(values)
    y = np.array(results)
    processor = spm.SentencePieceProcessor()
    processor.Load('tmp/model.model')
    body = request.args.get('body')
    word = ' '.join(processor.EncodeAsPieces(body))
    test_values = [word]
    count_vect.fit(values)
    x_test = count_vect.transform(test_values)
    b_clf = BernoulliNB()
    b_clf.fit(X, y)
    results = b_clf.predict(x_test)
    return jsonify({ 'result': str(results[0]), 'body': body })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('FLASK_ENV') == 'development')
