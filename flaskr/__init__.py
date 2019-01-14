import os
import types
from flask import Flask, request, jsonify, g
import numpy as np
import pandas as pd
import sentencepiece as spm
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

values = None
results = None
b_clf = None
processor = None
count_vect = None

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        JSON_AS_ASCII = False,
    )

    if test_config is None: # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        global values,results,b_clf,processor, count_vect
        if (count_vect is None):
            count_vect = CountVectorizer()
        if (values is None and results is None):
            values = []
            results = []
            tsv = pd.read_table(os.path.join(app.instance_path, 'data.tsv')).values.tolist()
            for row in tsv:
                values.append(row[-1])
                results.append(1 if row[1] == 'p' else 0)

        if (b_clf is None):
            X = count_vect.fit_transform(values)
            y = np.array(results)
            b_clf = BernoulliNB()
            b_clf.fit(X, y)
        if (processor is None):
            processor = spm.SentencePieceProcessor()
            processor.Load(os.path.join(app.instance_path, 'model.model'))
        body = request.args.get('body')
        word = ' '.join(processor.EncodeAsPieces(body))
        test_values = [word]
        x_test = count_vect.transform(test_values)
        results = b_clf.predict(x_test)
        return jsonify({ 'result': str(results[0]), 'body': body })

    from . import db
    db.init_app(app)

    return app
