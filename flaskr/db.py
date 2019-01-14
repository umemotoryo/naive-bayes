import os
import click
from flask import current_app, g

import mysql.connector as mc
import pandas as pd
import pandas.io.sql as pdsql

def get_db():
    if 'db' not in g:
        g.db = mc.connect(
                host = os.environ.get('DB_HOST', 'localhost'),
                port = os.environ.get('DB_PORT', 3306),
                user = os.environ.get('DB_USER', 'root'),
                password = os.environ.get('DB_PASS', ''),
                database = os.environ.get('DB_NAME', ''),
        )
        # g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_app(app):
    app.teardown_appcontext(close_db)
