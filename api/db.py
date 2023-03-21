import MySQLdb
import os
import numpy as np
from typing import Dict, List
from numpy.typing import NDArray


def connect():
    """ https://mysqlclient.readthedocs.io/user_guide.html#mysqldb """
    return MySQLdb.connect(host=os.environ['MYSQL_HOST'],
                           user=os.environ['MYSQL_USER'],
                           passwd=os.environ['MYSQL_ROOT_PASSWORD'],
                           db=os.environ['MYSQL_DATABASE'])


def drop_table() -> None:
    """
    Drop the table if exists.
    """
    db = connect()
    cur = db.cursor()
    try:
        cur.execute("""DROP TABLE models""")
        db.commit()
        db.close()
    except Exception as e:
        db.close()


def trunc_table_models() -> None:
    """
    Clear the table if exists.
    """
    db = connect()

    cur = db.cursor()
    try:
        cur.execute("""TRUNCATE TABLE models""")
        db.commit()
        db.close()
    except Exception as e:
        db.close()


def create_table_models() -> None:
    """
    Create the models table.
    """
    db = connect()

    cur = db.cursor()

    try:
        cur.execute("""CREATE TABLE models (
                       id INT PRIMARY KEY AUTO_INCREMENT,
                       model VARBINARY(50000) NOT NULL,
                       model_class INT NOT NULL,
                       params VARBINARY(1000) NOT NULL,
                       n_dims INT NOT NULL,
                       n_trained INT NOT NULL DEFAULT 0,
                       n_classes INT NOT NULL DEFAULT 2)""")
        db.commit()
        db.close()
    except Exception as e:
        db.close()


def query_model(_id: int) -> Dict:
    """
    Query the given model id, if exists return a dictionary with results, otherwise an empty dictionary.
    """
    db = connect()
    cur = db.cursor()
    q = """SELECT model_class, n_dims, n_classes, n_trained, model, params FROM models WHERE id = %s"""
    cur.execute(q, (_id,))
    r = cur.fetchone()
    db.close()

    if r:
        result = {'model': r[0],
                  'd': r[1],
                  'n_classes': r[2],
                  'n_trained': r[3],
                  'model_bin': r[4],
                  'params': r[5]
                  }
    else:
        result = {}

    return result


def update_model(_id: int, model_bin: bytes, n_trained: int) -> None:
    """
    Update model binary and n_trained values
    """
    db = connect()
    cur = db.cursor()
    q = """UPDATE models SET model=%s, n_trained=%s WHERE id = %s"""
    cur.execute(q, (model_bin, n_trained, _id))
    db.commit()
    db.close()


def insert_model(model_bin: bytes, model_params: bytes, model_cls: str, n_dims: int, n_classes: int) -> int:
    """
    Insert a new model instance and return the unique id generated.
    """
    db = connect()
    cur = db.cursor()

    query = f"""INSERT INTO models (model, params, model_class, n_dims, n_classes)
                        VALUES (%s, %s, %s, %s, %s)"""

    _ = cur.execute(query, (model_bin, model_params, model_cls, n_dims, n_classes))
    model_id = db.insert_id()
    db.commit()
    db.close()

    return model_id


def query_all_models() -> NDArray:
    """
    Fetch `model id, model class name, num trained` for all models in the db and
    return as numpy array.
    """

    db = connect()
    cur = db.cursor()

    # get the model counts to initiate the numpy array effectively
    q = "SELECT count(1) from models"
    cur.execute(q)
    r = cur.fetchone()
    n_rows = r[0]

    # get all models and create a numpy array
    q = """SELECT id, model_class, n_trained FROM models"""
    cur.execute(q)
    model_array = np.fromiter(cur.fetchall(), count=n_rows, dtype=(int, 3))
    db.close()

    return model_array


def group_models() -> List[Dict]:
    """
    Fetch all models, group by the number of times they are trained, for each group list the model ids.
    Return the groups as a list of Dictionaries.
    """

    db = connect()
    cur = db.cursor()

    # get all models and create a numpy array
    q = """SELECT n_trained, GROUP_CONCAT(id SEPARATOR ',') FROM models GROUP BY n_trained"""
    cur.execute(q)
    groups = []
    r = cur.fetchone()
    while r:
        res = {'n_trained': r[0], 'model_ids': list(map(int, r[1].split(',')))}
        groups.append(res)
        r = cur.fetchone()
    db.close()

    return groups

