import pandas as pd
from flask import Flask, request, jsonify
from db import create_table_models, trunc_table_models, \
    insert_model, query_model, update_model, query_all_models, group_models, drop_table
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
import pickle
import base64
import ast
from enum import Enum
import uwsgi


application = Flask(__name__)
# drop_table()
create_table_models()


class Classifier(Enum):
    SGDClassifier = 0
    CategoricalNB = 1
    MLPClassifier = 2


@application.route('/health/', methods=['GET'])
def check_health():
    """
    Always return "status: ok"
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'})


@application.route('/trunc_table/', methods=['GET'])
def trunc_table():
    """
    When called, truncate the models table.
    """
    if request.method == 'GET':
        trunc_table_models()
        return jsonify({'status': 'success'})


@application.route('/models/', methods=['POST', 'GET'])
def get_or_create_models():
    """
    Allows creation of a new model instance or querying all models in the database with `training_score`.
    """

    # create a model
    if request.method == 'POST':
        body = request.json
        model = body.get('model', None)
        model_params = body.get('params', {})
        d = body.get('d', None)
        n_classes = body.get('n_classes', None)

        # check for missing parameters
        if None in [model, d, n_classes]:
            return "Missing Parameters", 400

        # check for valid model
        if model == Classifier.SGDClassifier.name:
            model_class = SGDClassifier
        elif model == Classifier.CategoricalNB.name:
            model_class = CategoricalNB
        elif model == Classifier.MLPClassifier.name:
            model_class = MLPClassifier
        else:
            return "Invalid model", 400

        # create model
        try:
            clf = model_class(**model_params)
        except TypeError:
            return "Check params", 400
        except Exception as e:
            return f"Model creation error: {e}", 500

        # serialize model as binary string
        model_bin = pickle.dumps(clf)
        params_bin = pickle.dumps(clf.get_params())

        # insert the model in the database
        try:
            model_id = insert_model(model_bin, params_bin, Classifier[model].value, d, n_classes)
        except Exception as e:
            return f"Unable to Insert: {e}", 400

        return jsonify({'id': model_id})

    # get all models with `training_score`
    if request.method == 'GET':

        model_array = query_all_models()

        # add a column of zeros to hold `training_score`
        model_array = np.c_[model_array, np.zeros(model_array.shape[0])]

        # for each valid model class
        for c in Classifier:
            mask = model_array[:, 1] == c.value

            # get the `n_trained` values for the subset of models belonging to the current model class
            c_n_trained = model_array[mask, 2]

            # set the scores
            if c_n_trained.size == 1:
                model_array[mask, 3] = 1.0
            elif c_n_trained.size > 0:
                # get the ranks of the models
                ranks = np.argsort(np.argsort(c_n_trained))
                R = len(ranks)

                # rank normalization: (r-1) / (R-1)
                # since ranks using numpy argsort will start at 0, r-1 is replaced with r
                model_array[mask, 3] = ranks / (R - 1)

        df = pd.DataFrame(model_array, columns=['id', 'model', 'n_trained', 'training_score'])
        df['id'] = df.id.astype('int')
        df['n_trained'] = df.n_trained.astype('int')

        # replace the integer representation of the model class with string representation
        df['model'] = df.model.apply(lambda x: Classifier(x).name)
        result = {'models': df.to_dict(orient='records')}

        return jsonify(result)


@application.route('/models/<int:model_id>/', methods=['GET'])
def get_model(model_id):
    if request.method == 'GET':

        # query the model
        result = query_model(model_id)
        if not result:
            return "Model does not exist", 404

        # delete model binary string from the result
        del result['model_bin']
        # decode model params
        result['params'] = pickle.loads(result['params'])

        # replace the integer representation of model class with string
        result['model'] = Classifier(result['model']).name

        return jsonify(result)


@application.route('/models/<int:model_id>/train/', methods=['POST'])
def train_model(model_id):
    if request.method == 'POST':

        uwsgi.lock()
        try:
            # query the model
            result = query_model(model_id)
            if not result:
                return "Model does not exist", 404

            body = request.json
            x = body.get('x', None)
            y = body.get('y', None)

            # check for missing parameters
            if None in [x, y]:
                return "Missing Parameters", 400

            # check the feature vector dimension matches with the model to be trained
            if len(x) != result['d']:
                return "Dimension mismatch", 400
            # convert the features to 2D vector
            X = np.array(x).reshape((1, -1))

            # check that given sample class is valid for the model to be trained
            if y not in range(result['n_classes']):
                return "Invalid Class Value", 400

            # convert the class to 1D vector
            y = np.array(y).reshape((1,))

            clf = pickle.loads(result['model_bin'])
            try:
                clf.partial_fit(X, y, classes=range(result['n_classes']))
            except Exception as e:
                return f"Error while training: {e}", 500

            # serialize the updated model to db, increment the `n_trained` value
            update_model(model_id, pickle.dumps(clf), result['n_trained'] + 1)
        finally:
            uwsgi.unlock()

        return jsonify({'status': 'success'})


@application.route('/models/<int:model_id>/predict/', methods=['GET'])
def get_prediction(model_id):
    if request.method == 'GET':

        # query the model
        result = query_model(model_id)
        if not result:
            return "Model does not exist", 404

        # decode the base64 string into a feature list
        b64str = request.args.get('x')
        b64bytes = b64str.encode('ascii')
        bstr = base64.b64decode(b64bytes)
        msg = bstr.decode('ascii')
        x = ast.literal_eval(msg)

        # check the feature vector dimension matches with the model to be trained
        if len(x) != result['d']:
            return "Dimension mismatch", 400
        # convert the features to 2D vector
        X = np.array(x).reshape((1, -1))

        clf = pickle.loads(result['model_bin'])

        y = clf.predict(X)

        return jsonify({'x': x, 'y': int(y)})


@application.route('/models/groups/', methods=['GET'])
def get_model_groups():
    if request.method == 'GET':
        return jsonify({'groups': group_models()})




