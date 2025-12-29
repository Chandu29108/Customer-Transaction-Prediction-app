import time, os, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, make_response, Response, render_template
from flask_restx import Api, fields, Resource
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import joblib
import sklearn

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='1.0',
    title='My Customer Transaction Prediction API',
    description='Customer Transaction Prediction API')
ns = api.namespace('transaction_prediction',
                   description='Transaction Prediction')

# Load the scaler and PCA objects
scaler = joblib.load('./models/robust_scaler.pkl')
pca    = joblib.load('./models/pca.pkl')

# Load model
model = joblib.load('./models/mlp_pipeline.pkl')

# Number of features (var_1 to var_200)
number_features = 175

# Parser for /predict (form fields)
parser = api.parser()
for idx in range(number_features):
    parser.add_argument(
        f'var_{idx}',
        type=float,
        required=True,
        help=f'feature {idx}',
        location='form'
    )

parser.add_argument(
    'ID_code',
    type=str,
    required=False,
    help='Customer ID',
    location='form'
)
# Response model
resource_fields = api.model('Resource', {
    'result': fields.List(fields.Float)
})

# Upload parser
upload_parser = api.parser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True
)

@ns.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        # Parse uploaded file
        args = upload_parser.parse_args()
        file = args.get('file')  # This is FileStorage
        # read csv into dataframe
        df = pd.read_csv(file)
        results = self.get_results(df)
        return {
            'url': 'File uploaded successfully',
            'rows': len(df),
            'results': results
        }, 200

    def get_results(self, df):
        # ID_code column (if present)
        id_codes = df['ID_code'] if 'ID_code' in df.columns else ['unknown'] * len(df)

        # Assume all columns except ID_code are features
        features = [c for c in df.columns if c != 'ID_code'][:175]
        X = df[features].values

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # prob of transaction

        # build list of dicts, one per row
        results = []
        for id_code, p, pr in zip(id_codes, predictions, probabilities):
            result = {
                "prediction": int(p),
                "probability": float(pr),
                "ID_code": str(id_code)
            }
            results.append(result)
        return results


@ns.route('/predict')
class PredictionApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
        # Build feature array
        X = []
        for idx in range(number_features):
            X.append(args[f'var_{idx}'])
        X = np.array(X).reshape(1, -1)

        # Apply scaler and PCA transformations
        X = scaler.transform(X)
        X = pca.transform(X)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'ID_code': args.get('ID_code', 'unknown')
        }
        return {'result': [result]}, 200


# --- HTML Routes (for web frontend) ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    # This route is for direct form submission (non-AJAX)
    return render_template('result.html')

# --- AJAX endpoint for form (same as RESTPlus but returns JSON) ---

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Read all var_0 to var_174 from form
        X = []
        for idx in range(number_features):
            val = request.form.get(f'var_{idx}', '0')
            X.append(float(val))
        X = np.array(X).reshape(1, -1)

        # Apply the scaler and PCA transformations
        X = scaler.transform(X)
        X = pca.transform(X)
        # Apply the scaler and PCA transformations
        X = scaler.transform(X)
        X = pca.transform(X)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        id_code = request.form.get('ID_code', 'unknown')

        result = {
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'ID_code': id_code
        }
        return jsonify(result)
    except Exception as e:
                return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(threaded=True, host="0.0.0.0", port=5000, debug=True)





