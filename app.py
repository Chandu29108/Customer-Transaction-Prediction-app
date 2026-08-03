import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_restx import Api, fields, Resource
from flask_cors import CORS
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version="1.0",
    title="Customer Transaction Prediction API",
    description="Predicts whether a customer will make a transaction, "
                 "given 200 anonymized numeric features.",
)
ns = api.namespace("transaction_prediction", description="Transaction Prediction")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Fitted on 200 raw features -> 200 scaled features
scaler = joblib.load(os.path.join(MODELS_DIR, "robust_scaler.pkl"))
# Fitted on 200 scaled features -> 175 principal components
pca = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
# Trained on 175 PCA components. Whichever model scored best in
# src/train.py (Logistic Regression, XGBoost, or MLP) is saved here.
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

# The model expects this many RAW input features (var_0 .. var_199).
# This must match what the scaler was fit on -- NOT the PCA output size.
N_RAW_FEATURES = scaler.n_features_in_
FEATURE_COLUMNS = [f"var_{i}" for i in range(N_RAW_FEATURES)]


def transform_features(X: np.ndarray) -> np.ndarray:
    """Apply the same scaler -> PCA pipeline used at training time."""
    X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X_df)
    X_pca = pca.transform(X_scaled)
    return X_pca


def predict_from_raw(X_raw: np.ndarray):
    """X_raw: (n_samples, N_RAW_FEATURES) -> (predictions, probabilities)."""
    X_pca = transform_features(X_raw)
    predictions = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca)[:, 1]
    return predictions, probabilities


# ---- Swagger request/response models ----

parser = api.parser()
for idx in range(len(FEATURE_COLUMNS)):
    parser.add_argument(f"var_{idx}", type=float, required=True, help=f"feature {idx}", location="form")
parser.add_argument("ID_code", type=str, required=False, help="Customer ID", location="form")

resource_fields = api.model("Resource", {
    "result": fields.List(fields.Raw),
})

upload_parser = api.parser()
upload_parser.add_argument("file", location="files", type=FileStorage, required=True)


@ns.route("/upload/")
@api.expect(upload_parser)
class Upload(Resource):
    """Batch predictions from an uploaded CSV of customer rows."""

    def post(self):
        args = upload_parser.parse_args()
        file = args.get("file")
        df = pd.read_csv(file)

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return {"error": f"CSV is missing {len(missing)} required feature columns, "
                              f"e.g. {missing[:5]}"}, 400

        id_codes = df["ID_code"] if "ID_code" in df.columns else ["unknown"] * len(df)
        X_raw = df[FEATURE_COLUMNS].values

        predictions, probabilities = predict_from_raw(X_raw)

        results = [
            {"ID_code": str(id_code), "prediction": int(p), "probability": float(pr)}
            for id_code, p, pr in zip(id_codes, predictions, probabilities)
        ]
        return {"rows": len(df), "results": results}, 200


@ns.route("/predict")
class PredictionApi(Resource):
    """Single-record prediction via the documented REST endpoint."""

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
        X_raw = np.array([[args[f"var_{i}"] for i in range(len(FEATURE_COLUMNS))]])

        prediction, probability = predict_from_raw(X_raw)

        result = {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "ID_code": args.get("ID_code", "unknown"),
        }
        return {"result": [result]}, 200


# ---- HTML routes ----

@app.route("/")
def index():
    return render_template("index.html", n_features=len(FEATURE_COLUMNS))


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX endpoint used by the HTML form."""
    try:
        X_raw = np.array(
            [[float(request.form.get(f"var_{i}", 0.0)) for i in range(len(FEATURE_COLUMNS))]]
        )
        prediction, probability = predict_from_raw(X_raw)
        id_code = request.form.get("ID_code", "unknown")

        result = {
            "prediction": int(prediction[0]),
            "probability": round(float(probability[0]), 4),
            "ID_code": id_code,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0", port=5000, debug=True)





