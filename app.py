
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Vehicle Classification",
    page_icon="🚗",
    layout="wide"
)

st.title("Vehicle Silhouette Classification")
st.write("Machine Learning Assignment 2")

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

@st.cache_resource
def load_models():
    models = {}
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        models[name] = joblib.load(path)

    encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    return models, encoder

models, encoder = load_models()

st.success("All five trained models loaded successfully!")

selected_model = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

uploaded_file = st.file_uploader(
    "Upload Vehicle Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(data.head())

    if "class" not in data.columns:
        st.error("CSV must contain the class column.")
    else:
        X = data.drop(columns=["class"])
        y = data["class"]

        expected_features = list(
            models[selected_model].feature_names_in_
        )

        if set(X.columns) != set(expected_features):
            st.error(
                "Uploaded CSV does not contain the expected features."
            )
        else:
            X = X[expected_features]

            model = models[selected_model]

            predictions = model.predict(X)

            predicted_labels = encoder.inverse_transform(
                predictions.astype(int)
            )

            st.subheader("Predictions")

            prediction_df = pd.DataFrame({
                "Actual Class": y,
                "Predicted Class": predicted_labels
            })

            st.dataframe(prediction_df)

            if set(y).issubset(set(encoder.classes_)):

                y_encoded = encoder.transform(y)

                probabilities = model.predict_proba(X)

                st.subheader("Model Performance")

                metrics = {
                    "Accuracy": accuracy_score(
                        y_encoded, predictions
                    ),
                    "AUC": roc_auc_score(
                        y_encoded,
                        probabilities,
                        multi_class="ovr",
                        average="macro"
                    ),
                    "Precision": precision_score(
                        y_encoded,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_encoded,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_encoded,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "MCC": matthews_corrcoef(
                        y_encoded, predictions
                    )
                }

                columns = st.columns(3)

                for i, (name, value) in enumerate(metrics.items()):
                    columns[i % 3].metric(
                        name, f"{value:.4f}"
                    )

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_encoded,
                    predictions,
                    labels=np.arange(len(encoder.classes_))
                )

                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=encoder.classes_,
                        columns=encoder.classes_
                    )
                )

                st.subheader("Classification Report")

                st.text(
                    classification_report(
                        y_encoded,
                        predictions,
                        target_names=encoder.classes_,
                        zero_division=0
                    )
                )

            csv = prediction_df.to_csv(index=False)

            st.download_button(
                "Download Predictions",
                csv,
                "vehicle_predictions.csv",
                "text/csv"
            )
