# %%

import os

import dotenv
dotenv.load_dotenv()

import pandas as pd
import numpy as np
import mlflow

from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

from fastembed import TextEmbedding

MLFLOW_URI = os.getenv("MLFLOW_URI")
DATA_PATH = os.getenv("DATA_PATH")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(experiment_name='ragia_guardrails')

# %%
DENSE_MODEL = "intfloat/multilingual-e5-large"

dense_model = TextEmbedding(DENSE_MODEL)


df = pd.read_excel(DATA_PATH)
df.head()

# %%

df.shape

# %%
X = list(dense_model.passage_embed(df["Pergunta"].tolist()))
y = df["Resposta"].tolist()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Treino Resposta", np.mean(y_train) )
print("Teste Resposta", np.mean(y_test) )

# %%

with mlflow.start_run():

    clf = ensemble.RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    pred_probs = clf.predict_proba(X_train)
    preds = clf.predict_proba(X_train)[:,1] > np.mean(y_train)
    acc_train = metrics.accuracy_score(y_train, preds)
    auc_train = metrics.roc_auc_score(y_train, pred_probs[:,1])
    print(metrics.confusion_matrix(y_train, preds))

    pred_probs = clf.predict_proba(X_test)
    preds = clf.predict_proba(X_test)[:,1] > np.mean(y_train)
    acc_test = metrics.accuracy_score(y_test, preds)
    auc_test = metrics.roc_auc_score(y_test, pred_probs[:,1])
    print(metrics.confusion_matrix(y_test, preds))

    mlflow.log_metrics({
        "acc_train":acc_train,
        "auc_train":auc_train,
        "acc_test":acc_test,
        "auc_test":auc_test,
    })

    clf.fit(X,y)

    mlflow.sklearn.log_model(clf, "model")

# %%

print(np.mean(y_train))
# %%
