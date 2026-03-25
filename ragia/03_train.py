# %%
import pandas as pd
import numpy as np
import mlflow

from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

from fastembed import TextEmbedding

mlflow.set_tracking_uri("http://192.168.0.18:5000")
mlflow.set_experiment(experiment_id=568283666648943226)

# %%
DENSE_MODEL = "intfloat/multilingual-e5-large"

dense_model = TextEmbedding(DENSE_MODEL)

df = pd.read_excel("https://docs.google.com/spreadsheets/d/1u1MPiL3q4SAfelDeoT39fqxon7EtV0BBMePqk1J9apA/export?format=xlsx&id=1u1MPiL3q4SAfelDeoT39fqxon7EtV0BBMePqk1J9apA&gid=0")
df.head()


X = list(dense_model.passage_embed(df["Pergunta"]))
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
        n_estimators=100,
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