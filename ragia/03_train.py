# %%
import pandas as pd
import numpy as np

from fastembed import TextEmbedding

DENSE_MODEL = "intfloat/multilingual-e5-large"

dense_model = TextEmbedding(DENSE_MODEL)

df = pd.read_excel("https://docs.google.com/spreadsheets/d/1u1MPiL3q4SAfelDeoT39fqxon7EtV0BBMePqk1J9apA/export?format=xlsx&id=1u1MPiL3q4SAfelDeoT39fqxon7EtV0BBMePqk1J9apA&gid=0")
df.head()

# %%

X = list(dense_model.passage_embed(df["Pergunta"]))
y = df["Resposta"].tolist()

# %%

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


print("Treino Resposta", np.mean(y_train) )
print("Teste Resposta", np.mean(y_test) )

# %%

clf = ensemble.RandomForestClassifier(n_estimators=100,
                                      min_samples_leaf=3,
                                      random_state=42)

clf.fit(X_train, y_train)

pred_probs = clf.predict_proba(X_train)
preds = clf.predict_proba(X_train)[:,1] > np.mean(y_train)

metrics.confusion_matrix(y_train, preds)

# %%

pred_probs = clf.predict_proba(X_test)
preds = clf.predict_proba(X_test)[:,1] > np.mean(y_train)

metrics.confusion_matrix(y_test, preds)
# %%
