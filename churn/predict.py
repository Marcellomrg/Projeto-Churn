# %%
import pandas as pd
# %%
model_pkl = pd.read_pickle("model.pkl")
features = model_pkl["features"]
model = model_pkl["model"]

# %%
df = pd.read_csv("data/abt_churn.csv")
df.head()
# %%
df_oot = df[df["dtRef"] == df["dtRef"].max()].drop("flagChurn",axis = 1)
df_oot
# %%
amostra = df_oot.sample(3)
amostra
# %%
predict = model.predict_proba(amostra[features])[:,1]
predict
# %%
amostra['proba'] = predict
amostra
# %%
