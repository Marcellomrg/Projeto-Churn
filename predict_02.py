# %%
import pandas as pd
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
# %%
# Importando meu modelo
models = mlflow.search_registered_models(filter_string="name = 'model_churn'")
lastest_version = max([i.version for i in models[0].latest_versions])

model = mlflow.sklearn.load_model(f"models:/model_churn/{lastest_version}")
features = model.feature_names_in_

# %%
# Meus dados
df = pd.read_csv("data/abt_churn.csv")
df.head()
# %%
df_oot = df[df["dtRef"] == df["dtRef"].max()].drop("flagChurn",axis = 1)
df_oot
# %%
amostra = df_oot.sample(3)
amostra
# %%
# Meu predict do churn
predict = model.predict_proba(amostra[features])[:,1]
predict
# %%
amostra['proba'] = predict
amostra
# %%

