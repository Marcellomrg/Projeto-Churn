# %%
# Projeto com objetivo de criar o melhor modelo para prever meu churn
import pandas as pd
# %%
df = pd.read_csv("data/abt_churn.csv")
df.head()
# %%
# Sample
# Separando meus dados out of time
oot = df[df["dtRef"] == df["dtRef"].max()].copy()
oot
# %%
# Separando meus dados para base de treino/teste

df_analise = df[df["dtRef"] < df["dtRef"].max()].copy()
df_analise
# %%
# Separando minhas variaveis da minha target y
features = df_analise.columns.to_list()[2:-1]
target = "flagChurn"

target
X = df_analise[features]
y = df_analise[target]
# %%
# Separando meus dados entre base treino e base teste
from sklearn import model_selection
x_train,x_test,y_train,y_test = model_selection.train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
# %%
df_train  = x_train
df_train["flagChurn"] = y_train
df_train
# %%
df_test = x_test
df_test["flagChurn"] = y_test
df_test

# %%
# Fazendo Explore dos meus dados

df_train
# %%
