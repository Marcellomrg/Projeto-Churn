# %%
# Projeto com objetivo de criar o melhor modelo para prever meu churn
import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
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

X = df_analise[features]
y = df_analise[target]
# %%
# Separando meus dados entre base treino e base teste
from sklearn import model_selection
x_train,x_test,y_train,y_test = model_selection.train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
# %%
df_train  = pd.DataFrame()
df_train = x_train.copy()
df_train["flagChurn"] = y_train
df_train
# %%
df_test = pd.DataFrame()
df_test = x_test.copy()
df_test["flagChurn"] = y_test
df_test

# %%
# Explore - Missing
df_train.isna().sum().sort_values(ascending=False)
# %%
summary = df_train.groupby(by=target).agg(["mean","median"]).T
summary["Diff_abs"] = summary[0] - summary[1]
summary["Diff_rel"] = summary[0]/summary[1]

# %%
summary.sort_values(by=["Diff_rel"],ascending=False)
# %%
from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(x_train,y_train)
plt.figure(dpi=400,figsize=(4,4))
tree.plot_tree(arvore,feature_names=x_train.columns
                    ,filled=True
                    ,class_names=[str(i) for i in arvore.classes_])

# %%
# Escolhendo as Features mais importates para meu modelo
features_importance = (pd.Series(arvore.feature_importances_, index=x_train.columns)
                            .sort_values(ascending=False)
                            .reset_index())
features_importance['acum'] =  features_importance[0].cumsum()
features_importance[features_importance['acum'] < 0.96]

best_features = (features_importance[features_importance['acum'] < 0.96]['index']
                 .tolist())
best_features
# %%
# MODIFY

from feature_engine import discretisation

tree_discretization = discretisation.DecisionTreeDiscretiser(
        variables=best_features
        ,regression=False
        ,bin_output= "bin_number"
        ,cv=3)

tree_discretization.fit(x_train,y_train)
# %%
x_train_transform = tree_discretization.transform(x_train)
x_train_transform

# %%
# Model

from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None,random_state=42)
reg.fit(x_train_transform,y_train)
# %%
