# %%
# Projeto com objetivo de criar o melhor modelo para prever meu churn
import pandas as pd
import mlflow
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from feature_engine import discretisation,encoding
from sklearn import pipeline
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name="Churn_experiment")
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
arvore = tree.DecisionTreeClassifier(random_state=42)
# %%
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
tree_discretization = discretisation.DecisionTreeDiscretiser(
        variables=best_features
        ,regression=False
        ,bin_output= "bin_number"
        ,cv=3)

hot_encoding = encoding.OneHotEncoder(variables=best_features,
                                      ignore_format=True)


# %%
# Model

#model = linear_model.LogisticRegression(penalty=None,random_state=42,max_iter=1000000)
#model = naive_bayes.BernoulliNB()
model = ensemble.RandomForestClassifier(random_state=42,
                                            n_jobs=2,
                                            )
#model = ensemble.AdaBoostClassifier(random_state=42,
#                                    n_estimators=500,
#                                    learning_rate=0.1)

param = {
        "min_samples_leaf":[20,25,50,10,15],
        "n_estimators":[500,100,200,1000],
        "criterion":["gini","entropy","log_loss"]
}

grid = model_selection.GridSearchCV(model,
                                    param_grid=param
                                    ,cv=3
                                    ,scoring="roc_auc"
                                    ,verbose=4)

model_pipeline = pipeline.Pipeline(
                steps=[
                    ('Discretizar',tree_discretization),
                    ("Onehot",hot_encoding),
                    ("Grid",grid)
                ])


with mlflow.start_run(run_name=model.__str__()):
        
        mlflow.sklearn.autolog()
        model_pipeline.fit(x_train[best_features],y_train)


        # Metricas de ajuste para treino

        y_train_predict = model_pipeline.predict(x_train[best_features])
        y_train_proba = model_pipeline.predict_proba(x_train[best_features])[:,1]

        acc_train = metrics.accuracy_score(y_train,y_train_predict)
        auc_train = metrics.roc_auc_score(y_train,y_train_proba)
        roc_train = metrics.roc_curve(y_train,y_train_proba)
        print("Acuracia Treino:",acc_train)
        print("AUC TREINO:",auc_train)

        # Metricas de ajuste para teste

        y_test_predict = model_pipeline.predict(x_test[best_features])
        y_test_proba = model_pipeline.predict_proba(x_test[best_features])[:,1]

        acc_test = metrics.accuracy_score(y_test,y_test_predict)
        auc_test = metrics.roc_auc_score(y_test,y_test_proba)
        roc_test = metrics.roc_curve(y_test,y_test_proba)
        print("Acuracia Teste:",acc_test)
        print("AUC Teste:",auc_test)

        # Metricas de ajuste para minha OOT

        y_oot_predict = model_pipeline.predict(oot[best_features])
        y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]

        acc_oot = metrics.accuracy_score(oot[target],y_oot_predict)
        auc_oot = metrics.roc_auc_score(oot[target],y_oot_proba)
        roc_oot = metrics.roc_curve(oot[target],y_oot_proba)
        print("Acuracia Oot:",acc_oot)
        print("AUC Oot:",auc_oot)

        mlflow.log_metrics({
                "acc_train":acc_train,
                "auc_train":auc_train,
                "acc_test":acc_test,
                "auc_test":auc_test,
                "acc_oot":acc_oot,
                "auc_oot":auc_oot,

        }
        )

# %%
plt.figure(dpi=400)
plt.plot(roc_train[0],roc_train[1])
plt.plot(roc_test[0],roc_test[1])
plt.plot(roc_oot[0],roc_oot[1])
plt.plot([0,1],[0,1],"--",color="black")
plt.xlabel("1 - Especificidade")
plt.ylabel("Sensibilidade")
plt.grid(True)
plt.title("Curva ROC")
plt.legend([
            f"Treino:{100*auc_train:.2f}",
            f"Teste:{100*auc_test:.2f}",
            f"Oot:{100*auc_oot:.2f}",
]
)


# %%

