# %%
import pandas as pd
from pathlib import Path
import plotly.express as px

DATA_PATH = Path("task/math_data.csv")
data = pd.read_csv(DATA_PATH, sep=",")


# %%
# =============================================================================
# 1. EDA – liczba uczniów i szkół
# =============================================================================

n_szkol = data["id_szkoly"].nunique()
n_uczniow = data["id_ucznia"].nunique()

podzial_grupa = data.groupby("grupa").agg(
    n_uczniow=("id_ucznia", "nunique"),
    n_szkol=("id_szkoly", "nunique"),
).rename(index={0: "kontrolna", 1: "eksperymentalna"}).reset_index().replace({"grupa": {0: "kontrolna", 1: "eksperymentalna"}})

fig_uczniow = px.bar(
    podzial_grupa,
    x="grupa",
    y="n_uczniow",
    color="grupa",
    text="n_uczniow",
    title="Liczba uczniów biorących udział w badaniu",
    labels={"grupa": "Grupa", "n_uczniow": "Liczba uczniów"}
)
fig_uczniow.update_xaxes(
    tickvals=["kontrolna", "eksperymentalna"],
    ticktext=["kontrolna", "eksperymentalna"]
)
fig_uczniow.show()

fig_szkol = px.bar(
    podzial_grupa,
    x="grupa",
    y="n_szkol",
    color="grupa",
    text="n_szkol",
    title="Liczba szkół biorących udział w badaniu",
    labels={"grupa": "Grupa", "n_szkol": "Liczba szkół"}
)
fig_szkol.update_xaxes(
    tickvals=["kontrolna", "eksperymentalna"],
    ticktext=["kontrolna", "eksperymentalna"]
)
fig_szkol.show()
# %%
# =============================================================================
# 2. EDA – statystyki opisowe zadań (pierwszy vs drugi pomiar)
# =============================================================================

mat_columns = [c for c in data.columns if c.startswith("mat_")]

pretest = data[data["pomiar"] == 1]
posttest = data[data["pomiar"] == 2]

count_pretest = pretest[mat_columns].notna().sum()
count_posttest = posttest[mat_columns].notna().sum()
zadania_pretest = count_pretest[count_pretest > count_posttest].index.tolist()
zadania_posttest = count_posttest[count_posttest > count_pretest].index.tolist()

def sort_mat(name):
    n = name.replace("mat_", "")
    return int(n) if n.isdigit() else 0

print("Zadania w pierwszym pomiarze (pretest):", sorted(zadania_pretest, key=sort_mat))
print("Zadania w drugim pomiarze (posttest):  ", sorted(zadania_posttest, key=sort_mat))

# Statystyki opisowe – pierwszy pomiar (pretest)
pretest_stats = pretest[zadania_pretest].apply(pd.to_numeric, errors="coerce")
stats_pretest = pretest_stats.describe().T[["count", "mean", "std"]]
stats_pretest

# Statystyki opisowe – drugi pomiar (posttest)
posttest_stats = posttest[zadania_posttest].apply(pd.to_numeric, errors="coerce")
stats_posttest = posttest_stats.describe().T[["count", "mean", "std"]]
stats_posttest

# %%
