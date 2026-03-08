# %%
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer
import pingouin as pg
from girth import twopl_mml, threepl_mml

import warnings
warnings.filterwarnings('ignore')

# %%
# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = Path("task/math_data.csv")
LOWER_MAT_THRESHOLD, UPPER_MAT_THRESHOLD = 0.05, 0.95

# %%
# =============================================================================
# Podział danych
# =============================================================================
data = pd.read_csv(DATA_PATH, sep=",")

pretest = data[data["pomiar"] == 1]
posttest = data[data["pomiar"] == 2]

non_mat_cols = [c for c in data.columns if not c.startswith("mat_")]
mat_cols = [c for c in data.columns if c.startswith("mat_")]

def _mat_sort_key(name):
    """Sort key: mat_1, mat_2, ..., mat_9, mat_10 (numeric order)."""
    n = name.replace("mat_", "")
    return int(n) if n.isdigit() else 0

# %%
# =============================================================================
# Step 1: Weryfikacja danych
# =============================================================================

# %% Weryfikacja braków danych: Null tylko w kolumnach mat_ i oznacza brak zadania w danym pomiarze
nulls_non_mat = data[non_mat_cols].isna().any(axis=1).sum()
nulls_mat = data[mat_cols].isna().sum()

print(f"Liczba brakujących wartości w kolumnach innych niż mat_*: {nulls_non_mat}")

# %% Dla każdego pomiaru: które zadania mają odpowiedzi (nie-Null) vs które są puste (zadanie nie w pomiarze)
zadania_w_pretest = pretest[mat_cols].notna().any(axis=0)
zadania_w_posttest = posttest[mat_cols].notna().any(axis=0)
print("\nZadania z danymi w pretest (pomiar=1):", zadania_w_pretest[zadania_w_pretest].index.tolist())
print("Zadania z danymi w posttest (pomiar=2):", zadania_w_posttest[zadania_w_posttest].index.tolist())

# %% Weryfikacja randomizacji: każda szkoła ma przypisaną tylko jedną grupę
grupa_per_szkola = data.groupby("id_szkoly")["grupa"].nunique()
szkoly_wiecej_grup = grupa_per_szkola[grupa_per_szkola > 1]

print("\nWeryfikacja randomizacji (jedna grupa na szkołę)")
print(f"Liczba szkół: {data['id_szkoly'].nunique()}")
print(f"Szkół z więcej niż jedną grupą: {len(szkoly_wiecej_grup)}")

# %% Uczniowie w preteście, ale nie w postteście
id_pretest = set(data[data["pomiar"] == 1]["id_ucznia"])
id_posttest = set(data[data["pomiar"] == 2]["id_ucznia"])
tylko_pretest = id_pretest - id_posttest
tylko_posttest = id_posttest - id_pretest

print("\nUczniowie: pretest vs posttest")
print(f"Uczniów tylko w preteście (brak w postteście): {len(tylko_pretest)}")
print(f"Uczniów tylko w postteście (brak w preteście): {len(tylko_posttest)}")

# %%
# =============================================================================
# Step 2: Eskploracja danych
# =============================================================================

# %% Liczba szkół i uczniów w podziale na grupę
n_szkol = data["id_szkoly"].nunique()
n_uczniow = data["id_ucznia"].nunique()

# Liczba szkół i uczniów w podziale na grupę
macierz_grupa = (
    data.groupby("grupa")
    .agg(n_szkol=("id_szkoly", "nunique"), n_uczniów=("id_ucznia", "nunique"))
    .rename(index={0: "kontrolna", 1: "eksperymentalna"})
)
macierz_grupa.index.name = "grupa"
macierz_grupa.loc["TOTAL"] = [n_szkol, n_uczniow]

print("\nMacierz: szkoły i uczniowie w podziale na grupę")
print(macierz_grupa)

# %% Trudność zadań (średni wynik) w każdym pomiarze + zadania ekstremalne

zadania_pretest = zadania_w_pretest[zadania_w_pretest].index.tolist()
zadania_posttest = zadania_w_posttest[zadania_w_posttest].index.tolist()

pretest_num = pretest[zadania_pretest].apply(pd.to_numeric, errors="coerce")
posttest_num = posttest[zadania_posttest].apply(pd.to_numeric, errors="coerce")

trudnosc_pretest = pretest_num.mean().reindex(
    sorted(pretest_num.columns, key=_mat_sort_key)
)
trudnosc_posttest = posttest_num.mean().reindex(
    sorted(posttest_num.columns, key=_mat_sort_key)
)

# %% Zadania ekstremalnie łatwe (śr. blisko 1) lub trudne (śr. blisko 0)

ekstremalne_pretest = trudnosc_pretest[(trudnosc_pretest <= LOWER_MAT_THRESHOLD) | (trudnosc_pretest >= UPPER_MAT_THRESHOLD)]
ekstremalne_posttest = trudnosc_posttest[(trudnosc_posttest <= LOWER_MAT_THRESHOLD) | (trudnosc_posttest >= UPPER_MAT_THRESHOLD)]

fig_trudnosc = px.bar(
    x=trudnosc_pretest.index,
    y=trudnosc_pretest.values,
    title="Trudność zadań – pretest (średni wynik)",
    labels={"x": "Zadanie", "y": "Średni wynik"},
)
fig_trudnosc.update_yaxes(range=[0, 1])
fig_trudnosc.add_hline(y=UPPER_MAT_THRESHOLD, line_dash="dash", line_color="red")
fig_trudnosc.add_hline(y=LOWER_MAT_THRESHOLD, line_dash="dash", line_color="red")
fig_trudnosc.show()

fig_trudnosc_post = px.bar(
    x=trudnosc_posttest.index,
    y=trudnosc_posttest.values,
    title="Trudność zadań – posttest (średni wynik)",
    labels={"x": "Zadanie", "y": "Średni wynik"},
)
fig_trudnosc_post.update_yaxes(range=[0, 1])
fig_trudnosc_post.add_hline(y=UPPER_MAT_THRESHOLD, line_dash="dash", line_color="red")
fig_trudnosc_post.add_hline(y=LOWER_MAT_THRESHOLD, line_dash="dash", line_color="red")
fig_trudnosc_post.show()

# %%
# =============================================================================
# Step 3: Jednowymiarowość – analiza czynnikowa + scree plot
# =============================================================================

# %% Sprawdzenie korelacji i proporcji wariancji która może być wspólna
X_pretest = (
    pretest[zadania_pretest]
    .apply(pd.to_numeric, errors="coerce")
    .dropna(how="any")
)
chi_square, p_value = calculate_bartlett_sphericity(X_pretest)
kmo_all, kmo_model = calculate_kmo(X_pretest)
print(f"Pretest - Bartlett p-value: {p_value:.4f}, KMO: {kmo_model:.4f}")

X_posttest = (
    posttest[zadania_posttest]
    .apply(pd.to_numeric, errors="coerce")
    .dropna(how="any")
)
chi_square, p_value = calculate_bartlett_sphericity(X_posttest)
kmo_all, kmo_model = calculate_kmo(X_posttest)
print(f"Posttest - Bartlett p-value: {p_value:.4f}, KMO: {kmo_model:.4f}")

# %% scree plot
def create_scree_plot(data, title):
    fa = FactorAnalyzer(rotation=None)
    fa.fit(data)
    ev, _ = fa.get_eigenvalues()
    
    factors = list[int](range(1, len(ev) + 1))
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=factors, 
        y=ev, 
        mode='lines+markers',
        name='Wartości własne',
        marker=dict[str, int | str](size=10, color='royalblue'),
        line=dict[str, int](width=3)
    ))

    fig.add_shape(
        type="line",
        x0=1, y0=1, x1=max(factors), y1=1,
        line=dict[str, str | int](color="Red", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=max(factors)*0.9, y=1.2,
        text="Kryterium Kaisera (EV=1)",
        showarrow=False,
        font=dict[str, str](color="red")
    )

    fig.update_layout(
        title=title,
        xaxis_title="Numer czynnika",
        yaxis_title="Wartość własna (Eigenvalue)",
        template="plotly_white",
        xaxis=dict[str, str | int](tickmode='linear', tick0=1, dtick=1),
        hovermode="x unified"
    )

    fig.show()

# %%
create_scree_plot(X_pretest, title="Scree Plot - Pretest (Pomiar 1)")

# %%
create_scree_plot(X_posttest, title="Scree Plot - Posttest (Pomiar 2)")

# %%
# =============================================================================
# Step 4: CTT i IRT
# =============================================================================

# %% CTT jako baza - alfa cronbacha
alpha_pre = pg.cronbach_alpha(data=X_pretest)
print(f"Alfa Cronbacha (Pretest): {alpha_pre[0]:.3f} (CI: {alpha_pre[1]})")

alpha_post = pg.cronbach_alpha(data=X_posttest)
print(f"Alfa Cronbacha (Posttest): {alpha_post[0]:.3f}")

# %% moc dyskryminacyjna CTT
total_scores_pre = X_pretest.sum(axis=1)
item_discrimination_pre = X_pretest.apply(lambda col: col.corr(total_scores_pre))

print("Moc dyskryminacyjna (Top 5 zadań - Pretest):")
print(item_discrimination_pre.sort_values(ascending=False).head())

# %% IRT 2PL
matrix_pretest = X_pretest.to_numpy(dtype=int).T
matrix_posttest = X_posttest.to_numpy(dtype=int).T

estimates_pretest_2pl = twopl_mml(matrix_pretest)
estimates_posttest_2pl = twopl_mml(matrix_posttest)

items_pretest_2pl = pd.DataFrame({
    'Trudność (b)': estimates_pretest_2pl['Difficulty'],
    'Dyskryminacja (a)': estimates_pretest_2pl['Discrimination'],
}, index=X_pretest.columns).reindex(sorted(X_pretest.columns, key=_mat_sort_key))

items_posttest_2pl = pd.DataFrame({
    'Trudność (b)': estimates_posttest_2pl['Difficulty'],
    'Dyskryminacja (a)': estimates_posttest_2pl['Discrimination'],
}, index=X_posttest.columns).reindex(sorted(X_posttest.columns, key=_mat_sort_key))

print("Parametry IRT - 2PL (pretest):")
print(items_pretest_2pl)
print("\nParametry IRT - 2PL (posttest):")
print(items_posttest_2pl)

# %% Wykres ICC: uniwersalna funkcja 2PL i 3PL
def create_icc_curve(theta, a, b, c=None):
    """Uniwersalna krzywa ICC: 2PL (c=None) lub 3PL (c=zgadywanie)."""
    p = 1 / (1 + np.exp(-a * (theta - b)))
    if c is not None:
        p = c + (1 - c) * p
    return p


def plot_icc_curves(theta, curves, title, y_floor=0):
    """Rysuje krzywe ICC. curves: lista dictów z kluczami a, b, label oraz opcjonalnie c (3PL)."""
    fig = go.Figure()
    for curve in curves:
        c = curve.get("c")
        y = create_icc_curve(theta, curve["a"], curve["b"], c=c)
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=y,
                mode="lines",
                name=curve["label"],
                line=dict[str, int](width=2),
            )
        )
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    if y_floor > 0:
        fig.add_hline(y=y_floor, line_dash="dot", line_color="gray", annotation_text="c (zgadywanie)")
    fig.update_layout(
        title=title,
        xaxis_title="θ (umiejętność)",
        yaxis_title="P(poprawna odpowiedź)",
        yaxis=dict[str, list[int]](range=[y_floor, 1]),
        legend=dict[str, str | float](yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.show()


theta = np.linspace(-3, 3, 301)

# 2PL: Trudność (b)
plot_icc_curves(
    theta,
    [
        {"a": 1.5, "b": -1, "label": "Łatwe (b = -1)"},
        {"a": 1.5, "b": 0, "label": "Średnie (b = 0)"},
        {"a": 1.5, "b": 1.5, "label": "Trudne (b = 1.5)"},
    ],
    title="2PL: Położenie krzywej (Trudność b)",
)

# 2PL: Dyskryminacja (a)
plot_icc_curves(
    theta,
    [
        {"a": 0.5, "b": 0, "label": "Słaba dyskryminacja (a = 0.5)"},
        {"a": 1.5, "b": 0, "label": "Średnia (a = 1.5)"},
        {"a": 2.5, "b": 0, "label": "Silna dyskryminacja (a = 2.5)"},
    ],
    title="2PL: Nachylenie krzywej (Dyskryminacja a)",
)

# %% IRT 3PL
estimates_pretest_3pl = threepl_mml(matrix_pretest)
estimates_posttest_3pl = threepl_mml(matrix_posttest)

items_pretest_3pl = pd.DataFrame({
    'Trudność (b)': estimates_pretest_3pl['Difficulty'],
    'Dyskryminacja (a)': estimates_pretest_3pl['Discrimination'],
    'Prawdopodobieństwo (c)': estimates_pretest_3pl['Guessing'],
}, index=X_pretest.columns).reindex(sorted(X_pretest.columns, key=_mat_sort_key))

items_posttest_3pl = pd.DataFrame({
    'Trudność (b)': estimates_posttest_3pl['Difficulty'],
    'Dyskryminacja (a)': estimates_posttest_3pl['Discrimination'],
    'Prawdopodobieństwo (c)': estimates_posttest_3pl['Guessing'],
}, index=X_posttest.columns).reindex(sorted(X_posttest.columns, key=_mat_sort_key))

print("Parametry IRT - 3PL (pretest):")
print(items_pretest_3pl)
print("\nParametry IRT - 3PL (posttest):")
print(items_posttest_3pl)

# prawdopodobieństwo zgadniecia zadania (c) jest wszędzie bliske 0 co znaczy że zgadywanie nie ma dużego wpływu. Dlatego wybieramy 2PL.

# %% Wykres ICC 3PL:
plot_icc_curves(
    theta,
    [
        {"a": 1.5, "b": 0, "label": "2PL (c = 0)"},
        {"a": 1.5, "b": 0, "c": 0.2, "label": "3PL (c = 0.2)"},
        {"a": 1.5, "b": 0, "c": 0.25, "label": "3PL (c = 0.25)"},
    ],
    title="3PL: Punkt startowy (zgadywanie c) – szansa na traf przy bardzo niskim θ",
    y_floor=0.25,
)

# %% ICC dla wszystkich zadań pretestu (parametry z 2PL)
curves_pretest_2pl = [
    {
        "a": row["Dyskryminacja (a)"],
        "b": row["Trudność (b)"],
        "label": idx,
    }
    for idx, row in items_pretest_2pl.iterrows()
]
plot_icc_curves(
    theta,
    curves_pretest_2pl,
    title="ICC dla wszystkich zadań pretestu (2PL)",
)

# %% ICC dla wszystkich zadań pretestu (parametry z 3PL)
curves_pretest_3pl = [
    {
        "a": row["Dyskryminacja (a)"],
        "b": row["Trudność (b)"],
        "c": row["Prawdopodobieństwo (c)"],
        "label": idx,
    }
    for idx, row in items_pretest_3pl.iterrows()
]
plot_icc_curves(
    theta,
    curves_pretest_3pl,
    title="ICC dla wszystkich zadań pretestu (3PL)",
    y_floor=0,
)

# %%
