import pandas as pd


def print_title(title, width=80):
    """
    Affiche un titre formaté pour les textes
    """
    w = int((width - len(title) - 6) / 2)
    print("\n┌" + "-" * (w) + "* " + title + " *" + "-" * w + "┐")


def print_end(width=80):
    """
    Affiche une fin pour les titres
    """
    w = width - 2
    print("└" + "-" * (w) + "┘")


def print_col(texte):
    """
    Affiche un texte colonne
    """
    print("├─" + texte)


def quick_df_info(df, titre="Information"):
    print_title(titre)
    print_col(f"------- Shape: {df.shape} - Colonnes:")
    for i, col in enumerate(df.columns):
        dtype_str = str(df[col].dtype)
        print_col(f"{col:<25} {dtype_str:<10}")
    print_end()


def print_results(model_scores):
    """
    Affiche les résultats des modèles dans un tableau formaté

    Args:
        model_scores: liste de dictionnaires avec les scores des modèles
    """
    # Tableau des performances des modèles
    scores_df = pd.DataFrame(model_scores)
    scores_df.set_index("Model", inplace=True)
    scores_df.sort_values(by=["F2-score", "Recall"], ascending=False, inplace=True)
    print_title("TABLEAU DES PERFORMANCES DES MODÈLES PAR F2-score")
    print(scores_df.round(3).to_string(float_format="{:.3f}".format))
    print_end()
