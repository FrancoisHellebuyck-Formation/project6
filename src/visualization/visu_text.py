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
