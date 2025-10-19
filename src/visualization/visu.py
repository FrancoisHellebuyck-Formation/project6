import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
from typing import Union, Optional

warnings.filterwarnings("ignore")


def create_donut_chart(
    data: Union[pd.DataFrame, pd.Series, dict, list],
    labels: Optional[Union[str, list]] = None,
    values: Optional[str] = None,
    title: str = "Répartition en pourcentage",
    hole_size: float = 0.6,
    figsize: tuple = (8, 8),
    palette: str = "husl",
    show_percentages: bool = True,
    center_text: Optional[str] = None,
    explode: Optional[list] = None,
    startangle: int = 90,
    fig=None,
    ax=None,
) -> tuple:
    """
    Crée un graphique en beignet avec calcul automatique des pourcentages.
    Compte automatiquement les occurrences des valeurs string.

    Parameters:
    -----------
    data : DataFrame, Series, dict ou list
        Les données à visualiser
    labels : str ou list, optional
        Non utilisé pour DataFrame (utilise les valeurs de la colonne values)
    values : str, optional
        Nom de la colonne à analyser (pour DataFrame)
    title : str
        Titre du graphique
    hole_size : float
        Taille du trou central (0.0 à 1.0)
    figsize : tuple
        Taille de la figure (largeur, hauteur) - ignoré si fig et ax sont fournis
    palette : str
        Palette de couleurs seaborn
    show_percentages : bool
        Afficher les pourcentages sur le graphique
    center_text : str, optional
        Texte à afficher au centre
    explode : list, optional
        Liste pour séparer certaines parts
    startangle : int
        Angle de départ du premier segment
    fig : matplotlib.figure.Figure, optional
        Figure matplotlib existante
    ax : matplotlib.axes.Axes, optional
        Axe matplotlib existant

    Returns:
    --------
    tuple : (fig, ax)
        La figure et l'axe utilisés

    Examples:
    ---------
    # Utilisation simple (crée fig et ax automatiquement)
    fig, ax = create_donut_chart(data, title="Mon graphique")

    # Avec fig et ax existants
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    create_donut_chart(data1, fig=fig, ax=axes[0,0], title="Graphique 1")
    create_donut_chart(data2, fig=fig, ax=axes[0,1], title="Graphique 2")

    # Avec un DataFrame (compte les occurrences de chaque valeur)
    df = pd.DataFrame({'statut': ['Actif', 'Inactif', 'Actif', 'Suspendu', 'Actif']})
    create_donut_chart(df, values='statut')

    # Avec une liste de strings (compte automatiquement)
    statuts = ['Actif', 'Inactif', 'Actif', 'Suspendu', 'Actif', 'Actif']
    create_donut_chart(statuts, title="Distribution des statuts")
    """

    # Traitement des données selon le type d'entrée
    if isinstance(data, pd.DataFrame):
        if values is None:
            raise ValueError(
                "Pour un DataFrame, spécifiez la colonne 'values' à analyser"
            )

        # Compter les occurrences des valeurs dans la colonne spécifiée
        value_counts = data[values].value_counts()
        plot_labels = value_counts.index.tolist()
        plot_values = value_counts.values.tolist()

    elif isinstance(data, pd.Series):
        # Compter les occurrences dans la Series
        value_counts = data.value_counts()
        plot_labels = value_counts.index.tolist()
        plot_values = value_counts.values.tolist()

    elif isinstance(data, dict):
        # Si c'est un dict, on suppose que les valeurs sont des comptages
        plot_labels = list(data.keys())
        plot_values = list(data.values())

    elif isinstance(data, list):
        # Compter les occurrences dans la liste
        if all(isinstance(x, str) for x in data):
            # Si c'est une liste de strings, compter les occurrences
            value_counts = pd.Series(data).value_counts()
            plot_labels = value_counts.index.tolist()
            plot_values = value_counts.values.tolist()
        else:
            # Si c'est une liste de nombres, traiter comme avant
            plot_values = data
            if labels is None:
                plot_labels = [f"Catégorie {i+1}" for i in range(len(data))]
            elif isinstance(labels, list):
                if len(labels) != len(data):
                    raise ValueError(
                        "Le nombre de labels doit correspondre au nombre de valeurs"
                    )
                plot_labels = labels
            else:
                raise ValueError(
                    "Labels doit être une liste pour des données de type list"
                )
    else:
        raise ValueError("Type de données non supporté")

    # Calcul des pourcentages
    total = sum(plot_values)

    # Configuration du style
    sns.set_style("white")
    colors = sns.color_palette(palette, len(plot_values))

    # Création ou utilisation de la figure et de l'axe
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True  # Afficher le plot à la fin
    else:
        show_plot = False  # Ne pas afficher automatiquement

    # Formatage des pourcentages pour l'affichage
    autopct_format = "%1.1f%%" if show_percentages else ""

    # Création du pie chart
    wedges, texts, autotexts = ax.pie(
        plot_values,
        labels=plot_labels,
        colors=colors,
        autopct=autopct_format,
        startangle=startangle,
        pctdistance=0.85,
        explode=explode,
    )

    # Création du trou central
    centre_circle = plt.Circle((0, 0), hole_size, fc="white")
    ax.add_artist(centre_circle)

    # Stylisation du texte des pourcentages
    if show_percentages:
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

    # Texte au centre si spécifié
    if center_text:
        plt.text(
            0,
            0,
            center_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            fontweight="bold",
        )
    elif center_text is None and total:
        # Affichage du total par défaut
        plt.text(
            0,
            0,
            f"Total\n{total}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight="bold",
        )

    # Configuration finale
    ax.axis("equal")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Affichage seulement si fig et ax n'étaient pas fournis
    if show_plot:
        plt.tight_layout()
        plt.show()

    return fig, ax


# Configuration du style des graphiques
plt.style.use("default")
sns.set_palette("husl")


class InteractiveScatterPlot:
    def __init__(self, data=None, vq=None):
        """
        Initialise le scatter plot interactif

        Parameters:
        data (DataFrame): DataFrame pandas avec les données. Si None, utilise des données d'exemple.
        vq (list, optional): Liste de colonnes numériques à utiliser pour le scatter plot.
        """
        if data is None:
            # Création de données d'exemple
            np.random.seed(42)
            n_samples = 200
            self.data = pd.DataFrame(
                {
                    "Taille (cm)": np.random.normal(170, 10, n_samples),
                    "Poids (kg)": np.random.normal(70, 15, n_samples),
                    "Âge": np.random.randint(18, 80, n_samples),
                    "Salaire (k€)": np.random.exponential(35, n_samples) + 20,
                    "Expérience (années)": np.random.poisson(8, n_samples),
                    "Score satisfaction": np.random.beta(2, 2, n_samples) * 10,
                }
            )
            # Ajout de corrélations réalistes
            self.data["Poids (kg)"] = (
                self.data["Taille (cm)"] * 0.8 + np.random.normal(0, 5, n_samples) - 66
            )
            self.data["Salaire (k€)"] = self.data[
                "Expérience (années)"
            ] * 2.5 + np.random.normal(30, 10, n_samples)
        else:
            self.data = data.copy()

        # Sélection des colonnes numériques uniquement
        if vq is not None:
            numeric_columns = vq
        else:
            numeric_columns = self.data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        if len(numeric_columns) < 2:
            raise ValueError(
                "Le DataFrame doit contenir au moins 2 colonnes numériques"
            )

        self.numeric_columns = numeric_columns

        # Création des widgets
        self.create_widgets()

        # Zone d'affichage du graphique
        self.output = widgets.Output()

        # Interface utilisateur
        self.setup_ui()

    def create_widgets(self):
        """Crée les widgets de l'interface"""
        # Dropdown pour la variable X
        self.x_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=self.numeric_columns[0],
            description="Variable X:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Dropdown pour la variable Y
        self.y_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=(
                self.numeric_columns[1]
                if len(self.numeric_columns) > 1
                else self.numeric_columns[0]
            ),
            description="Variable Y:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Slider pour la taille des points
        self.size_slider = widgets.IntSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description="Taille points:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Slider pour la transparence
        self.alpha_slider = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description="Transparence:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Checkbox pour afficher la ligne de régression
        self.regression_checkbox = widgets.Checkbox(
            value=False,
            description="Ligne de régression",
            style={"description_width": "initial"},
        )

        # Bouton pour actualiser
        self.update_button = widgets.Button(
            description="Actualiser le graphique",
            button_style="primary",
            layout=widgets.Layout(width="200px"),
        )

        # Connexion des événements
        self.x_dropdown.observe(self.on_change, names="value")
        self.y_dropdown.observe(self.on_change, names="value")
        self.size_slider.observe(self.on_change, names="value")
        self.alpha_slider.observe(self.on_change, names="value")
        self.regression_checkbox.observe(self.on_change, names="value")
        self.update_button.on_click(self.on_button_click)

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Groupe des contrôles
        controls_box = widgets.VBox(
            [
                widgets.HTML("<h3>🎯 Configuration du Scatter Plot</h3>"),
                widgets.HBox([self.x_dropdown, self.y_dropdown]),
                widgets.HBox([self.size_slider, self.alpha_slider]),
                widgets.HBox([self.regression_checkbox, self.update_button]),
                widgets.HTML("<hr>"),
            ]
        )

        # Interface complète
        self.ui = widgets.VBox([controls_box, self.output])

        # Affichage initial
        self.update_plot()

    def on_change(self, change):
        """Callback appelé quand un widget change"""
        self.update_plot()

    def on_button_click(self, button):
        """Callback pour le bouton d'actualisation"""
        self.update_plot()

    def update_plot(self):
        """Met à jour le graphique"""
        with self.output:
            clear_output(wait=True)

            # Récupération des valeurs
            x_var = self.x_dropdown.value
            y_var = self.y_dropdown.value
            point_size = self.size_slider.value
            alpha = self.alpha_slider.value
            show_regression = self.regression_checkbox.value

            # Vérification que les variables sont différentes
            if x_var == y_var:
                print("⚠️ Attention: Les variables X et Y sont identiques!")
                return

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 7))

            # Scatter plot
            ax.scatter(
                self.data[x_var],
                self.data[y_var],
                s=point_size,
                alpha=alpha,
                # c=range(len(self.data)),
                cmap="viridis",
                edgecolors="white",
                linewidth=0.5,
            )

            # Ligne de régression si demandée
            if show_regression:
                z = np.polyfit(self.data[x_var], self.data[y_var], 1)
                p = np.poly1d(z)
                ax.plot(
                    self.data[x_var], p(self.data[x_var]), "r--", alpha=0.8, linewidth=2
                )

                # Calcul du coefficient de corrélation
                corr = self.data[x_var].corr(self.data[y_var])
                ax.text(
                    0.05,
                    0.95,
                    f"r = {corr:.3f}",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=12,
                    fontweight="bold",
                )

            # Personnalisation du graphique
            ax.set_xlabel(x_var, fontsize=12, fontweight="bold")
            ax.set_ylabel(y_var, fontsize=12, fontweight="bold")
            ax.set_title(
                f"Scatter Plot: {x_var} vs {y_var}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.grid(True, alpha=0.3)

            # Ajout d'une colorbar
            # cbar = plt.colorbar(scatter, ax=ax)
            # cbar.set_label('Index des points', rotation=270, labelpad=15)

            # Statistiques descriptives
            stats_text = f"""📊 Statistiques:
            • Points: {len(self.data)}
            • {x_var}: μ={self.data[x_var].mean():.2f}, σ={self.data[x_var].std():.2f}
            • {y_var}: μ={self.data[y_var].mean():.2f}, σ={self.data[y_var].std():.2f}"""

            ax.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                fontsize=9,
                verticalalignment="bottom",
            )

            plt.tight_layout()
            plt.show()

    def display(self):
        """Affiche l'interface interactive"""
        display(self.ui)


# Fonction principale pour utiliser facilement le widget
def create_interactive_scatterplot(data=None, vq=None):
    """
    Crée et affiche un scatter plot interactif

    Parameters:
    data (DataFrame, optional): DataFrame pandas avec vos données.
                               Si None, utilise des données d'exemple.
    vq (list, optional): Liste de colonnes numériques à utiliser pour le scatter plot.

    Returns:
    InteractiveScatterPlot: Instance de la classe pour une utilisation avancée
    """
    plot_widget = InteractiveScatterPlot(data, vq)
    plot_widget.display()
    return plot_widget


# Exemple d'utilisation simple
if __name__ == "__main__":
    print("🚀 Lancement du scatter plot interactif...")
    print("💡 Utilisez: create_interactive_scatterplot() pour commencer")

    # Utilisation avec des données d'exemple
    widget = create_interactive_scatterplot()

# ═══════════════════════════════════════════════════════════════
# 📋 GUIDE D'UTILISATION:
# ═══════════════════════════════════════════════════════════════
"""
UTILISATION SIMPLE:
─────────────────────
widget = create_interactive_scatterplot()

UTILISATION AVEC VOS DONNÉES:
───────────────────────────────
import pandas as pd
mes_donnees = pd.read_csv('mon_fichier.csv')
widget = create_interactive_scatterplot(mes_donnees)

FONCTIONNALITÉS:
──────────────────
✅ Sélection dynamique des variables X et Y
✅ Contrôle de la taille des points
✅ Ajustement de la transparence
✅ Ligne de régression avec coefficient de corrélation
✅ Statistiques descriptives automatiques
✅ Colormap pour différencier les points
✅ Interface responsive et intuitive

REQUIREMENTS:
───────────────
pip install pandas numpy matplotlib ipywidgets seaborn
"""

# Configuration du style
plt.style.use("default")
sns.set_style("whitegrid")


class InteractiveRawBarPlot:
    def __init__(self, data=None):
        """
        Initialise le graphique en barres côte à côte pour distribution brute

        Parameters:
        data (DataFrame): DataFrame pandas avec les données. Si None, utilise des données d'exemple.
        """
        if data is None:
            # Création de données d'exemple réalistes
            np.random.seed(42)
            n_samples = 1000

            # Simulation d'employés
            ages = np.random.randint(22, 65, n_samples)
            salaires = ages * 1000 + np.random.normal(10000, 8000, n_samples)
            salaires = np.clip(salaires, 25000, 120000)

            # Variables catégorielles
            departements = np.random.choice(
                ["IT", "Marketing", "RH", "Finance", "Ventes"], n_samples
            )
            niveaux = np.random.choice(
                ["Junior", "Senior", "Manager"], n_samples, p=[0.5, 0.3, 0.2]
            )
            genres = np.random.choice(["Homme", "Femme"], n_samples)

            # Simulation de départ d'entreprise (corrélé avec l'âge)
            prob_depart = np.where(ages < 30, 0.3, np.where(ages > 55, 0.4, 0.15))
            a_quitte = np.random.binomial(1, prob_depart, n_samples)

            # Simulation de satisfaction (corrélée avec salaire)
            satisfaction_prob = (salaires - 25000) / (120000 - 25000)
            satisfaction = np.random.choice(
                ["Satisfait", "Insatisfait"],
                n_samples,
                p=[
                    (satisfaction_prob + 0.3).clip(0, 1),
                    (0.7 - satisfaction_prob).clip(0, 1),
                ].T,
            )

            # Simulation de formation
            formation = np.random.choice(["Oui", "Non"], n_samples, p=[0.6, 0.4])

            # Simulation de performance
            performance = np.random.choice(
                ["Faible", "Moyen", "Élevé"], n_samples, p=[0.2, 0.5, 0.3]
            )

            self.data = pd.DataFrame(
                {
                    "Age": ages,
                    "Salaire": salaires.astype(int),
                    "Années_Experience": np.clip(
                        ages - 22 + np.random.randint(-3, 4, n_samples), 0, 40
                    ),
                    "Nb_Projets": np.random.poisson(3, n_samples),
                    "Score_Performance": np.random.randint(60, 100, n_samples),
                    "Département": departements,
                    "Niveau": niveaux,
                    "Genre": genres,
                    "A_Quitté_Entreprise": [
                        "Oui" if x == 1 else "Non" for x in a_quitte
                    ],
                    "Satisfaction": satisfaction,
                    "Formation_Reçue": formation,
                    "Performance": performance,
                    "Télétravail": np.random.choice(
                        ["Oui", "Non"], n_samples, p=[0.7, 0.3]
                    ),
                    "Équipe": np.random.choice(["Alpha", "Beta", "Gamma"], n_samples),
                }
            )

        else:
            self.data = data.copy()

        # Identification des types de variables
        self.numeric_columns = self.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.all_columns = self.numeric_columns + self.categorical_columns

        if len(self.all_columns) < 2:
            raise ValueError("Le DataFrame doit contenir au moins 2 colonnes")

        # Création des widgets
        self.create_widgets()

        # Zone d'affichage du graphique
        self.output = widgets.Output()

        # Interface utilisateur
        self.setup_ui()

    def create_widgets(self):
        """Crée les widgets de l'interface"""
        # Dropdown pour la variable X (celle qu'on veut voir en détail)
        self.x_dropdown = widgets.Dropdown(
            options=self.all_columns,
            value=self.all_columns[0],
            description="Variable X:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Dropdown pour la variable de comparaison (hue - barres côte à côte)
        self.hue_dropdown = widgets.Dropdown(
            options=self.categorical_columns,
            value=self.categorical_columns[0] if self.categorical_columns else None,
            description="Comparer avec:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Slider pour limiter les valeurs affichées (si trop de catégories)
        self.limit_slider = widgets.IntSlider(
            value=20,
            min=5,
            max=50,
            step=5,
            description="Max catégories:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Checkbox pour trier par fréquence
        self.sort_checkbox = widgets.Checkbox(
            value=True,
            description="Trier par fréquence",
            style={"description_width": "initial"},
        )

        # Checkbox pour graphique horizontal
        self.horizontal_checkbox = widgets.Checkbox(
            value=False,
            description="Orientation horizontale",
            style={"description_width": "initial"},
        )

        # Slider pour la taille de la figure
        self.figsize_slider = widgets.IntSlider(
            value=14,
            min=10,
            max=20,
            step=2,
            description="Largeur figure:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Dropdown pour la palette de couleurs
        self.palette_dropdown = widgets.Dropdown(
            options=[
                "Set1",
                "Set2",
                "pastel",
                "dark",
                "muted",
                "bright",
                "deep",
                "colorblind",
            ],
            value="Set1",
            description="Palette:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )

        # Checkbox pour afficher les pourcentages
        self.percentage_checkbox = widgets.Checkbox(
            value=False,
            description="Afficher %",
            style={"description_width": "initial"},
        )

        # Bouton pour actualiser
        self.update_button = widgets.Button(
            description="🔄 Actualiser",
            button_style="primary",
            layout=widgets.Layout(width="150px"),
        )

        # Connexion des événements
        self.x_dropdown.observe(self.on_change, names="value")
        self.hue_dropdown.observe(self.on_change, names="value")
        self.limit_slider.observe(self.on_change, names="value")
        self.sort_checkbox.observe(self.on_change, names="value")
        self.horizontal_checkbox.observe(self.on_change, names="value")
        self.figsize_slider.observe(self.on_change, names="value")
        self.palette_dropdown.observe(self.on_change, names="value")
        self.percentage_checkbox.observe(self.on_change, names="value")
        self.update_button.on_click(self.on_button_click)

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Titre et description
        title_html = widgets.HTML(
            """
        <h3>📊 Distribution Brute - Barres Côte à Côte</h3>
        <p><i>Visualisez la distribution complète de vos données avec comparaison directe</i></p>
        <p><strong>Exemple:</strong> Age (X) vs A_Quitté_Entreprise → Une barre "Oui" et "Non" pour chaque âge</p>
        """
        )

        # Groupe des contrôles principaux
        main_controls = widgets.VBox(
            [
                widgets.HTML("<b>🎯 Variables:</b>"),
                widgets.HBox([self.x_dropdown, self.hue_dropdown]),
            ]
        )

        # Groupe des contrôles d'affichage
        display_controls = widgets.VBox(
            [
                widgets.HTML("<b>📐 Affichage:</b>"),
                widgets.HBox([self.limit_slider, self.sort_checkbox]),
                widgets.HBox([self.percentage_checkbox, self.horizontal_checkbox]),
            ]
        )

        # Groupe des contrôles de style
        style_controls = widgets.VBox(
            [
                widgets.HTML("<b>🎨 Style:</b>"),
                widgets.HBox(
                    [self.figsize_slider, self.palette_dropdown, self.update_button]
                ),
            ]
        )

        # Interface complète
        self.ui = widgets.VBox(
            [
                title_html,
                main_controls,
                display_controls,
                style_controls,
                widgets.HTML("<hr>"),
                self.output,
            ]
        )

        # Affichage initial
        self.update_plot()

    def on_change(self, change):
        """Callback appelé quand un widget change"""
        self.update_plot()

    def on_button_click(self, button):
        """Callback pour le bouton d'actualisation"""
        self.update_plot()

    def update_plot(self):
        """Met à jour le graphique"""
        with self.output:
            clear_output(wait=True)

            # Récupération des valeurs
            x_var = self.x_dropdown.value
            hue_var = self.hue_dropdown.value
            max_categories = self.limit_slider.value
            sort_by_freq = self.sort_checkbox.value
            horizontal = self.horizontal_checkbox.value
            fig_width = self.figsize_slider.value
            palette = self.palette_dropdown.value
            show_percentage = self.percentage_checkbox.value

            # Vérification des variables
            if x_var == hue_var:
                print("⚠️ Attention: Veuillez sélectionner deux variables différentes!")
                return

            try:
                # Préparation des données
                data_to_plot = self.data.copy()

                # Si la variable X est numérique avec beaucoup de valeurs uniques, on peut la discrétiser
                if x_var in self.numeric_columns:
                    unique_values = data_to_plot[x_var].nunique()
                    if unique_values > max_categories:
                        print(
                            f"💡 {x_var} a {unique_values} valeurs uniques. Création de {max_categories} groupes..."
                        )
                        data_to_plot[f"{x_var}_grouped"] = pd.cut(
                            data_to_plot[x_var], bins=max_categories, precision=1
                        )
                        x_var_plot = f"{x_var}_grouped"
                    else:
                        x_var_plot = x_var
                else:
                    x_var_plot = x_var

                # Limitation du nombre de catégories si nécessaire
                if data_to_plot[x_var_plot].nunique() > max_categories:
                    if sort_by_freq:
                        top_categories = (
                            data_to_plot[x_var_plot]
                            .value_counts()
                            .head(max_categories)
                            .index
                        )
                    else:
                        top_categories = data_to_plot[x_var_plot].unique()[
                            :max_categories
                        ]

                    data_to_plot = data_to_plot[
                        data_to_plot[x_var_plot].isin(top_categories)
                    ]
                    print(
                        f"📊 Affichage des {len(top_categories)} {'principales' if sort_by_freq else 'premières'} catégories"
                    )

                # Calcul de la hauteur selon le nombre de catégories
                n_categories = data_to_plot[x_var_plot].nunique()
                fig_height = max(8, min(16, n_categories * 0.4 + 6))

                # Création du graphique
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                # Configuration de la palette
                sns.set_palette(palette)

                # Ordre des catégories
                if sort_by_freq:
                    order = data_to_plot[x_var_plot].value_counts().index
                else:
                    if x_var in self.numeric_columns or "grouped" in x_var_plot:
                        order = sorted(data_to_plot[x_var_plot].unique())
                    else:
                        order = None

                # Création du countplot avec barres côte à côte
                if horizontal:
                    sns.countplot(
                        data=data_to_plot, y=x_var_plot, hue=hue_var, order=order, ax=ax
                    )
                    xlabel = "Nombre d'observations"
                    ylabel = x_var
                else:
                    sns.countplot(
                        data=data_to_plot, x=x_var_plot, hue=hue_var, order=order, ax=ax
                    )
                    xlabel = x_var
                    ylabel = "Nombre d'observations"

                # Rotation des labels si nécessaire
                if not horizontal and n_categories > 10:
                    plt.xticks(rotation=45, ha="right")
                elif horizontal and n_categories > 15:
                    plt.yticks(rotation=0)

                # Ajout des valeurs ou pourcentages sur les barres
                total_count = len(data_to_plot)

                for container in ax.containers:
                    if show_percentage:
                        labels = [
                            f"{v/total_count*100:.1f}%" if v > 0 else ""
                            for v in container.datavalues
                        ]
                    else:
                        labels = [
                            f"{int(v)}" if v > 0 else "" for v in container.datavalues
                        ]

                    ax.bar_label(
                        container,
                        labels=labels,
                        rotation=90 if horizontal else 0,
                        padding=3,
                        fontsize=9,
                    )

                # Personnalisation du graphique
                title = f"Distribution de {x_var} par {hue_var}"
                ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
                ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
                ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

                # Configuration de la légende
                ax.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc="upper left")

                # Ajout de la grille
                ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")

                # Statistiques descriptives
                total_obs = len(data_to_plot)
                categories_shown = data_to_plot[x_var_plot].nunique()
                hue_categories = data_to_plot[hue_var].nunique()

                stats_text = f"""📈 Statistiques:
                • Total observations: {total_obs}
                • Catégories {x_var}: {categories_shown}
                • Valeurs {hue_var}: {hue_categories}
                • Type: Distribution brute"""

                # Ajout de stats spécifiques
                if x_var in self.numeric_columns:
                    stats_text += f"\n• {x_var}: min={data_to_plot[x_var].min()}, max={data_to_plot[x_var].max()}"

                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9
                    ),
                    fontsize=10,
                    verticalalignment="top",
                )

                plt.tight_layout()
                plt.show()

                # Affichage d'informations supplémentaires
                print("✅ Graphique créé avec succès!")
                print(f"📊 Distribution brute de '{x_var}' comparée avec '{hue_var}'")
                print(
                    f"🎯 Chaque valeur de {x_var} a ses barres côte à côte pour {hue_var}"
                )

                # Tableau récapitulatif
                cross_tab = pd.crosstab(
                    data_to_plot[x_var_plot], data_to_plot[hue_var], margins=True
                )
                print("\n📋 Tableau croisé (top 10):")
                print(cross_tab.head(10))

            except Exception as e:
                print(f"❌ Erreur lors de la création du graphique: {str(e)}")
                print("💡 Vérifiez que les variables sélectionnées sont compatibles")

    def display(self):
        """Affiche l'interface interactive"""
        display(self.ui)


# Fonction principale pour utiliser facilement le widget
def create_raw_distribution_barplot(data=None):
    """
    Crée et affiche un graphique de distribution brute avec barres côte à côte

    Parameters:
    data (DataFrame, optional): DataFrame pandas avec vos données.
                               Si None, utilise des données d'exemple.

    Returns:
    InteractiveRawBarPlot: Instance de la classe pour une utilisation avancée
    """
    plot_widget = InteractiveRawBarPlot(data)
    plot_widget.display()
    return plot_widget


# Exemple d'utilisation simple
if __name__ == "__main__":
    print("🚀 Lancement du graphique de distribution brute...")
    print("💡 Utilisez: create_raw_distribution_barplot() pour commencer")

    # Utilisation avec des données d'exemple
    widget = create_raw_distribution_barplot()

# ═══════════════════════════════════════════════════════════════
# 📋 GUIDE D'UTILISATION:
# ═══════════════════════════════════════════════════════════════
"""
UTILISATION SIMPLE:
─────────────────────
widget = create_raw_distribution_barplot()

UTILISATION AVEC VOS DONNÉES:
───────────────────────────────
import pandas as pd
mes_donnees = pd.read_csv('mon_fichier.csv')
widget = create_raw_distribution_barplot(mes_donnees)

PRINCIPE:
──────────
✅ Distribution BRUTE - AUCUNE agrégation
✅ Pour chaque valeur de X, barres côte à côte selon hue
✅ Exemple: Age (22,23,24...) vs A_Quitté_Entreprise (Oui/Non)
   → Age 22: barre "Oui" + barre "Non" côte à côte
   → Age 23: barre "Oui" + barre "Non" côte à côte
   → etc.

EXEMPLES PARFAITS:
────────────────────
• Age vs A_Quitté_Entreprise → Distribution des départs par âge
• Salaire vs Satisfaction → Satisfaction pour chaque niveau de salaire
• Département vs Genre → Répartition hommes/femmes par département
• Score_Performance vs Formation_Reçue → Performance selon formation

FONCTIONNALITÉS:
──────────────────
✅ Variables numériques → Groupement automatique si trop de valeurs
✅ Tri par fréquence ou alphabétique
✅ Limitation du nombre de catégories affichées
✅ Pourcentages ou valeurs absolues
✅ Orientation horizontale/verticale
✅ Tableau croisé automatique
✅ Statistiques descriptives

REQUIREMENTS:
───────────────
pip install pandas numpy matplotlib seaborn ipywidgets
"""


def plot_vertical_feature_importance(feature_names, importances, top_n=10):
    """
    Affiche un diagramme en barres vertical des feature importances
    """
    # Prendre seulement les top_n features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]

    plt.figure(figsize=(10, 8))

    bars = plt.bar(
        range(len(top_features)),
        top_importances,
        color="purple",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.xlabel("Features", fontsize=12, fontweight="bold")
    plt.ylabel("Importance Globale", fontsize=12, fontweight="bold")
    plt.title(
        f"Top {top_n} Features les Plus Importantes", fontsize=16, fontweight="bold"
    )

    plt.xticks(range(len(top_features)), top_features, rotation=45, ha="right")

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


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
