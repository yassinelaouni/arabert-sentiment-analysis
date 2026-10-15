# Analyse des sentiments arabes avec AraBERT

Notebook pédagogique en français : fine-tuning de `aubmindlab/bert-base-arabertv02` avec PyTorch et Hugging Face, sélection par F1 macro de validation, puis évaluation du test officiel.

## Fichiers

- `arabert_sentiment_analysis.ipynb` : 42 cellules alternant explications et code complet.
- `requirements.txt` : dépendances compatibles avec **Python 3.10–3.12**.
- `.gitignore` : exclusions des données, environnements, checkpoints et sorties locales.

Les données et poids entraînés ne sont pas versionnés. Le dossier local `data/` doit contenir les quatre TSV du corpus Kaggle avant l’exécution.

## Lancement local

Cloner le dépôt puis, sous Windows avec Python 3.12 installé :

```powershell
git clone https://github.com/yassinelaouni/arabert-sentiment-analysis.git
cd arabert-sentiment-analysis
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name arabert --display-name "Python (AraBERT)"
.\.venv\Scripts\python.exe -m jupyterlab arabert_sentiment_analysis.ipynb
```

Sélectionner le noyau **Python (AraBERT)**. Python 3.13/3.14 n’est pas pris en charge par cette pile épinglée. Sur Linux/macOS, utiliser `python3.12 -m venv .venv` et `.venv/bin/python`.

Pour NVIDIA, installer d’abord une roue PyTorch compatible avec le pilote via le [sélecteur officiel](https://pytorch.org/get-started/locally/). La contrainte `torch>=2.6,<2.9` permet de conserver une installation CUDA compatible ; sa version exacte est enregistrée à chaque exécution. Les autres dépendances principales sont épinglées. Aucun token n’est inclus.

## Kaggle / Google Colab

Importer le notebook, sélectionner un GPU et activer Internet pour l’installation et le téléchargement initial du modèle. Utiliser un noyau Python 3.10–3.12. Sur Kaggle, ajouter le dataset en entrée : les TSV seront recherchés automatiquement sous `/kaggle/input`. Sur Colab, importer et extraire l’archive avec le panneau Fichiers, puis définir `DATA_DIR`, par exemple `/content/data`.

Exécuter les cellules dans l’ordre. La première installe les dépendances ; mettre `INSTALL_DEPENDENCIES=False` si elles sont déjà installées. Redémarrer le noyau si des bibliothèques étaient chargées avant leur installation.

## Données et options

Source : [Arabic Sentiment Twitter Corpus — mksaad](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus).

Télécharger et extraire les quatre fichiers suivants dans `data/` :

```text
data/
  train_Arabic_tweets_negative_20190413.tsv
  train_Arabic_tweets_positive_20190413.tsv
  test_Arabic_tweets_negative_20190413.tsv
  test_Arabic_tweets_positive_20190413.tsv
```

Les quatre TSV inspectés sont en UTF-8, sans en-tête, avec une tabulation entre `label` et `tweet`. Le train officiel contient 47 000 lignes et le test officiel 11 751 lignes ; `neg` signifie négatif, `pos` positif. Le notebook vérifie ces caractéristiques avant lecture. La copie en fichiers individuels `arabic_tweets/` n’est pas chargée.

Si les données ne sont plus présentes, télécharger et extraire l’archive dans `data/`, ou régler `DOWNLOAD_WITH_KAGGLE=True` après configuration privée de l’API Kaggle **hors du notebook**. Fournir les quatre TSV, y compris le test. Conserver `EXPECT_OFFICIAL_TEST=True` pour ce corpus.

- `QUICK_MODE=True` par défaut : 256/128/128 exemples au maximum, une époque, avec le vrai modèle. Les effectifs peuvent diminuer après déduplication des entrées tokenisées.
- Pour l’expérience complète : mettre `QUICK_MODE=False`, redémarrer le noyau et tout exécuter ; trois époques sont alors utilisées.
- `DATA_DIR`, le plafond de longueur et les paramètres d’entraînement sont configurables. La longueur effective est choisie sur le train uniquement.
- Sans CUDA : repli CPU ; le fine-tuning complet peut être très long. Le notebook détecte aussi MPS, sans précision mixte.

Les doublons du train présents dans le test sont retirés du train. Les contradictions internes et collisions de tokenisation sont exclues avec un journal : les scores portent donc sur le test officiel **nettoyé**, ou son sous-ensemble en mode rapide. Le test n’intervient jamais dans le choix du checkpoint.

## Sorties et vérification

Chaque exécution écrit dans `outputs/quick_...` ou `outputs/full_...` : modèle/tokenizer, configuration d’inférence, métriques, courbes, matrice de confusion, prédictions, erreurs et manifestes des partitions. Une cellule montre le rechargement local et vérifie la cohérence des probabilités. Aucun suivi externe ni publication automatique.

**Vérifications réalisées le 15 octobre 2026 :** format Jupyter et syntaxe des 21 cellules de code validés ; dépendances installées et contrôlées sans conflit ; imports vérifiés sous Windows/Python 3.12.14/PyTorch 2.6.0 CPU. Les 58 751 lignes réelles ont été inspectées, lues et prétraitées ; la préparation du mode rapide 256/128/128 et les assertions de séparation sont passées. Un test supplémentaire de bout en bout avec **quatre exemples réels par partition** et un pas d’optimisation a exécuté le fine-tuning du vrai AraBERT, la sélection du checkpoint, les courbes, les rapports, l’inférence, la sauvegarde et le rechargement. Les probabilités avant/après sauvegarde concordent à `1e-4` près.

**Reste à exécuter :** le mode rapide standard avec entraînement sur 256 exemples, puis les trois époques du mode complet sur GPU. CUDA et la précision mixte n’ont pas pu être testés dans cet environnement CPU. Le téléchargement par le client Kaggle avec authentification privée n’a pas été testé ; l’archive publique Kaggle a bien été téléchargée et inspectée. Le notebook livré ne contient aucun résultat d’entraînement prérempli et aucun score complet n’est annoncé. Les scores du test technique à quatre exemples ne mesurent pas les performances du corpus. Les probabilités ne sont pas des certitudes, et aucune classe neutre n’a été apprise.
