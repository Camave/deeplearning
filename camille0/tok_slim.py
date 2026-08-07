from datasets import load_dataset, Dataset, load_from_disk
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PARAMÈTRES D'EXTRACTION ---
NOMBRE_DOCUMENTS = 150_000  # Cible ~1 Go
NOM_DOSSIER_LOCAL = "corpus_fr_1gb_mc4"
LANGUE = "fr"

# =======================================================
# BLOC 1 : TÉLÉCHARGEMENT ET SAUVEGARDE (La partie longue)
# =======================================================

if not os.path.exists(NOM_DOSSIER_LOCAL):
    print(f"Chargement du split français de C4 (multilingue) en mode streaming...")
    
    # Correction de l'erreur: Utilisation de la configuration 'fr' pour MC4
    try:
        ds_stream = load_dataset("mc4", languages=[LANGUE], split="train", streaming=True)
    except Exception as e:
        print(f"\nTentative de chargement 'mc4' échouée: {e}")
        print("Tentative avec la configuration 'multilingual' du C4...")
        ds_stream = load_dataset("allenai/c4", f"multilingual/{LANGUE}", split="train", streaming=True)
    
    # Extraction des documents
    print(f"Extraction des {NOMBRE_DOCUMENTS:,} premiers documents...")
    echantillon = islice(ds_stream, NOMBRE_DOCUMENTS)

    # Conversion et chargement en RAM
    try:
        liste_data = list(echantillon)
    except MemoryError:
        print("\nERREUR: Mémoire vive (RAM) insuffisante. Réduisez le NOMBRE_DOCUMENTS.")
        exit()

    # Création et Sauvegarde du Dataset local
    ds_local = Dataset.from_list(liste_data)
    ds_local.save_to_disk(NOM_DOSSIER_LOCAL)

    print(f"\n✅ Corpus Français de {len(ds_local):,} documents sauvegardé dans : {NOM_DOSSIER_LOCAL}")
else:
    print(f"Le dataset '{NOM_DOSSIER_LOCAL}' existe déjà. Chargement local...")


# =======================================================
# BLOC 2 : VISUALISATION DU CONTENU ET DES STATISTIQUES
# =======================================================

# Chargement du dataset local (rapide, que ce soit la 1ère ou 2e exécution)
ds_local = load_from_disk(NOM_DOSSIER_LOCAL)


### A. Inspection du Contenu (Les 3 Premiers)
print("\n" + "="*50)
print("🔍 INSPECTION DE LA QUALITÉ DES DONNÉES")
print("="*50)
for i in range(3):
    print(f"--- Document {i+1} ---")
    text_sample = ds_local[i]['text'][:500] + "..." if len(ds_local[i]['text']) > 500 else ds_local[i]['text']
    print(text_sample)
    print("-" * 50)


### B. Visualisation des Statistiques de Longueur

# 1. Fonction pour calculer le nombre de mots
def count_words(example):
    example["word_count"] = len(example["text"].split())
    return example

# 2. Appliquer la fonction
# Utilisation d'un sous-ensemble pour ne pas tout mapper si le dataset grossit
ds_sample_for_viz = ds_local.select(range(5000)).map(count_words)
word_counts = ds_sample_for_viz["word_count"]

# 3. Afficher les statistiques clés
print("\n" + "="*50)
print("📊 STATISTIQUES DE LONGUEUR (sur 5000 échantillons)")
print(f"Longueur moyenne des documents : {np.mean(word_counts):.0f} mots")
print(f"Longueur médiane des documents : {np.median(word_counts):.0f} mots")
print(f"Nombre de documents de moins de 10 mots : {np.sum(np.array(word_counts) < 10)}")


# 4. Tracer l'histogramme de fréquence
plt.figure(figsize=(10, 6))
# Limiter l'affichage à 500 mots, car les longs articles faussent la vue
plt.hist(word_counts, bins=50, range=(0, 500), color='salmon', edgecolor='black') 
plt.title(f'Distribution de la Longueur des Documents MC4 ({LANGUE})')
plt.xlabel('Nombre de Mots par Document')
plt.ylabel('Fréquence')
plt.grid(axis='y', alpha=0.5)
plt.show()