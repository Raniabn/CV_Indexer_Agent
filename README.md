# 🤖 AI Recruitment Assistant - Module d'Ingestion Vectorielle

Ce dépôt contient le moteur d'indexation (Ingestion Agent) de notre système de matching intelligent. Ce module est responsable de la transformation des CVs parsés en représentations vectorielles (Embeddings) pour permettre une recherche sémantique avancée.

## 🌟 Aperçu du Projet
L'objectif de cet agent est de créer une base de données vectorielle capable de comprendre le contexte et le sens des compétences des candidats, allant au-delà d'une simple recherche par mots-clés traditionnelle.

## 🛠️ Stack Technique
Pour garantir la précision du matching, nous avons intégré les technologies suivantes :

* **Modèle d'Embedding** : `Sentence-BERT (SBERT)` via la bibliothèque `sentence-transformers`.
    * *Modèle utilisé* : `all-MiniLM-L6-v2`.
    * *Dimensions* : Chaque profil est représenté par un vecteur de **384 dimensions**.
* **Vector Database** : `ChromaDB`.
    * *Métrique de Similarité* : **Cosine Similarity** (pour mesurer la proximité entre les CVs et les offres).
* **Format de Données** : Entrées structurées en **JSON**.

## 🚀 Fonctionnement du Pipeline (ETL Vectoriel)
1.  **Extract** : Lecture des données extraites du fichier `final_results.json`.
2.  **Transform (Vectorisation)** : Combinaison des champs clés (*Spécialité, Compétences, Expérience*) et transformation en vecteurs numériques via S-BERT.
3.  **Load** : Indexation et stockage des vecteurs et des métadonnées dans la collection locale `my_database_final`.

## 📂 Structure du Repository
* `ingestion_agent.py` : Le script principal pour traiter et indexer les données.
* `final_results.json` : Fichier contenant les données des candidats (Input).
* `requirements.txt` : Liste des dépendances nécessaires.
* `.gitignore` : Configuration pour exclure les fichiers volumineux et temporaires (comme la DB locale).

## 💻 Installation & Utilisation

### 1. Prérequis
Assurez-vous d'avoir Python 3.8+ installé.

### 2. Installation des dépendances
```bash
pip install -r requirements.txt