import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


def run_ingestion_agent():
    print("🚀 [AGENT INGESTION] : Initialisation du Pipeline...")

    # 1. Configuration de la base de données vectorielle (ChromaDB)
    # On définit le chemin où la base sera stockée physiquement
    db_path = "./my_database_final"
    client = chromadb.PersistentClient(path=db_path)

    # On supprime l'ancienne collection pour garantir la cohérence des données (Clean Start)
    try:
        client.delete_collection(name="cv_collection")
        print("🗑️ Ancienne collection supprimée.")
    except:
        print("✨ Création d'une nouvelle collection.")

    # Création de la collection avec la métrique 'cosine' pour la précision du matching
    collection = client.create_collection(
        name="cv_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # 2. Chargement du modèle de Langage (S-BERT)
    # Ce modèle transforme le texte en vecteurs numériques de 384 dimensions
    print("⏳ Chargement du modèle Sentence-BERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Chargement des données parsées (JSON)
    # C'est ici qu'on récupère le travail fait par l'agent de parsing
    json_input = 'final_results.json'
    if not os.path.exists(json_input):
        print(f"❌ Erreur : Le fichier {json_input} est introuvable.")
        return

    with open(json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4. Vectorisation et Indexation
    print(f"📊 Traitement de {len(data)} profils en cours...")

    for i, item in enumerate(data):
        nom = item.get('nom', f"Candidat_{i}")

        # Concaténation des informations clés pour enrichir le vecteur
        # On regroupe Spécialité, Compétences et Expérience
        full_content = f"{item.get('specialite', '')} {item.get('competences', '')} {item.get('experience', '')}"

        # Transformation du texte en vecteur numérique
        vector = model.encode(full_content).tolist()

        # Ajout à la base vectorielle avec les métadonnées
        collection.add(
            ids=[f"id_{i}"],
            embeddings=[vector],
            metadatas=[{"nom": nom}],
            documents=[full_content]  # Indispensable pour la recherche sémantique
        )
        print(f"✅ Profil indexé avec succès : {nom}")

    print("\n✨ OPÉRATION TERMINEE : La base vectorielle est prête pour le matching.")


if __name__ == "__main__":
    run_ingestion_agent()