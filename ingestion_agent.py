import json, os, chromadb, shutil

from sentence_transformers import SentenceTransformer





class IndexeurRecrutement:

    def __init__(self):

        self.base_dir = r"C:\Users\dell\Desktop\PFE_AI_Matching_Project\Shared_Data"

        self.json_path = os.path.join(self.base_dir, "final_results.json")

        self.db_path = os.path.join(self.base_dir, "my_database_final")



        # Suppression de l'ancienne base pour appliquer la nouvelle métrique

        if os.path.exists(self.db_path):

            shutil.rmtree(self.db_path)

            print("🗑️ Ancienne base supprimée pour mise à jour du moteur de calcul.")



        self.client = chromadb.PersistentClient(path=self.db_path)



        # CRITIQUE : Utilisation de 'cosine' pour éviter les scores à 0%

        self.collection = self.client.create_collection(

            name="cv_collection",

            metadata={"hnsw:space": "cosine"}

        )

        self.model = SentenceTransformer('all-MiniLM-L6-v2')



    def indexer_donnees(self):

        with open(self.json_path, 'r', encoding='utf-8') as f:

            data = json.load(f)



        print(f"📊 Indexation de {len(data)} profils...")



        for i, item in enumerate(data):

            # On fusionne toutes les données pour un matching puissant

            contenu_complet = f"{item.get('nom_complet')} {item.get('experience_principale')} "

            contenu_complet += " ".join(item.get('hard_skills', [])) + " "

            contenu_complet += " ".join(item.get('soft_skills', []))



            # Extraction propre des métadatas

            nom = str(item.get('nom_complet', f"Cand_{i}"))

            spec = item.get('experience_principale', "Ingénieur")



            # Si la spécialité est une liste, on la simplifie

            if isinstance(spec, list):

                spec = "Ingénieur Spécialisé"



            vecteur = self.model.encode(contenu_complet).tolist()



            self.collection.add(

                ids=[f"id_{i}"],

                embeddings=[vecteur],

                metadatas={"nom": nom, "specialite": str(spec)},

                documents=[contenu_complet]

            )

            print(f"✅ Indexé : {nom}")





if __name__ == "__main__":

    indexeur = IndexeurRecrutement()

    indexeur.indexer_donnees()

    print("✨ Ingestion terminée avec succès (Mode Cosine).")
