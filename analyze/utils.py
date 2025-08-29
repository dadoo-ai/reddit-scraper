import pandas as pd
from pathlib import Path

def group_csv_files(folder_path: str, output_folder: str, file_name: str):
    # Dossier où sont tes CSV
    folder = Path(folder_path)

    # Récupérer tous les fichiers CSV du dossier
    csv_files = list(folder.glob("*.csv"))

    # Charger et concaténer tous les CSV
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"chargé : {f.name} ({df.shape[0]} lignes)")
        except Exception as e:
            print(f"erreur sur {f.name} : {e}")

    # Fusion en un seul DataFrame
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        print(f"\nFusion terminée → total {merged.shape[0]} lignes / {merged.shape[1]} colonnes")

        # Sauvegarder en un seul CSV
        merged.to_csv(f"{output_folder}/{file_name}.csv", index=False, encoding="utf-8-sig")

        print(f" fichier écrit : {output_folder}/{file_name}.csv")
    else:
        print("Aucun CSV trouvé ou lisible.")
