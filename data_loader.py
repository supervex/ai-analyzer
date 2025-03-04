import os
import json

def load_json_files(folder_path):
    """Carica e unisce tutti i file JSON dalla cartella specificata."""
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    obj = json.load(file)
                    if isinstance(obj, list):
                        data.extend(obj)
                    elif isinstance(obj, dict):
                        data.append(obj)
                    else:
                        print(f"Formato inatteso in {filename}: {type(obj)}")
                except json.JSONDecodeError as e:
                    print(f"Errore nel decodificare {filename}: {e}")
    return data

