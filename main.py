import joblib
import math  
import tkinter as tk
import numpy as np
from tkinter import scrolledtext
from data_loader import load_json_files
from text_processing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


import nltk
import os
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def compute_ai_centroid(vectorizer, conversations):
    """
    Calcola il centroide delle frasi AI a partire dalle conversazioni.
    Usa le risposte AI già processate (cleaned) per creare un vettore medio.
    """
    ai_texts = [clean_text(convo["ai_response"]) for convo in conversations if convo["ai_response"]]
    if not ai_texts:
        print("Nessuna frase AI trovata per il calcolo del centroide.")
        return None
    # Trasforma le frasi AI in vettori e calcola la media
    X_ai = vectorizer.transform(ai_texts)
    centroid = np.array(X_ai.mean(axis=0))
    return centroid

def classify_sentence_similarity(sentence, vectorizer, ai_centroid, threshold=0.07):
    """
    Classifica una frase basandosi sulla similarità con il centroide delle frasi AI.
    
    Parametri:
      - sentence: la frase da classificare.
      - vectorizer: il TfidfVectorizer addestrato.
      - ai_centroid: il centroide delle frasi AI (array).
      - threshold: soglia per decidere se la frase è tipicamente AI (default 0.07).
    
    Restituisce:
      - label: "ai" se la similarità è >= soglia, altrimenti "human".
      - confidence: percentuale di similarità (0-100%).
    """
    sentence_vector = vectorizer.transform([sentence])
    sim = cosine_similarity(sentence_vector, ai_centroid).flatten()[0]
    confidence = sim * 100  # Convertiamo in percentuale
    label = "ai" if sim >= threshold else "human"
    return label, confidence


# Assicurati di scaricare tutte le risorse necessarie per il tokenizing.
nltk.download('punkt')
nltk.download('punkt_tab')

CHECKPOINT_PATH = "optimized_classifier.pkl"

def save_checkpoint(model, checkpoint_path=CHECKPOINT_PATH):
    joblib.dump(model, checkpoint_path)
    print("Checkpoint salvato.")

def load_checkpoint(checkpoint_path=CHECKPOINT_PATH):
    if os.path.exists(checkpoint_path):
        model = joblib.load(checkpoint_path)
        print("Checkpoint caricato.")
        return model
    else:
        print("Nessun checkpoint trovato.")
        return None

def extract_conversation(entry):
    conversation = {"topic": None, "human_request": None, "ai_response": None}
    
    if "topic" in entry:
        conversation["topic"] = entry["topic"]
    elif "id" in entry:
        conversation["topic"] = entry["id"]
    else:
        print("Attenzione: né 'topic' né 'id' presenti.")
        conversation["topic"] = "unknown"
    
    if "input" in entry:
        text = entry["input"]
        if "[|Human|]" in text and "[|AI|]" in text:
            human_part = text.split("[|Human|]")[1].split("[|AI|]")[0].strip()
            ai_part = text.split("[|AI|]")[1].strip()
            conversation["human_request"] = human_part
            conversation["ai_response"] = ai_part
        elif "[|Umano|]" in text and "[|AI|]" in text:
            human_part = text.split("[|Umano|]")[1].split("[|AI|]")[0].strip()
            ai_part = text.split("[|AI|]")[1].strip()
            conversation["human_request"] = human_part
            conversation["ai_response"] = ai_part
        else:
            conversation["human_request"] = text
    elif "conversations" in entry:
        if not entry["conversations"]:
            print("Attenzione: 'conversations' è vuoto in un elemento.")
        for conv in entry["conversations"]:
            from_lower = conv.get("from", "").lower()
            if from_lower in ["human", "umano"] and not conversation["human_request"]:
                conversation["human_request"] = conv.get("value", "").strip()
            elif from_lower in ["gpt", "ai"] and not conversation["ai_response"]:
                conversation["ai_response"] = conv.get("value", "").strip()
    else:
        conversation["human_request"] = ""
        conversation["ai_response"] = ""
    
    return conversation

def prepare_dataset_mix(conversations):
    """
    Prepara il dataset creando un campione per ogni richiesta e risposta.
    Inoltre, calcola i sample weights basati sulla frequenza delle classi.
    Questo approccio bilancia il dataset, dando più peso agli esempi meno numerosi.
    """
    texts = []
    labels = []
    for convo in conversations:
        if convo["human_request"]:
            texts.append(clean_text(convo["human_request"]))
            labels.append("human")
        if convo["ai_response"]:
            texts.append(clean_text(convo["ai_response"]))
            labels.append("ai")
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    print("Distribuzione classi:", counts)
    print(f"Dataset preparato: {len(texts)} campioni totali.")
    return texts, labels, sample_weights

def train_classifier_weighted(texts, labels, sample_weights):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X, labels, sample_weight=sample_weights)
    
    print("Modello addestrato con sample weights con successo.")
    
    with open("model.pkl", "wb") as f:
        pickle.dump((vectorizer, clf), f)
    
    return vectorizer, clf

def classify_text_extended(user_input, vectorizer, clf, min_sentence_words=3, sentence_threshold=0.9):
    """
    - Divide il testo in frasi usando nltk.sent_tokenize.
    - Per ogni frase (con almeno 'min_sentence_words' parole):
       • Calcola la probabilità che la frase sia "ai".
       • Assegna un peso pari al numero di parole.
    - Calcola due metriche globali:
         1. Media pesata delle probabilità.
         2. Votazione maggioritaria.
    Ritorna:
      - analyzed_results: lista di tuple (frase, label, percentuale AI, peso)
      - global_label_weighted: classificazione globale basata sulla media pesata.
      - global_prob_weighted: probabilità globale (in percentuale) basata sul calcolo pesato.
      - global_label_majority: classificazione globale basata sul voto maggioritario.
      - majority_counts: dizionario con conteggio dei voti per "ai" e "human".
    """
    sentences = nltk.sent_tokenize(user_input)
    
    analyzed_results = []
    total_weighted_prob = 0.0
    total_weight = 0
    vote_ai = 0
    vote_human = 0
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) < min_sentence_words:
            continue
        
        sentence_vector = vectorizer.transform([sentence])
        proba = clf.predict_proba(sentence_vector)
        ai_index = list(clf.classes_).index("ai")
        ai_prob = proba[0][ai_index]
        weight = len(words)
        
        total_weighted_prob += ai_prob * weight
        total_weight += weight
        
        label = "ai" if ai_prob >= sentence_threshold else "human"
        if label == "ai":
            vote_ai += 1
        else:
            vote_human += 1
        
        analyzed_results.append((sentence, label, ai_prob * 100, weight))
    
    if total_weight > 0:
        global_prob_weighted = total_weighted_prob / total_weight
    else:
        global_prob_weighted = 0
    
    global_label_weighted = "ai" if global_prob_weighted >= sentence_threshold else "human"
    
    if vote_ai > vote_human:
        global_label_majority = "ai"
    elif vote_human > vote_ai:
        global_label_majority = "human"
    else:
        global_label_majority = "ambiguous"
    
    return analyzed_results, global_label_weighted, global_prob_weighted * 100, global_label_majority, {"ai": vote_ai, "human": vote_human}

def extract_ai_text(text):
    """
    Estrae la parte del testo corrispondente alla risposta AI.
    Se sono presenti tag come [|AI|] o [| Ai |], restituisce il testo che segue.
    Altrimenti, restituisce il testo originale.
    """
    if "[|AI|]" in text:
        parts = text.split("[|AI|]")
        ai_text = " ".join(part.strip() for part in parts[1:] if part.strip())
        return ai_text
    elif "[| Ai |]" in text:
        parts = text.split("[| Ai |]")
        ai_text = " ".join(part.strip() for part in parts[1:] if part.strip())
        return ai_text
    else:
        return text


def create_gui(conversations, vectorizer, clf, ai_centroid):
    def on_classify():
        user_input = input_textbox.get("1.0", "end-1c")
        if not user_input.strip():
            phrase_output_textbox.delete("1.0", "end")
            final_output_textbox.delete("1.0", "end")
            phrase_output_textbox.insert("end", "Errore: Inserisci del testo da analizzare.\n")
            return

        # Estrae la parte di risposta AI dal testo in input.
        ai_text = extract_ai_text(user_input)

        # Pulizia delle aree di output.
        phrase_output_textbox.delete("1.0", "end")
        final_output_textbox.delete("1.0", "end")
        
        # === Sezione 1: Classificazione per Frase (Similarità) ===
        phrase_output_textbox.insert("end", "=== Analisi per Frase (Similarità) ===\n\n")
        similarity_results = []
        sentences = nltk.sent_tokenize(ai_text)
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 3:  # Considera solo frasi con almeno 3 parole.
                continue
            # Calcola la similarità per ciascuna frase, utilizzando il threshold di 0.07.
            sim_label, sim_confidence = classify_sentence_similarity(sentence, vectorizer, ai_centroid, threshold=0.07)
            similarity_results.append((sentence, sim_label, sim_confidence))
        
        if similarity_results:
            for sent, sim_label, sim_conf in similarity_results:
                phrase_output_textbox.insert("end", f"Frase: {sent}\n")
                phrase_output_textbox.insert("end", f"  -> Similarità: {sim_conf:.2f}% => Classificata come {sim_label.upper()}\n\n")
        else:
            phrase_output_textbox.insert("end", "Nessuna frase analizzata per similarità.\n")
        
        # === Sezione 2: Risultato Complessivo Finale (Aggregazione e Ricalibrazione) ===
        if similarity_results:
            avg_similarity = sum(sim_conf for _, _, sim_conf in similarity_results) / len(similarity_results)
        else:
            avg_similarity = 0.0

        # Applichiamo una funzione logistica per ricalibrare l'output.
        # Con k = 0.2 e x₀ = 10, se avg_similarity = 10, p_AI = 50%.
        # p_AI = 100 / (1 + exp(-0.2*(avg_similarity - 10)))
        p_AI = 100 / (1 + math.exp(-0.2 * (avg_similarity - 10)))
        p_Human = 100 - p_AI

        final_classification = "AI" if p_AI >= 50 else "HUMAN"

        final_output_textbox.insert("end", "=== Risultati Finali Aggregati ===\n\n")
        final_output_textbox.insert("end", f"Media Similarità Aggregata (raw): {avg_similarity:.2f}%\n")
        final_output_textbox.insert("end", f"Probabilità AI: {p_AI:.2f}%\n")
        final_output_textbox.insert("end", f"Probabilità Human: {p_Human:.2f}%\n")
        final_output_textbox.insert("end", f"Classificazione Finale: {final_classification}\n")
        
        print_to_console("Analisi completata.")
        
    def print_to_console(message):
        console_textbox.config(state=tk.NORMAL)
        console_textbox.insert(tk.END, message + "\n")
        console_textbox.yview(tk.END)
        console_textbox.config(state=tk.DISABLED)

    # Creazione della finestra principale
    root = tk.Tk()
    root.title("Analizzatore di Conversazioni AI")
    
    instructions = tk.Label(
        root,
        text="Istruzioni:\n1. Inserisci il testo da analizzare.\n2. Premi 'Classifica'.\n"
             "Verranno mostrate due aree di output:\n"
             "   - Classificazione Frase per Frase\n"
             "   - Risultati Finali Aggregati (probabilità ricalibrate)",
        font=("Arial", 12), justify="left"
    )
    instructions.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
    
    input_label = tk.Label(root, text="Testo di input:")
    input_label.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    input_textbox = scrolledtext.ScrolledText(root, width=80, height=10)
    input_textbox.grid(row=2, column=0, columnspan=2, padx=10, pady=5)
    
    classify_button = tk.Button(root, text="Classifica", command=on_classify, font=("Arial", 12))
    classify_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
    
    # Area di output per la classificazione frase per frase
    phrase_label = tk.Label(root, text="Classificazione Frase per Frase:")
    phrase_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
    phrase_output_textbox = scrolledtext.ScrolledText(root, width=60, height=10)
    phrase_output_textbox.grid(row=5, column=0, padx=10, pady=5)
    
    # Area di output per i risultati finali aggregati
    final_label = tk.Label(root, text="Risultati Finali Aggregati:")
    final_label.grid(row=4, column=1, padx=10, pady=5, sticky="w")
    final_output_textbox = scrolledtext.ScrolledText(root, width=60, height=10)
    final_output_textbox.grid(row=5, column=1, padx=10, pady=5)
    
    # Area per il log (facoltativa)
    console_label = tk.Label(root, text="Log di elaborazione:")
    console_label.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="w")
    console_textbox = scrolledtext.ScrolledText(root, width=125, height=10, state=tk.DISABLED)
    console_textbox.grid(row=7, column=0, columnspan=2, padx=10, pady=5)
    
    print_to_console(f"Elementi caricati: {len(conversations)}")
    texts_temp, labels_temp, sample_weights = prepare_dataset_mix(conversations)
    print_to_console(f"Dataset preparato: {len(texts_temp)} campioni totali.")
    
    root.mainloop()



def save_model_with_threshold(model, vectorizer, optimal_threshold, filename="model_with_threshold.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "threshold": optimal_threshold}, f)
    print("Modello e soglia ottimale salvati in", filename)

def load_model_with_threshold(filename="model_with_threshold.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print("Modello con soglia ottimale caricato da", filename)
        return data["model"], data["vectorizer"], data["threshold"]
    else:
        print("Nessun file di modello con soglia trovato.")
        return None, None, None

def main():
    # Percorso della cartella dati
    data_folder = r"C:\Users\Alberto\Desktop\AiAnalyzer\data"
    data = load_json_files(data_folder)
    print(f"Elementi caricati: {len(data)}")
    
    # Estrae le conversazioni e filtra quelle valide
    conversations = [extract_conversation(entry) for entry in data]
    valid_conversations = [convo for convo in conversations if convo["human_request"] or convo["ai_response"]]
    
    # Prepara il dataset "mix" con sample weights
    texts, labels, sample_weights = prepare_dataset_mix(valid_conversations)
    
    # Suddivide il dataset in training e validation set (80/20)
    texts_train, texts_val, labels_train, labels_val, weights_train, weights_val = train_test_split(
        texts, labels, sample_weights, test_size=0.2, random_state=42
    )
    
    # Carica il modello se esiste, altrimenti addestra con i sample weights
    model = load_checkpoint()
    if model is not None:
        vectorizer = model.named_steps['vectorizer']
        clf = model.named_steps['classifier']
    else:
        vectorizer, clf = train_classifier_weighted(texts_train, labels_train, weights_train)
    
    # Ottimizzazione della soglia sul validation set con calibrazione:
    X_val = vectorizer.transform(texts_val)
    calibrated_clf = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
    calibrated_clf.fit(X_val, labels_val)
    clf = calibrated_clf
    
    probabilities = clf.predict_proba(X_val)
    ai_index = list(clf.classes_).index("ai")
    ai_probs = probabilities[:, ai_index]
    
    plt.figure(figsize=(8,6))
    plt.hist(ai_probs, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribuzione delle probabilità 'AI' (Calibrato)")
    plt.xlabel("Probabilità AI")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.show()
    
    precision, recall, thresholds = precision_recall_curve(labels_val, ai_probs, pos_label="ai")
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    best_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_index]
    print(f"Soglia ottimale trovata: {optimal_threshold:.2f}")
    
    def classify_with_threshold(probs, threshold):
        return ["ai" if prob >= threshold else "human" for prob in probs]
    
    y_val_pred = classify_with_threshold(ai_probs, optimal_threshold)
    print(classification_report(labels_val, y_val_pred))
    
    save_model_with_threshold(model, vectorizer, optimal_threshold, filename="model_with_threshold.pkl")
    
    # Calcolo del centroide AI: rappresenta la "media" (vettoriale) delle risposte AI
    ai_centroid = compute_ai_centroid(vectorizer, valid_conversations)
    if ai_centroid is None:
        print("Impossibile calcolare il centroide AI. Assicurati di avere risposte AI valide nel dataset.")
    
    # Avvio della GUI, passando anche il centroide AI per la classificazione per similarità
    create_gui(valid_conversations, vectorizer, clf, ai_centroid)

if __name__ == "__main__":
    main()

