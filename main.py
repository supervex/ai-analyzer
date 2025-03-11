import joblib
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

import nltk
import os
import pickle

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

def prepare_dataset(conversations):
    human_sentences = []
    ai_sentences = []
    for convo in conversations:
        if convo["human_request"]:
            human_sentences.append(clean_text(convo["human_request"]))
        if convo["ai_response"]:
            ai_sentences.append(clean_text(convo["ai_response"]))
    texts = human_sentences + ai_sentences
    labels = ['human'] * len(human_sentences) + ['ai'] * len(ai_sentences)
    print(f"Dataset preparato: {len(texts)} frasi totali.")
    return texts, labels

def optimize_hyperparameters(texts, labels):
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    param_grid = {
        'vectorizer__ngram_range': [(1,1), (1,2), (1,3)],
        'vectorizer__max_features': [None, 5000, 10000],
        'vectorizer__min_df': [1, 2, 5],
        'vectorizer__max_df': [0.7, 0.9, 1.0],
        'classifier__alpha': [0.1, 0.5, 1.0]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, n_jobs=1)
    grid_search.fit(texts, labels)
    
    print("Migliori parametri trovati:", grid_search.best_params_)
    print("Miglior accuracy ottenuta:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    save_checkpoint(best_model)
    
    return best_model

def load_trained_model():
    model = load_checkpoint()
    if model is not None:
        vectorizer = model.named_steps['vectorizer']
        clf = model.named_steps['classifier']
        return vectorizer, clf
    else:
        return None, None

def train_classifier(texts, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X, labels)
    
    print("Modello di fallback addestrato con successo.")
    
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
            # Saltiamo le frasi troppo brevi
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
    
    # Votazione maggioritaria
    if vote_ai > vote_human:
        global_label_majority = "ai"
    elif vote_human > vote_ai:
        global_label_majority = "human"
    else:
        global_label_majority = "ambiguous"
    
    return analyzed_results, global_label_weighted, global_prob_weighted * 100, global_label_majority, {"ai": vote_ai, "human": vote_human}

# Nuova funzione per estrarre il testo AI dall'input
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

def create_gui(conversations, vectorizer, clf):
    def on_classify():
        user_input = input_textbox.get("1.0", "end-1c")
        if not user_input.strip():
            print_to_console("Errore: Inserisci del testo da analizzare.")
            return

        # Estrai la parte AI dall'input per valutare solo la risposta
        ai_text = extract_ai_text(user_input)
        
        results, global_label_weighted, global_prob_weighted, global_label_majority, majority_counts = classify_text_extended(ai_text, vectorizer, clf)
        
        output_textbox.delete("1.0", "end")
        
        # Sezione: Analisi per singola frase
        output_textbox.insert("end", "=== Analisi per Frase ===\n\n")
        if results:
            for sent, label, ai_prob, weight in results:
                output_textbox.insert("end", f"Frase: {sent}\n")
                output_textbox.insert("end", f"  -> Classificazione: {label.upper()}\n")
                output_textbox.insert("end", f"  -> Percentuale AI: {ai_prob:.2f}% | Peso: {weight}\n\n")
        else:
            output_textbox.insert("end", "Nessuna frase analizzata (forse tutte troppo brevi?)\n")
        
        # Sezione: Risultato globale
        if global_label_weighted == "ai":
            overall_confidence = global_prob_weighted
            confidence_message = f"{overall_confidence:.2f}% (AI)"
        else:
            overall_confidence = 100 - global_prob_weighted
            confidence_message = f"{overall_confidence:.2f}% (Human)"
        
        output_textbox.insert("end", "=== Risultato Globale ===\n")
        output_textbox.insert("end", f"Media Ponderata: {global_label_weighted.upper()} (Confidence: {confidence_message})\n")
        output_textbox.insert("end", f"Votazione Maggioritaria: {global_label_majority.upper()} (AI: {majority_counts['ai']}, Human: {majority_counts['human']})\n")
        print_to_console("Analisi completata.")
        
    def print_to_console(message):
        console_textbox.config(state=tk.NORMAL)
        console_textbox.insert(tk.END, message + "\n")
        console_textbox.yview(tk.END)
        console_textbox.config(state=tk.DISABLED)
        
    root = tk.Tk()
    root.title("Analizzatore di Conversazioni AI")
    
    instructions = tk.Label(root,
        text="Istruzioni:\n1. Inserisci il testo da analizzare.\n2. Premi 'Classifica'.\n3. Verranno mostrate l'analisi per singola frase e l'aggregazione globale (media pesata e voto maggioritario).",
        font=("Arial", 12), justify="left")
    instructions.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    
    input_label = tk.Label(root, text="Testo di input:")
    input_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    input_textbox = scrolledtext.ScrolledText(root, width=60, height=10)
    input_textbox.grid(row=2, column=0, padx=10, pady=5)
    
    classify_button = tk.Button(root, text="Classifica", command=on_classify, font=("Arial", 12))
    classify_button.grid(row=3, column=0, padx=10, pady=10)
    
    output_label = tk.Label(root, text="Risultati della classificazione:")
    output_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
    output_textbox = scrolledtext.ScrolledText(root, width=60, height=10)
    output_textbox.grid(row=5, column=0, padx=10, pady=5)
    
    console_label = tk.Label(root, text="Log dell'elaborazione:")
    console_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
    console_textbox = scrolledtext.ScrolledText(root, width=60, height=10, state=tk.DISABLED)
    console_textbox.grid(row=7, column=0, padx=10, pady=5)
    
    print_to_console(f"Elementi caricati: {len(conversations)}")
    texts, _ = prepare_dataset(conversations)
    print_to_console(f"Dataset preparato: {len(texts)} frasi totali.")
    
    root.mainloop()
def save_model_with_threshold(model, vectorizer, optimal_threshold, filename="model_with_threshold.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer, "threshold": optimal_threshold}, f)
    print("Modello e soglia ottimale salvati in", filename)

# Carica il modello insieme alla soglia ottimale.
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
    
    # Prepara il dataset
    texts, labels = prepare_dataset(valid_conversations)
    
    # Suddivide il dataset in training e validation set (80/20)
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Carica il modello addestrato se esiste, altrimenti esegue l'ottimizzazione
    model = load_checkpoint()
    if model is not None:
        vectorizer = model.named_steps['vectorizer']
        clf = model.named_steps['classifier']
    else:
        model = optimize_hyperparameters(texts_train, labels_train)
        vectorizer = model.named_steps['vectorizer']
        clf = model.named_steps['classifier']
    
    # Ottimizzazione della soglia sul validation set:
    X_val = vectorizer.transform(texts_val)
    probabilities = clf.predict_proba(X_val)
    ai_index = list(clf.classes_).index("ai")
    ai_probs = probabilities[:, ai_index]
    
    precision, recall, thresholds = precision_recall_curve(labels_val, ai_probs, pos_label="ai")
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    best_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_index]
    
    print(f"Soglia ottimale trovata: {optimal_threshold:.2f}")
    
    # (Opzionale) Valutazione sulle predizioni del validation set con la soglia ottimale
    def classify_with_threshold(probs, threshold):
        return ["ai" if prob >= threshold else "human" for prob in probs]
    y_val_pred = classify_with_threshold(ai_probs, optimal_threshold)
    print(classification_report(labels_val, y_val_pred))
    
    # Salva il modello insieme alla soglia ottimale, se desiderato
    save_model_with_threshold(model, vectorizer, optimal_threshold, filename="model_with_threshold.pkl")
    
    # Avvia la GUI per l'inferenza
    create_gui(valid_conversations, vectorizer, clf)

if __name__ == "__main__":
    main()
