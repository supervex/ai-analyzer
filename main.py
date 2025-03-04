import joblib
import tkinter as tk
from tkinter import scrolledtext
from data_loader import load_json_files
from text_processing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

# Assicurati di avere il pacchetto 'punkt' per il tokenizing
nltk.download('punkt')

def extract_conversation(entry):
    """
    Estrae le informazioni della conversazione da un elemento del dataset.
    Gestisce sia la versione inglese (con [|Human|] e [|AI|])
    che quella italiana (con [|Umano|] e [|AI|]).
    """
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
    """
    Crea un dataset etichettato partendo dalle conversazioni estratte.
    """
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

def train_classifier(texts, labels):
    """
    Vettorizza il testo e addestra un classificatore (Multinomial Naive Bayes).
    Salva il modello e il vettorizzatore **alla fine** dell'addestramento.
    """
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy del classificatore: {accuracy:.2f}")
    
    # Salva il modello e il vettorizzatore alla fine dell'addestramento
    joblib.dump(clf, 'classifier_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Modello salvato alla fine dell'addestramento.")
    
    return vectorizer, clf

def load_trained_model():
    """
    Carica il modello e il vettorizzatore salvati se esistono.
    """
    try:
        clf = joblib.load('classifier_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("Modello caricato con successo.")
        return vectorizer, clf
    except FileNotFoundError:
        print("Modello non trovato. Procedo con l'addestramento...")
        return None, None

def classify_text(text, vectorizer, clf):
    """
    Classifica il testo in frasi e calcola la percentuale di probabilità AI.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    sentences_vec = vectorizer.transform(sentences)
    preds = clf.predict(sentences_vec)
    probas = clf.predict_proba(sentences_vec)
    
    ai_index = list(clf.classes_).index("ai")
    results = []
    for i, sentence in enumerate(sentences):
        ai_prob = probas[i, ai_index] * 100  # Percentuale
        results.append((sentence, preds[i], ai_prob))
    return results

def create_gui(conversations, vectorizer, clf):
    """
    Crea l'interfaccia grafica per l'interazione con l'utente.
    """
    def on_classify():
        # Ottieni il testo dall'input dell'utente
        input_text = input_textbox.get("1.0", "end-1c")
        
        # Classifica il testo
        results = classify_text(input_text, vectorizer, clf)
        
        # Pulisce la finestra di output e mostra i risultati
        output_textbox.delete(1.0, "end")
        for sent, label, ai_prob in results:
            output_textbox.insert("end", f"Frase: {sent}\nClassificazione: {label}, Percentuale AI: {ai_prob:.2f}%\n\n")
    
    # Crea la finestra principale
    root = tk.Tk()
    root.title("Analizzatore di Conversazioni AI")

    # Crea i widget
    input_label = tk.Label(root, text="Inserisci il testo da analizzare:")
    input_textbox = scrolledtext.ScrolledText(root, width=50, height=10)
    classify_button = tk.Button(root, text="Classifica", command=on_classify)
    output_label = tk.Label(root, text="Risultati della classificazione:")
    output_textbox = scrolledtext.ScrolledText(root, width=50, height=10)

    # Posiziona i widget nella finestra
    input_label.grid(row=0, column=0, padx=10, pady=10)
    input_textbox.grid(row=1, column=0, padx=10, pady=10)
    classify_button.grid(row=2, column=0, padx=10, pady=10)
    output_label.grid(row=3, column=0, padx=10, pady=10)
    output_textbox.grid(row=4, column=0, padx=10, pady=10)

    # Avvia l'interfaccia grafica
    root.mainloop()

def main():
    # Carica i dati dal dataset
    data_folder = r"C:\Users\Alberto\Desktop\AiAnalyzer\data"
    data = load_json_files(data_folder)
    print(f"Elementi caricati: {len(data)}")
    
    # Estrai le conversazioni dal dataset
    conversations = [extract_conversation(entry) for entry in data]
    
    # Filtra eventuali conversazioni incomplete (facoltativo)
    valid_conversations = []
    for convo in conversations:
        if convo["human_request"] or convo["ai_response"]:
            valid_conversations.append(convo)
        else:
            print(f"Attenzione: conversazione incompleta: {convo}")
    
    # Prepara il dataset etichettato per l'addestramento
    texts, labels = prepare_dataset(valid_conversations)
    
    # Carica il modello salvato se esiste, altrimenti addestra un nuovo modello
    vectorizer, clf = load_trained_model()
    if not vectorizer or not clf:
        vectorizer, clf = train_classifier(texts, labels)
    
    # Crea la GUI per l'interazione con l'utente
    create_gui(valid_conversations, vectorizer, clf)

if __name__ == "__main__":
    main()
