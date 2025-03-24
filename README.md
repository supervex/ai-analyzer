# AI Analyzer  

AI Analyzer è un'applicazione in Python progettata per analizzare conversazioni tra esseri umani e assistenti intelligenti. Grazie all'uso del machine learning, il programma è in grado di identificare se una risposta è stata generata da un'intelligenza artificiale, contribuendo alla comprensione dei pattern tipici delle risposte AI e supportando la rilevazione di contenuti generati automaticamente.  

## 🚀 Funzionalità  

- **🔍 Riconoscimento delle risposte AI**: Analizza le risposte nelle conversazioni e determina se provengono da un assistente AI o da un essere umano.  
- **🤖 Machine Learning**: Utilizza modelli di apprendimento automatico pre-addestrati per effettuare la classificazione.  
- **🖥️ Interfaccia Grafica Intuitiva**: Include una GUI user-friendly basata su Tkinter per caricare e analizzare i file di conversazione.  

## ⚙️ Come funziona  

1. 📂 L'utente carica un file contenente un file di testo.  
2. 🔢 Il programma elabora i dati e li vettorizza per renderli compatibili con il modello di machine learning.  
3. 🏷️ Il classificatore determina se ciascuna frase è stata generata da un'AI o da un essere umano.  
4. 📊 I risultati vengono mostrati all'utente tramite l'interfaccia grafica.  

## 🧠 Modelli Utilizzati  

L'analisi si basa su modelli pre-addestrati, tra cui:  

- **Vectorizer.pkl**: Trasforma il testo in una rappresentazione numerica.  
- **Classifier_model.pkl**: Modello di classificazione che identifica le risposte AI.  

Puoi scaricare questi modelli dalla repository ufficiale del progetto per utilizzarli con l'applicazione.  

## 📋 Prerequisiti  

Per eseguire AI Analyzer, assicurati di avere installato:  

- **Python 3.7+**  
- **Librerie richieste** (puoi installarle con il seguente comando):  

  ```bash
  pip install -r requirements.txt

