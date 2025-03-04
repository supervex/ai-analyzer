# AI Analyzer

AI Analyzer è un'applicazione sviluppata in Python che analizza conversazioni tra esseri umani e assistenti intelligenti (AI). Utilizza un modello di machine learning per identificare se una risposta proviene da un'intelligenza artificiale, migliorando così la comprensione dei pattern nelle risposte AI e aiutando nella rilevazione di contenuti generati automaticamente.

## Funzionalità

- **Analisi delle risposte AI**: Il programma esamina le risposte generate dall'intelligenza artificiale e confronta con le domande poste dagli esseri umani.
- **Modelli di machine learning**: Utilizza modelli pre-addestrati per rilevare risposte AI, come classificatori basati su algoritmi di machine learning.
- **Interfaccia grafica**: Ha una GUI semplice che permette di caricare e analizzare conversazioni.

## Come funziona

L'applicazione carica un file di conversazione che contiene le domande degli utenti e le risposte generate dall'intelligenza artificiale. Successivamente, il programma analizza i dati utilizzando un modello di machine learning per determinare se una risposta è generata da un AI o se è un'istanza umana.

### Modelli

- I modelli pre-addestrati, come `vectorizer.pkl` e `classifier_model.pkl`, sono utilizzati per eseguire l'analisi. Puoi scaricare questi modelli dalla repository e utilizzarli con il programma.

## Prerequisiti

Prima di eseguire l'applicazione, assicurati di avere installato:

- Python 3.7+
- Le seguenti librerie:
  - `scikit-learn`
  - `numpy`
  - `tkinter`
  - `joblib`
  - `git-lfs` (per i file di grandi dimensioni come i modelli)

Puoi installare le dipendenze usando il file `requirements.txt`:

```bash
pip install -r requirements.txt
