import re

def clean_text(text):
    """
    Rimuove la punteggiatura (tranne gli apostrofi) e i numeri dal testo.
    Mantiene le lettere accentate e gli apostrofi, converte il testo in minuscolo 
    e rimuove gli spazi extra.
    """
    # La regex permette: lettere (inclusi caratteri accentati), spazi e apostrofi.
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\s]", '', text)
    text = text.lower()
    text = " ".join(text.split())
    return text

def analyze_length(text):
    """
    Restituisce il numero di parole nel testo.
    """
    return len(text.split())

