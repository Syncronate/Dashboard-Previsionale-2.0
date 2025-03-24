import requests
import json
import time
from datetime import datetime, timedelta
import sys
import urllib3
import gspread
from google.oauth2 import service_account  # Importa la libreria corretta
import os # Importa 'os' per accedere alle variabili d'ambiente

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def estrai_dati_meteo():
    """
    Extracts weather data from API and appends it to a Google Sheet.
    This function runs continuously, updating the sheet every hour.
    """
    # Google Sheets setup
    nome_foglio = "Dati Meteo Stazioni"
    # credenziali_path = "credentials.json"  # NON USIAMO PIÙ IL FILE LOCALE
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    try:
        # Carica le credenziali DA VARIABILE D'AMBIENTE
        credentials_json = os.environ.get("GCP_CREDENTIALS_JSON")
        if not credentials_json:
            raise EnvironmentError("Variabile d'ambiente GCP_CREDENTIALS_JSON non trovata.")
        credentials_info = json.loads(credentials_json)
        scoped_credentials = service_account.Credentials.from_service_account_info( # Usa google.oauth2.service_account.Credentials
            credentials_info, scopes=scope
        )
        client = gspread.authorize(scoped_credentials)


        try:
            foglio = client.open(nome_foglio).sheet1
            print(f"Foglio Google '{nome_foglio}' aperto con successo.")
        except gspread.SpreadsheetNotFound:
            foglio = client.create(nome_foglio).sheet1
            print(f"Foglio Google '{nome_foglio}' creato con successo.")
    except Exception as e:
        print(f"Errore nell'autenticazione a Google Sheets: {e}")
        sys.exit(1)

    # ... (resto del tuo script - API information, estrazione dati, aggiornamento foglio, ecc. - RIMANE INVARIATO) ...

if __name__ == "__main__":
    estrai_dati_meteo()
