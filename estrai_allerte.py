import requests
import json
from datetime import datetime
import sys
import os
import traceback
import urllib3
import gspread
from google.oauth2.service_account import Credentials
import pytz

# --- CONFIGURAZIONE ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NOME_FOGLIO_PRINCIPALE = os.environ.get("GSHEET_NAME", "Dati Meteo Stazioni")
NOME_WORKSHEET_OGGI = "Allerte Oggi"
NOME_WORKSHEET_DOMANI = "Allerte Domani"

NOME_FILE_CREDENZIALI = "credentials.json"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
API_URL_OGGI = "https://allertameteo.regione.marche.it/o/api/allerta/get-stato-allerta"
API_URL_DOMANI = "https://allertameteo.regione.marche.it/o/api/allerta/get-stato-allerta-domani"

AREE_INTERESSATE = ["2", "4"]
AREA_COMBINATA_NOME = "2-4"

# --- MODIFICATO: Dizionario per la traduzione dei livelli di allerta ---
TRADUZIONE_ALLERTE = {
    "Red": "Rossa",
    "Orange": "Arancione",
    "Yellow": "Gialla",
    "Green": "Verde",
    "white": "Nessuna",  # Aggiunto: 'white' significa assenza di fenomeno
    "N/D": "Nessuna"     # Valore di default interno
}

# --- MODIFICATO: Dizionario per definire la gravità di un'allerta ---
LIVELLI_SEVERITA = {
    "white": 1,      # Aggiunto: Severità minima, ma riconosciuta
    "Green": 1,      # Severità minima (vigilanza)
    "Yellow": 2,
    "Orange": 3,
    "Red": 4,
    "N/D": 0         # Severità zero solo per i dati non trovati
}
# --- FINE CONFIGURAZIONE ---


def autentica_google_sheets():
    """Gestisce l'autenticazione a Google Sheets."""
    credenziali_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOME_FILE_CREDENZIALI)
    print("Autenticazione Google Sheets...")
    try:
        creds = Credentials.from_service_account_file(credenziali_path, scopes=SCOPE)
        client = gspread.authorize(creds)
        print("Autenticazione riuscita.")
        return client
    except FileNotFoundError:
        print(f"ERRORE CRITICO: File credenziali '{NOME_FILE_CREDENZIALI}' non trovato.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRORE CRITICO nell'autenticazione: {e}")
        sys.exit(1)


def apri_o_crea_worksheet(spreadsheet, nome_worksheet):
    """Apre un worksheet. Se non esiste, lo crea."""
    try:
        worksheet = spreadsheet.worksheet(nome_worksheet)
        print(f"Worksheet '{nome_worksheet}' trovato.")
    except gspread.WorksheetNotFound:
        print(f"Worksheet '{nome_worksheet}' non trovato. Verrà creato.")
        worksheet = spreadsheet.add_worksheet(title=nome_worksheet, rows="100", cols="20")
    return worksheet


def processa_e_combina_allerte(dati_api):
    """
    Estrae i dati, li filtra per le aree di interesse e restituisce un singolo
    dizionario con l'allerta più grave per ogni tipo di evento.
    """
    allerte_combinate = {}
    tipi_evento_riscontrati = set()

    if not dati_api:
        return allerte_combinate, tipi_evento_riscontrati

    dati_filtrati = [d for d in dati_api if d.get("area") in AREE_INTERESSATE]

    for area_data in dati_filtrati:
        eventi_str = area_data.get("eventi", "")
        if not eventi_str:
            continue

        eventi_dict = {}
        try:
            for pair in eventi_str.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    eventi_dict[key.strip()] = value.strip()
        except Exception as e:
            print(f"WARN: Impossibile processare eventi per area {area_data.get('area')}: {e}")
            continue
        
        for tipo_evento, livello_attuale in eventi_dict.items():
            tipi_evento_riscontrati.add(tipo_evento)
            livello_precedente = allerte_combinate.get(tipo_evento, "N/D")

            if LIVELLI_SEVERITA.get(livello_attuale, 0) > LIVELLI_SEVERITA.get(livello_precedente, 0):
                allerte_combinate[tipo_evento] = livello_attuale
    
    return allerte_combinate, tipi_evento_riscontrati


def scrivi_su_foglio(worksheet, header, dati_allerte, timestamp):
    """Svuota il foglio e scrive l'intestazione e una singola riga di dati."""
    try:
        print(f"\nReset del foglio '{worksheet.title}'...")
        worksheet.clear()
        
        riga_dati = [timestamp, AREA_COMBINATA_NOME]
        for tipo_evento in header[2:]:
            livello_inglese = dati_allerte.get(tipo_evento, "N/D")
            livello_italiano = TRADUZIONE_ALLERTE.get(livello_inglese, livello_inglese)
            riga_dati.append(livello_italiano)

        print(f"Scrittura dati su '{worksheet.title}'...")
        worksheet.update('A1', [header, riga_dati], value_input_option='USER_ENTERED')
        print("Scrittura completata con successo.")

    except Exception as e:
        print(f"ERRORE CRITICO durante la scrittura su '{worksheet.title}': {e}")
        traceback.print_exc()


def estrai_dati_allerta():
    """Funzione principale che orchestra il processo."""
    start_time = datetime.now(pytz.timezone('Europe/Rome'))
    print(f"--- Inizio Script Allerte Meteo ({start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    client = autentica_google_sheets()
    try:
        spreadsheet = client.open(NOME_FOGLIO_PRINCIPALE)
        print(f"Foglio Google '{NOME_FOGLIO_PRINCIPALE}' aperto.")
    except gspread.SpreadsheetNotFound:
        print(f"ERRORE: Foglio Google '{NOME_FOGLIO_PRINCIPALE}' non trovato.")
        sys.exit(1)

    sheet_oggi = apri_o_crea_worksheet(spreadsheet, NOME_WORKSHEET_OGGI)
    sheet_domani = apri_o_crea_worksheet(spreadsheet, NOME_WORKSHEET_DOMANI)

    dati_per_giorno = {}
    tutti_i_tipi_evento = set()

    for giorno, url, worksheet in [("Oggi", API_URL_OGGI, sheet_oggi), ("Domani", API_URL_DOMANI, sheet_domani)]:
        print(f"\nRecupero dati per: {giorno}")
        try:
            response = requests.get(url, verify=False, timeout=20)
            response.raise_for_status()
            dati_combinati, tipi_evento = processa_e_combina_allerte(response.json())
            dati_per_giorno[giorno] = dati_combinati
            tutti_i_tipi_evento.update(tipi_evento)
            print(f"Dati combinati per '{giorno}' processati.")
        except Exception as e:
            print(f"ERRORE API per '{giorno}': {e}")
            dati_per_giorno[giorno] = {}

    if not tutti_i_tipi_evento:
        print("\nNessun tipo di evento trovato nelle allerte. Uscita.")
        sys.exit(0)
    
    header_finale = ['Data_Esecuzione', 'Area_Combinata'] + sorted(list(tutti_i_tipi_evento))
    formatted_time = start_time.strftime('%d/%m/%Y %H:%M')

    if NOME_WORKSHEET_OGGI in sheet_oggi.title:
        scrivi_su_foglio(sheet_oggi, header_finale, dati_per_giorno.get("Oggi", {}), formatted_time)

    if NOME_WORKSHEET_DOMANI in sheet_domani.title:
        scrivi_su_foglio(sheet_domani, header_finale, dati_per_giorno.get("Domani", {}), formatted_time)

    end_time = datetime.now(pytz.timezone('Europe/Rome'))
    print(f"\n--- Fine Script Allerte Meteo ({end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

if __name__ == "__main__":
    estrai_dati_allerta()
