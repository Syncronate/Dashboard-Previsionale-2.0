# estrai_allerte.py (CORRETTO)

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
NOME_WORKSHEET_ALLERTE = os.environ.get("WORKSHEET_NAME", "Allerte")
NOME_FILE_CREDENZIALI = "credentials.json"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
API_URL_OGGI = "https://allertameteo.regione.marche.it/o/api/allerta/get-stato-allerta"
API_URL_DOMANI = "https://allertameteo.regione.marche.it/o/api/allerta/get-stato-allerta-domani"
AREE_INTERESSATE = ["2", "4"]

# --- FINE CONFIGURAZIONE ---

def autentica_google_sheets():
    """Gestisce l'autenticazione a Google Sheets usando il metodo moderno."""
    credenziali_path = os.path.join(os.path.dirname(__file__), NOME_FILE_CREDENZIALI)
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

def apri_o_crea_worksheet(client, nome_foglio, nome_worksheet):
    """Apre un foglio Google e un worksheet. Se il worksheet non esiste, lo crea."""
    try:
        spreadsheet = client.open(nome_foglio)
        print(f"Foglio Google '{nome_foglio}' aperto.")
        try:
            worksheet = spreadsheet.worksheet(nome_worksheet)
            print(f"Worksheet '{nome_worksheet}' trovato.")
        except gspread.WorksheetNotFound:
            print(f"Worksheet '{nome_worksheet}' non trovato. Verrà creato.")
            worksheet = spreadsheet.add_worksheet(title=nome_worksheet, rows="100", cols="20")
        return worksheet
    except gspread.SpreadsheetNotFound:
        print(f"ERRORE: Foglio Google '{nome_foglio}' non trovato.")
        sys.exit(1)
    except Exception as e:
        print(f"Errore durante l'apertura/creazione: {e}")
        sys.exit(1)

# --- MODIFICA QUI ---
def processa_risposta_api(dati_api, giorno_riferimento):
    """Estrae e formatta i dati di allerta da una risposta API."""
    dati_processati, tipi_evento = [], set()
    if not dati_api:
        return dati_processati, tipi_evento

    for area_data in dati_api:
        if area_data.get("area") in AREE_INTERESSATE:
            try:
                # Logica riscritta con un ciclo for standard per maggiore robustezza
                eventi_dict = {}
                eventi_str = area_data.get("eventi", "")
                if eventi_str:  # Processa solo se la stringa non è vuota
                    for pair in eventi_str.split(','):
                        # Assicura che la coppia sia valida e contenga un ':' prima di dividerla
                        if ':' in pair:
                            key, value = pair.split(':', 1)  # Dividi solo al primo ':'
                            eventi_dict[key.strip()] = value.strip()
                
                tipi_evento.update(eventi_dict.keys())
                dati_processati.append({"giorno": giorno_riferimento, "area": area_data["area"], "eventi": eventi_dict})
            
            except Exception as e:
                # Manteniamo il blocco try/except per ogni evenienza
                print(f"WARN: Impossibile processare eventi per area {area_data.get('area')}: {e}")
    
    return dati_processati, tipi_evento
# --- FINE MODIFICA ---

def estrai_dati_allerta():
    """Funzione principale che orchestra il processo."""
    start_time = datetime.now(pytz.timezone('Europe/Rome'))
    print(f"--- Inizio Script Allerte Meteo ({start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    client = autentica_google_sheets()
    sheet = apri_o_crea_worksheet(client, NOME_FOGLIO_PRINCIPALE, NOME_WORKSHEET_ALLERTE)

    all_processed_data, all_event_types = [], set()
    for giorno, url in [("Oggi", API_URL_OGGI), ("Domani", API_URL_DOMANI)]:
        print(f"\nRecupero dati per: {giorno}")
        try:
            response = requests.get(url, verify=False, timeout=20)
            response.raise_for_status()
            dati_giorno, tipi_evento_giorno = processa_risposta_api(response.json(), giorno)
            all_processed_data.extend(dati_giorno)
            all_event_types.update(tipi_evento_giorno)
            print(f"Dati per '{giorno}' processati. Trovati {len(dati_giorno)} record pertinenti.")
        except Exception as e:
            print(f"ERRORE API per '{giorno}': {e}")

    if not all_processed_data:
        print("\nNessun dato di allerta valido trovato. Uscita.")
        sys.exit(0)
    
    header_finale = ['Data_Esecuzione', 'Giorno_Riferimento', 'Area'] + sorted(list(all_event_types))
    try:
        existing_header = sheet.row_values(1)
        if not existing_header:
            print("\nWorksheet vuoto. Scrittura nuova intestazione...")
            sheet.update('A1', [header_finale], value_input_option='USER_ENTERED')
        elif set(existing_header) != set(header_finale):
            print("\nATTENZIONE: Intestazione nel foglio diversa da quella generata. Uso l'intestazione esistente.")
            header_finale = existing_header
    except Exception as e:
        print(f"Errore gestione intestazione: {e}")
        
    righe_da_scrivere = []
    formatted_time = start_time.strftime('%d/%m/%Y %H:%M')
    for data_point in all_processed_data:
        riga = [formatted_time, data_point['giorno'], data_point['area']]
        riga.extend(data_point['eventi'].get(col_name, 'N/D') for col_name in header_finale[3:])
        righe_da_scrivere.append(riga)

    if righe_da_scrivere:
        print(f"\nPronte {len(righe_da_scrivere)} righe da aggiungere al foglio.")
        try:
            sheet.append_rows(righe_da_scrivere, value_input_option='USER_ENTERED')
            print(f"\nDati aggiunti con successo al worksheet '{NOME_WORKSHEET_ALLERTE}'.")
        except Exception as e:
            print(f"ERRORE CRITICO durante la scrittura su Google Sheets: {e}")
            traceback.print_exc()
            sys.exit(1)

    end_time = datetime.now(pytz.timezone('Europe/Rome'))
    print(f"\n--- Fine Script Allerte Meteo ({end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

if __name__ == "__main__":
    estrai_dati_allerta()
