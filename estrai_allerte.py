# estrai_allerte.py

import requests
import json
from datetime import datetime
import sys
import os
import traceback
import urllib3
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pytz

# --- CONFIGURAZIONE ---
# Disabilita warning SSL (se necessario, l'API non richiede certificati specifici)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Nomi per Google Sheets
NOME_FOGLIO_PRINCIPALE = "Dati Meteo Stazioni"
NOME_WORKSHEET_ALLERTE = "Allerte"
NOME_FILE_CREDENZIALI = "credentials.json"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# API e filtri
API_URL_OGGI = "https://allertameteo.regione.marche.it/o/api/allerta/get-stato-allerta"
API_URL_DOMANI = "https://allertameteo.regione.marche.it/o/api/allerta/get-stato-allerta-domani"
AREE_INTERESSATE = ["2", "4"] # Interessa solo alle aree 2 e 4

# --- FINE CONFIGURAZIONE ---

def autentica_google_sheets():
    """Gestisce l'autenticazione a Google Sheets e restituisce il client."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    credenziali_path = os.path.join(script_dir, NOME_FILE_CREDENZIALI)
    print("Autenticazione Google Sheets...")
    try:
        credenziali = ServiceAccountCredentials.from_json_keyfile_name(credenziali_path, SCOPE)
        client = gspread.authorize(credenziali)
        print("Autenticazione riuscita.")
        return client
    except Exception as e:
        print(f"ERRORE CRITICO nell'autenticazione: {e}")
        print(f"Verifica che il file '{NOME_FILE_CREDENZIALI}' sia presente in '{script_dir}' e valido.")
        sys.exit(1)

def apri_o_crea_worksheet(client, nome_foglio, nome_worksheet):
    """Apre un foglio Google e un worksheet specifico. Se il worksheet non esiste, lo crea."""
    try:
        spreadsheet = client.open(nome_foglio)
        print(f"Foglio Google '{nome_foglio}' aperto con successo.")
        try:
            worksheet = spreadsheet.worksheet(nome_worksheet)
            print(f"Worksheet '{nome_worksheet}' trovato.")
        except gspread.WorksheetNotFound:
            print(f"Worksheet '{nome_worksheet}' non trovato. Verrà creato.")
            worksheet = spreadsheet.add_worksheet(title=nome_worksheet, rows="100", cols="20")
            print(f"Worksheet '{nome_worksheet}' creato.")
        return worksheet
    except gspread.SpreadsheetNotFound:
        print(f"ERRORE: Il foglio Google '{nome_foglio}' non esiste.")
        print("Creane uno manualmente e condividilo con l'email del service account.")
        sys.exit(1)
    except Exception as e:
        print(f"Errore durante l'apertura/creazione del foglio/worksheet: {e}")
        sys.exit(1)

def processa_risposta_api(dati_api, giorno_riferimento):
    """
    Estrae e formatta i dati di allerta da una risposta API per le aree di interesse.
    Restituisce una lista di dizionari con i dati e un set con tutti i tipi di evento.
    """
    dati_processati = []
    tipi_evento = set()
    
    if not dati_api:
        print(f"Attenzione: la risposta API per '{giorno_riferimento}' è vuota.")
        return dati_processati, tipi_evento

    for area_data in dati_api:
        area = area_data.get("area")
        if area in AREE_INTERESSATE:
            eventi_str = area_data.get("eventi", "")
            eventi_dict = {}
            try:
                # Trasforma la stringa "key1:val1,key2:val2" in un dizionario
                eventi_dict = {
                    key_val.split(':')[0].strip(): key_val.split(':')[1].strip()
                    for key_val in eventi_str.split(',') if ':' in key_val
                }
                tipi_evento.update(eventi_dict.keys())
                dati_processati.append({
                    "giorno": giorno_riferimento,
                    "area": area,
                    "eventi": eventi_dict
                })
            except Exception as e:
                print(f"WARN: Impossibile processare stringa eventi '{eventi_str}' per area {area}. Errore: {e}")

    return dati_processati, tipi_evento


def estrai_dati_allerta():
    """Funzione principale che orchestra il processo."""
    italian_tz = pytz.timezone('Europe/Rome')
    start_time = datetime.now(italian_tz)
    print(f"--- Inizio Script Allerte Meteo ({start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")

    # 1. Autenticazione e apertura worksheet
    client = autentica_google_sheets()
    sheet = apri_o_crea_worksheet(client, NOME_FOGLIO_PRINCIPALE, NOME_WORKSHEET_ALLERTE)

    # 2. Chiamate API
    all_processed_data = []
    all_event_types = set()

    for giorno, url in [("Oggi", API_URL_OGGI), ("Domani", API_URL_DOMANI)]:
        print(f"\nRecupero dati allerta per: {giorno}")
        try:
            response = requests.get(url, verify=False, timeout=20)
            response.raise_for_status()
            dati_json = response.json()
            
            dati_giorno, tipi_evento_giorno = processa_risposta_api(dati_json, giorno)
            all_processed_data.extend(dati_giorno)
            all_event_types.update(tipi_evento_giorno)
            print(f"Dati per '{giorno}' processati con successo. Trovati {len(dati_giorno)} record pertinenti.")

        except requests.exceptions.RequestException as e:
            print(f"ERRORE nella richiesta API per '{giorno}': {e}")
        except json.JSONDecodeError:
            print(f"ERRORE: La risposta API per '{giorno}' non è un JSON valido.")

    if not all_processed_data:
        print("\nNessun dato di allerta valido trovato per le aree di interesse. Uscita.")
        sys.exit(0)
    
    # 3. Gestione Intestazione
    header_finale = ['Data_Esecuzione', 'Giorno_Riferimento', 'Area'] + sorted(list(all_event_types))
    
    try:
        # Controlla se il worksheet è vuoto per scrivere l'intestazione
        existing_header = sheet.row_values(1)
        if not existing_header:
            print("\nWorksheet vuoto. Scrittura nuova intestazione...")
            sheet.update('A1', [header_finale], value_input_option='USER_ENTERED')
            print(f"Intestazione scritta: {header_finale}")
        elif set(existing_header) != set(header_finale):
            # Opzionale: gestire il disallineamento dell'intestazione. Per ora, logghiamo un warning.
            print("\nATTENZIONE: L'intestazione nel foglio è diversa da quella generata dai dati correnti.")
            print(f"  Intestazione foglio: {existing_header}")
            print(f"  Intestazione attesa: {header_finale}")
            print("  Lo script proverà a scrivere i dati basandosi sull'intestazione esistente.")
            header_finale = existing_header # Usa l'intestazione esistente per coerenza

    except Exception as e:
        print(f"Errore durante la gestione dell'intestazione: {e}")
        # Non usciamo, proviamo comunque a scrivere
        
    # 4. Preparazione e scrittura righe
    righe_da_scrivere = []
    formatted_time = start_time.strftime('%d/%m/%Y %H:%M')

    for data_point in all_processed_data:
        riga = []
        for col_name in header_finale:
            if col_name == 'Data_Esecuzione':
                riga.append(formatted_time)
            elif col_name == 'Giorno_Riferimento':
                riga.append(data_point['giorno'])
            elif col_name == 'Area':
                riga.append(data_point['area'])
            else:
                # Aggiunge il valore dell'evento (es. 'white', 'green') o 'N/D' se non presente
                riga.append(data_point['eventi'].get(col_name, 'N/D'))
        righe_da_scrivere.append(riga)

    if righe_da_scrivere:
        print(f"\nPronte {len(righe_da_scrivere)} righe da aggiungere al foglio:")
        for r in righe_da_scrivere:
            print(f"  - {r}")
        try:
            sheet.append_rows(righe_da_scrivere, value_input_option='USER_ENTERED')
            print(f"\nDati aggiunti con successo al worksheet '{NOME_WORKSHEET_ALLERTE}'.")
        except Exception as e:
            print(f"ERRORE CRITICO durante la scrittura su Google Sheets: {e}")
            traceback.print_exc()
            sys.exit(1)

    end_time = datetime.now(italian_tz)
    print(f"\n--- Fine Script Allerte Meteo ({end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")


if __name__ == "__main__":
    estrai_dati_allerta()