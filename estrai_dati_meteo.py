import requests
import json
from datetime import datetime
import sys
import urllib3
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import traceback # Importato per logging errori più dettagliato

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- COSTANTI E CONFIGURAZIONE ---
NOME_FOGLIO = "Dati Meteo Stazioni"
NOME_FILE_CREDENZIALI = "credentials.json" # Assicurati sia nel posto giusto rispetto allo script
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
API_URL = "https://retemir.regione.marche.it/api/stations/rt-data"
STAZIONI_INTERESSATE = [
    "Misa", "Pianello di Ostra", "Nevola", "Barbara",
    "Serra dei Conti", "Arcevia", "Corinaldo", "Ponte Garibaldi",
]
# VERIFICA QUESTI TIPO_SENS! Qual è quello per la Pioggia Totale Giornaliera? (Es. 5?)
# Il tipoSens 0 sembra essere "Pioggia TOT Oggi" dal tuo esempio.
SENSORI_INTERESSATI_TIPOSENS = [0, 1, 5, 6, 9, 10, 100, 101]

# !!! IMPORTANTE: AGGIORNA QUESTA LISTA DOPO AVER CONTROLLATO L'OUTPUT API !!!
# Basato sul tuo esempio, "Pioggia TOT Oggi" è la descrizione corretta.
# Se altre stazioni usano descrizioni diverse, aggiungi le keyword qui.
DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS = ["Pioggia TOT Oggi"]

# Codice specifico per la stazione di Arcevia corretta
CODICE_ARCEVIA_CORRETTO = 732

# --- FINE CONFIGURAZIONE ---

def estrai_dati_meteo():
    """
    Estrae i dati meteo dall'API, filtra per stazione e codice specifico (per Arcevia),
    calcola la pioggia nell'intervallo di esecuzione (30 MINUTI) basandosi sull'ultimo dato
    nel Google Sheet e la scrive nella colonna "Pioggia Ora (mm)".
    Aggiunge la nuova riga di dati al foglio senza cancellare lo storico.
    """
    print(f"--- Inizio Script Dati Meteo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print("NOTA: Lo script calcola la pioggia nell'intervallo di esecuzione (30 min) e la scrive nella colonna 'Pioggia Ora (mm)'.") # Aggiunta nota chiarificatrice
    script_dir = os.path.dirname(os.path.abspath(__file__))
    credenziali_path = os.path.join(script_dir, NOME_FILE_CREDENZIALI)

    # --- Autenticazione Google Sheets ---
    print("Autenticazione Google Sheets...")
    try:
        credenziali = ServiceAccountCredentials.from_json_keyfile_name(credenziali_path, SCOPE)
        client = gspread.authorize(credenziali)
        try:
            sheet = client.open(NOME_FOGLIO).sheet1
            print(f"Foglio Google '{NOME_FOGLIO}' aperto con successo.")
        except gspread.SpreadsheetNotFound:
            print(f"Foglio Google '{NOME_FOGLIO}' non trovato. Verrà creato.")
            spreadsheet = client.create(NOME_FOGLIO)
            sheet = spreadsheet.sheet1
            print(f"Foglio Google '{NOME_FOGLIO}' creato con successo.")
        except Exception as e_open:
             print(f"Errore durante l'apertura del foglio '{NOME_FOGLIO}': {e_open}")
             sys.exit(1)

    except Exception as e_auth:
        print(f"Errore critico nell'autenticazione Google Sheets: {e_auth}")
        print(f"Verifica che il file '{NOME_FILE_CREDENZIALI}' esista in '{script_dir}' e sia valido.")
        sys.exit(1)

    # --- Lettura Intestazione e Ultimi Dati dal Foglio ---
    valori_precedenti_pioggia = {}
    intestazione_attuale_foglio = []
    last_row_data = []
    last_row_index = 0
    print("\nLettura intestazione e ultima riga dal foglio...")
    try:
        all_values = sheet.get_all_values()
        last_row_index = len(all_values)

        if last_row_index > 0:
             intestazione_attuale_foglio = all_values[0]
             print(f"Intestazione letta dal foglio (Riga 1): {intestazione_attuale_foglio}")
        else:
             print("Foglio vuoto. L'intestazione verrà creata.")

        if last_row_index > 1:
            last_row_data = all_values[-1]
            print(f"Ultima riga di dati letta (Indice logico {last_row_index}): {last_row_data}")

            header_map = {name: idx for idx, name in enumerate(intestazione_attuale_foglio)}

            for col_name, idx in header_map.items():
                is_pioggia_tot_col = any(keyword.lower() in col_name.lower() for keyword in DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS)

                # *** MODIFICA: Legge dalla colonna Pioggia TOT, ignorando la Pioggia Ora calcolata ***
                if is_pioggia_tot_col and "Pioggia Ora" not in col_name: # Ignora la colonna calcolata stessa
                    try:
                        station_name = col_name.split(" - ")[0].strip()
                    except IndexError:
                        print(f"WARN: Impossibile estrarre nome stazione da '{col_name}' nell'intestazione.")
                        continue

                    try:
                        if idx < len(last_row_data):
                            prev_value_str = last_row_data[idx]
                            if isinstance(prev_value_str, str) and prev_value_str.strip() != '' and prev_value_str.upper() != 'N/A':
                                prev_value_float = float(prev_value_str.replace(',', '.'))
                                valori_precedenti_pioggia[station_name] = prev_value_float
                                print(f"  -> Letto valore pioggia TOT precedente per '{station_name}' da colonna '{col_name}': {prev_value_float}")
                            elif isinstance(prev_value_str, (int, float)):
                                valori_precedenti_pioggia[station_name] = float(prev_value_str)
                                print(f"  -> Letto valore pioggia TOT precedente (numerico) per '{station_name}' da colonna '{col_name}': {valori_precedenti_pioggia[station_name]}")
                            else:
                                valori_precedenti_pioggia[station_name] = None
                                print(f"  -> Valore pioggia TOT precedente non valido/vuoto per '{station_name}' in colonna '{col_name}'.")
                        else:
                             valori_precedenti_pioggia[station_name] = None
                             print(f"  -> Indice colonna '{col_name}' ({idx}) fuori dai limiti per l'ultima riga ({len(last_row_data)} elementi).")

                    except (ValueError, TypeError) as e_parse:
                        print(f"  -> Errore conversione valore precedente per '{station_name}' ('{prev_value_str}'): {e_parse}. Impostato a None.")
                        valori_precedenti_pioggia[station_name] = None
        else:
             print("Nessuna riga di dati trovata nel foglio. Calcolo pioggia nell'intervallo (30 min) partirà dal valore attuale.")

    except gspread.exceptions.APIError as api_err:
         print(f"Errore API Google Sheets durante lettura dati: {api_err}")
    except Exception as e_read:
        print(f"Attenzione: Errore imprevisto durante lettura dal foglio. Calcolo pioggia potrebbe non funzionare. Errore: {e_read}")
        intestazione_attuale_foglio = []

    # --- Estrazione Dati API ---
    current_time = datetime.now()
    formatted_time = current_time.strftime('%d/%m/%Y %H:%M')
    print(f"\nEstrazione dati API alle {formatted_time}...")
    try:
        response = requests.get(API_URL, verify=False, timeout=30)
        response.raise_for_status()
        dati_meteo = response.json()
        print("Dati API ricevuti con successo.")

        if credenziali.access_token_expired:
            print("Token Google scaduto, rinnovo...")
            client.login()

        dati_api_per_riga = {}
        intestazioni_da_api = {}
        stazioni_processate_nomi = set()

        print("\nInizio processamento stazioni dall'API:")
        for stazione in dati_meteo:
            nome_stazione = stazione.get("nome")
            codice_stazione = stazione.get("codice")

            process_this_station = False
            if nome_stazione in STAZIONI_INTERESSATE:
                if nome_stazione in stazioni_processate_nomi:
                     print(f"INFO: Stazione '{nome_stazione}' (codice {codice_stazione}) già processata. Ignorando.")
                elif nome_stazione == "Arcevia":
                    if codice_stazione == CODICE_ARCEVIA_CORRETTO:
                        process_this_station = True
                        stazioni_processate_nomi.add(nome_stazione)
                    else:
                        print(f"INFO: Ignorando stazione '{nome_stazione}' con codice {codice_stazione} (si usa solo {CODICE_ARCEVIA_CORRETTO}).")
                else:
                    process_this_station = True
                    stazioni_processate_nomi.add(nome_stazione)

            if process_this_station:
                print(f"\nProcessando stazione: {nome_stazione} (Codice: {codice_stazione})")
                pioggia_tot_oggi_attuale = None

                for sensore in stazione.get("analog", []):
                    tipoSens = sensore.get("tipoSens")
                    if tipoSens in SENSORI_INTERESSATI_TIPOSENS:
                        descr_sensore = sensore.get("descr", "").strip()
                        valore_sensore = sensore.get("valore")
                        unita_misura = sensore.get("unmis", "").strip() if sensore.get("unmis") else ""

                        intestazione_sensore = f"{nome_stazione} - {descr_sensore} ({unita_misura})"
                        intestazioni_da_api[intestazione_sensore] = True

                        valore_convertito = 'N/A'
                        if valore_sensore is not None:
                            try:
                                 valore_convertito = float(str(valore_sensore).replace(',','.'))
                            except (ValueError, TypeError):
                                 valore_convertito = 'N/A'

                        val_display = valore_convertito if valore_convertito != 'N/A' else valore_sensore
                        na_info = " (Interpretato come N/A)" if valore_convertito == 'N/A' and valore_sensore is not None else ""
                        print(f"  - {descr_sensore}: {val_display} {unita_misura}{na_info}")

                        dati_api_per_riga[intestazione_sensore] = valore_convertito

                        is_pioggia_tot_giornaliera = any(keyword.lower() in descr_sensore.lower() for keyword in DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS)

                        if is_pioggia_tot_giornaliera:
                            if isinstance(valore_convertito, float):
                                pioggia_tot_oggi_attuale = valore_convertito
                                print(f"    -> Identificato come Pioggia Totale Giornaliera: {pioggia_tot_oggi_attuale}")
                            else:
                                pioggia_tot_oggi_attuale = None
                                print(f"    -> Identificato come Pioggia Totale Giornaliera, ma valore non è float ({valore_convertito}).")

                # --- Calcolo Pioggia nell'intervallo (30 min) per la stazione corrente ---
                # *** MODIFICA: Nome colonna ripristinato a "Pioggia Ora (mm)" ***
                pioggia_ora_key = f"{nome_stazione} - Pioggia Ora (mm)"
                intestazioni_da_api[pioggia_ora_key] = True # Assicura che la colonna calcolata sia considerata

                pioggia_calcolata_intervallo = 'N/A' # Nome variabile più descrittivo
                valore_precedente_float = valori_precedenti_pioggia.get(nome_stazione)

                # *** MODIFICA: Log aggiornati per chiarezza sul calcolo a 30 min nella colonna "Ora" ***
                print(f"  Calcolo Pioggia Intervallo (30 min) per colonna '{pioggia_ora_key}':")
                print(f"    - Pioggia Tot Oggi Attuale: {pioggia_tot_oggi_attuale}")
                print(f"    - Valore Tot Precedente (da foglio): {valore_precedente_float}")

                if pioggia_tot_oggi_attuale is not None:
                    if valore_precedente_float is None:
                        pioggia_calcolata_intervallo = pioggia_tot_oggi_attuale
                        print(f"    -> Nessun valore precedente valido. Pioggia Intervallo = Valore Attuale ({pioggia_calcolata_intervallo})")
                    elif pioggia_tot_oggi_attuale < valore_precedente_float:
                        pioggia_calcolata_intervallo = pioggia_tot_oggi_attuale
                        print(f"    WARN: Rilevato calo/reset pioggia (Prec: {valore_precedente_float}, Attuale: {pioggia_tot_oggi_attuale}). Pioggia Intervallo = Valore Attuale ({pioggia_calcolata_intervallo})")
                    else:
                        pioggia_calcolata_intervallo = round(pioggia_tot_oggi_attuale - valore_precedente_float, 2)
                        print(f"    -> Calcolo standard intervallo: {pioggia_tot_oggi_attuale} - {valore_precedente_float} = {pioggia_calcolata_intervallo}")

                    if isinstance(pioggia_calcolata_intervallo, (int, float)):
                         dati_api_per_riga[pioggia_ora_key] = float(pioggia_calcolata_intervallo)
                    else:
                         dati_api_per_riga[pioggia_ora_key] = 'N/A'
                         print(f"    -> Risultato calcolo non numerico. Impostato a N/A")

                else:
                    dati_api_per_riga[pioggia_ora_key] = 'N/A'
                    print(f"    -> Pioggia Tot Oggi Attuale non valida ('{pioggia_tot_oggi_attuale}'). Pioggia Intervallo = N/A")

                print(f"  -> Risultato per '{pioggia_ora_key}': {dati_api_per_riga.get(pioggia_ora_key, 'Errore Interno')} mm")
            # Fine del blocco 'if process_this_station:'

        # --- Preparazione e Scrittura Riga nel Foglio ---

        if not dati_api_per_riga:
            print("\nNessun dato valido estratto/processato per le stazioni/sensori selezionati dall'API.")
            sys.exit(0)

        intestazione_teorica_da_api = ['Data_Ora'] + sorted(intestazioni_da_api.keys())

        intestazione_da_usare = []
        if not intestazione_attuale_foglio:
            print("\nFoglio vuoto o intestazione non leggibile. Scrittura nuova intestazione...")
            try:
                # *** MODIFICA: L'intestazione creata conterrà "Pioggia Ora (mm)" ***
                sheet.append_row(intestazione_teorica_da_api, value_input_option='USER_ENTERED')
                intestazione_da_usare = intestazione_teorica_da_api
                print(f"Intestazione scritta: {intestazione_da_usare}")
            except Exception as e_write_header:
                print(f"Errore CRITICO durante la scrittura dell'intestazione iniziale: {e_write_header}")
                sys.exit(f"Impossibile scrivere l'intestazione nel foglio: {NOME_FOGLIO}. Uscita.")
        else:
             intestazione_da_usare = intestazione_attuale_foglio
             print(f"\nUtilizzo l'intestazione esistente dal foglio ({len(intestazione_da_usare)} colonne).")

             colonne_foglio_set = set(intestazione_attuale_foglio)
             colonne_api_set = set(intestazione_teorica_da_api)
             colonne_mancanti_nel_foglio = sorted(list(colonne_api_set - colonne_foglio_set))
             colonne_extra_nel_foglio = sorted(list(colonne_foglio_set - colonne_api_set))

             # *** MODIFICA: Rimosso il warning specifico per "Pioggia 30 Min" ***
             if colonne_mancanti_nel_foglio:
                 print(f"WARN: Colonne nei dati API ma non nel foglio: {colonne_mancanti_nel_foglio} (NON verranno scritte)")
                 # Aggiunto suggerimento generico se manca la colonna "Pioggia Ora"
                 if any("Pioggia Ora (mm)" in col for col in colonne_mancanti_nel_foglio):
                      print("      -> NOTA: La colonna 'Pioggia Ora (mm)' non è presente nell'intestazione del foglio. Potresti doverla aggiungere manualmente o cancellare l'intestazione esistente per ricrearla.")
             if colonne_extra_nel_foglio:
                 print(f"WARN: Colonne nel foglio ma non nei dati API attuali: {colonne_extra_nel_foglio} (Verrà scritto 'N/A')")


        riga_dati_finale = []
        for header_col in intestazione_da_usare:
            if header_col == 'Data_Ora':
                riga_dati_finale.append(formatted_time)
            else:
                valore = dati_api_per_riga.get(header_col, 'N/A')
                if isinstance(valore, float):
                     riga_dati_finale.append(valore)
                else:
                     riga_dati_finale.append(str(valore))

        print(f"\nRiga di dati pronta per l'inserimento ({len(riga_dati_finale)} elementi, in base all'intestazione del foglio):")
        preview_limit = 15
        print(riga_dati_finale[:preview_limit], "..." if len(riga_dati_finale) > preview_limit else "")

        try:
            sheet.append_row(riga_dati_finale, value_input_option='USER_ENTERED')
            print(f"\nDati aggiunti con successo al foglio Google '{NOME_FOGLIO}'")
        except Exception as e_append:
            print(f"ERRORE durante l'aggiunta della riga di dati al foglio: {e_append}")
            traceback.print_exc()
            sys.exit(f"Impossibile scrivere i dati nel foglio: {NOME_FOGLIO}. Uscita.")

    except requests.exceptions.RequestException as e_api:
        print(f"Errore nella richiesta API a {API_URL}: {e_api}")
        sys.exit(1)
    except json.JSONDecodeError as e_json:
        print(f"Errore nel parsing JSON dalla risposta API. Risposta ricevuta (primi 500 caratteri):\n{response.text[:500] if response else 'Nessuna risposta'}")
        print(f"Errore JSON: {e_json}")
        sys.exit(1)
    except Exception as e_main:
        print(f"Errore generico durante l'esecuzione: {e_main}")
        traceback.print_exc()
        sys.exit(1)

    print(f"--- Fine Script Dati Meteo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")


if __name__ == "__main__":
    estrai_dati_meteo()
