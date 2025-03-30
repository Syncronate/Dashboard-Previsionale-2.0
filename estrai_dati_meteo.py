import requests
import json
from datetime import datetime
import sys
import urllib3
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import traceback # Importato per logging errori più dettagliato
import pytz      # <-- AGGIUNTO per la gestione dei fusi orari

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
    # --- Calcolo Ora Iniziale Script (con fuso orario corretto) ---
    try:
        italian_tz = pytz.timezone('Europe/Rome')
        start_time_local = datetime.now(italian_tz)
        start_formatted_time = start_time_local.strftime('%Y-%m-%d %H:%M:%S %Z%z') # Formato più dettagliato per log
        print(f"--- Inizio Script Dati Meteo ({start_formatted_time}) ---")
    except Exception as e_tz_init:
         # Fallback a ora di sistema se pytz fallisce
         start_time_local = datetime.now()
         start_formatted_time = start_time_local.strftime('%Y-%m-%d %H:%M:%S')
         print(f"--- Inizio Script Dati Meteo ({start_formatted_time} - ATTENZIONE: Fuso orario Europe/Rome non caricato: {e_tz_init}) ---")

    print("NOTA: Lo script calcola la pioggia nell'intervallo di esecuzione (30 min) e la scrive nella colonna 'Pioggia Ora (mm)'.")
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
            # Scrivi subito l'intestazione se il foglio è nuovo
            print(f"Foglio Google '{NOME_FOGLIO}' creato con successo. Intestazione iniziale verrà scritta dopo l'estrazione API.")
        except Exception as e_open:
             print(f"Errore durante l'apertura/creazione del foglio '{NOME_FOGLIO}': {e_open}")
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
        all_values = sheet.get_all_values() # Può fallire se il foglio è stato appena creato e vuoto
        if all_values: # Verifica se ci sono dati
            last_row_index = len(all_values)

            if last_row_index > 0:
                 intestazione_attuale_foglio = all_values[0]
                 print(f"Intestazione letta dal foglio (Riga 1): {intestazione_attuale_foglio}")
            else:
                 print("Foglio apparentemente vuoto (nessuna riga). L'intestazione verrà creata.")

            if last_row_index > 1:
                last_row_data = all_values[-1]
                print(f"Ultima riga di dati letta (Indice logico {last_row_index}): {last_row_data}")

                # Crea mappa intestazione solo se esiste e ci sono dati
                if intestazione_attuale_foglio and last_row_data:
                    header_map = {name: idx for idx, name in enumerate(intestazione_attuale_foglio)}

                    for col_name, idx in header_map.items():
                        is_pioggia_tot_col = any(keyword.lower() in col_name.lower() for keyword in DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS)

                        # Legge dalla colonna Pioggia TOT, ignorando la Pioggia Ora calcolata
                        if is_pioggia_tot_col and "Pioggia Ora" not in col_name:
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
                 print("Nessuna riga di dati trovata nel foglio (solo intestazione o vuoto). Calcolo pioggia nell'intervallo (30 min) partirà dal valore attuale.")
        else:
             print("Foglio completamente vuoto. L'intestazione verrà creata dopo l'estrazione API.")


    except gspread.exceptions.APIError as api_err:
         print(f"Errore API Google Sheets durante lettura dati: {api_err}")
    except Exception as e_read:
        print(f"Attenzione: Errore imprevisto durante lettura dal foglio. Calcolo pioggia potrebbe non funzionare. Errore: {e_read}")
        # Non impostare intestazione a [], potrebbe essere valida ma la lettura dati è fallita
        # intestazione_attuale_foglio = [] # Rimosso

    # --- Estrazione Dati API e Calcolo Ora Corretta ---
    try:
        # Definisci il fuso orario italiano
        italian_tz = pytz.timezone('Europe/Rome')
        # Ottieni l'ora attuale nel fuso orario italiano
        current_time_local = datetime.now(italian_tz)
        # Formatta l'ora come necessario per il foglio ('gg/mm/aaaa HH:MM')
        formatted_time = current_time_local.strftime('%d/%m/%Y %H:%M')
        print(f"\nOra corrente (Europe/Rome) per i dati: {formatted_time}") # Log aggiuntivo per verifica

    except pytz.exceptions.UnknownTimeZoneError:
        print("ERRORE: Fuso orario 'Europe/Rome' non trovato. Assicurati che pytz sia installato.")
        # Fallback all'ora di sistema (come prima, ma con un warning)
        current_time_local = datetime.now()
        formatted_time = current_time_local.strftime('%d/%m/%Y %H:%M')
        print(f"ATTENZIONE: Utilizzo l'ora di sistema per i dati (potrebbe essere errata): {formatted_time}")
    except Exception as e_time:
        print(f"Errore imprevisto durante la gestione del fuso orario: {e_time}")
        # Fallback all'ora di sistema
        current_time_local = datetime.now()
        formatted_time = current_time_local.strftime('%d/%m/%Y %H:%M')
        print(f"ATTENZIONE: Utilizzo l'ora di sistema per i dati (potrebbe essere errata): {formatted_time}")

    print(f"Estrazione dati API alle {formatted_time}...")
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
                     # Già loggato se Arcevia è duplicata, evitiamo log ridondanti
                     if nome_stazione != "Arcevia":
                        print(f"INFO: Stazione '{nome_stazione}' (codice {codice_stazione}) già processata con altro codice? Ignorando duplicato nome.")
                elif nome_stazione == "Arcevia":
                    if codice_stazione == CODICE_ARCEVIA_CORRETTO:
                        process_this_station = True
                        stazioni_processate_nomi.add(nome_stazione) # Aggiungi solo se è quella corretta
                    else:
                        # Non loggare ogni volta che si incontra Arcevia sbagliata, diventa troppo verboso
                        pass # print(f"INFO: Ignorando stazione '{nome_stazione}' con codice {codice_stazione} (si usa solo {CODICE_ARCEVIA_CORRETTO}).")
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

                        # Pulisci eventuali spazi multipli o caratteri strani nell'intestazione
                        intestazione_sensore = f"{nome_stazione} - {descr_sensore} ({unita_misura})".replace("  ", " ").strip()
                        intestazioni_da_api[intestazione_sensore] = True

                        valore_convertito = 'N/A'
                        if valore_sensore is not None:
                            try:
                                 # Gestisce sia stringhe con virgola che numeri diretti
                                 valore_convertito = float(str(valore_sensore).replace(',', '.'))
                            except (ValueError, TypeError):
                                 valore_convertito = 'N/A' # Lascia 'N/A' se la conversione fallisce

                        # Log migliorato
                        val_display = f"{valore_convertito:.2f}" if isinstance(valore_convertito, float) else valore_convertito
                        raw_val_info = f"(raw: '{valore_sensore}')" if str(valore_sensore) != str(val_display) else ""
                        print(f"  - {descr_sensore}: {val_display} {unita_misura} {raw_val_info}")

                        dati_api_per_riga[intestazione_sensore] = valore_convertito

                        # Controlla se questo è il sensore della pioggia totale giornaliera
                        is_pioggia_tot_giornaliera = any(keyword.lower() in descr_sensore.lower() for keyword in DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS)

                        if is_pioggia_tot_giornaliera:
                            if isinstance(valore_convertito, float):
                                pioggia_tot_oggi_attuale = valore_convertito
                                print(f"    -> Identificato come Pioggia Totale Giornaliera: {pioggia_tot_oggi_attuale:.2f}")
                            else:
                                # Se non è float, lo trattiamo come non valido per il calcolo
                                pioggia_tot_oggi_attuale = None
                                print(f"    -> Identificato come Pioggia Totale Giornaliera, ma valore non numerico ({valore_convertito}).")

                # --- Calcolo Pioggia nell'intervallo (30 min) per la stazione corrente ---
                pioggia_ora_key = f"{nome_stazione} - Pioggia Ora (mm)"
                intestazioni_da_api[pioggia_ora_key] = True # Assicura che la colonna calcolata sia considerata

                pioggia_calcolata_intervallo = 'N/A'
                valore_precedente_float = valori_precedenti_pioggia.get(nome_stazione) # Restituisce None se la stazione non c'era

                print(f"  Calcolo Pioggia Intervallo (30 min) per colonna '{pioggia_ora_key}':")
                print(f"    - Pioggia Tot Oggi Attuale: {pioggia_tot_oggi_attuale if pioggia_tot_oggi_attuale is not None else 'N/D'}")
                print(f"    - Valore Tot Precedente (da foglio): {valore_precedente_float if valore_precedente_float is not None else 'Nessuno o N/D'}")

                if pioggia_tot_oggi_attuale is not None: # Procede solo se il valore attuale è un numero valido
                    if valore_precedente_float is None:
                        # Se non c'è storico o il valore precedente era N/A, la pioggia "dell'intervallo" è il totale attuale
                        # Questo è sensato se lo script parte da zero o dopo un errore
                        pioggia_calcolata_intervallo = pioggia_tot_oggi_attuale
                        print(f"    -> Nessun valore precedente valido. Pioggia Intervallo = Valore Attuale ({pioggia_calcolata_intervallo:.2f})")
                    elif pioggia_tot_oggi_attuale < valore_precedente_float:
                        # Reset del contatore a mezzanotte o errore sensore?
                        # In questo caso, la pioggia nell'intervallo è il nuovo valore totale
                        pioggia_calcolata_intervallo = pioggia_tot_oggi_attuale
                        print(f"    WARN: Rilevato calo/reset pioggia (Prec: {valore_precedente_float:.2f}, Attuale: {pioggia_tot_oggi_attuale:.2f}). Pioggia Intervallo = Valore Attuale ({pioggia_calcolata_intervallo:.2f})")
                    else:
                        # Calcolo standard della differenza
                        pioggia_calcolata_intervallo = round(pioggia_tot_oggi_attuale - valore_precedente_float, 2)
                        print(f"    -> Calcolo standard intervallo: {pioggia_tot_oggi_attuale:.2f} - {valore_precedente_float:.2f} = {pioggia_calcolata_intervallo:.2f}")

                    # Assegna il valore calcolato (o 'N/A' se il calcolo non ha prodotto un numero)
                    dati_api_per_riga[pioggia_ora_key] = float(pioggia_calcolata_intervallo) if isinstance(pioggia_calcolata_intervallo, (int, float)) else 'N/A'

                else:
                    # Se il valore attuale della pioggia totale non è valido, non possiamo calcolare l'intervallo
                    dati_api_per_riga[pioggia_ora_key] = 'N/A'
                    print(f"    -> Pioggia Tot Oggi Attuale non valida ('{pioggia_tot_oggi_attuale}'). Pioggia Intervallo = N/A")

                print(f"  -> Risultato per '{pioggia_ora_key}': {dati_api_per_riga.get(pioggia_ora_key, 'Errore Interno')} mm")
            # Fine del blocco 'if process_this_station:'

        # --- Preparazione e Scrittura Riga nel Foglio ---

        if not dati_api_per_riga:
            print("\nNessun dato valido estratto/processato per le stazioni/sensori selezionati dall'API.")
            # Non uscire necessariamente, potremmo dover scrivere N/A se l'intestazione esiste
            # sys.exit(0) # Rimosso

        # Definire l'intestazione teorica basata sui dati appena letti dall'API
        intestazione_teorica_da_api = ['Data_Ora'] + sorted(intestazioni_da_api.keys())

        intestazione_da_usare = []
        # Se il foglio era vuoto O se non siamo riusciti a leggere l'intestazione prima
        if not intestazione_attuale_foglio:
            print("\nFoglio vuoto o intestazione non letta precedentemente. Scrittura nuova intestazione...")
            try:
                # Controlla se la riga 1 è veramente vuota prima di scrivere
                row1 = sheet.row_values(1) if last_row_index > 0 else [] # Leggi riga 1 solo se non vuoto
                if not row1: # Se la riga 1 è vuota, scrivi l'intestazione
                    sheet.update('A1', [intestazione_teorica_da_api], value_input_option='USER_ENTERED')
                    # sheet.append_row(intestazione_teorica_da_api, value_input_option='USER_ENTERED') # append_row aggiunge in fondo
                    intestazione_da_usare = intestazione_teorica_da_api
                    print(f"Intestazione scritta in riga 1: {intestazione_da_usare}")
                    # Rileggi l'intestazione per sicurezza (anche se l'abbiamo appena scritta)
                    intestazione_attuale_foglio = intestazione_teorica_da_api
                    last_row_index = 1 # Ora c'è una riga (l'intestazione)
                else:
                    # La riga 1 esiste già, usala come intestazione
                    print("Intestazione già presente in riga 1, non sovrascritta.")
                    intestazione_da_usare = row1
                    intestazione_attuale_foglio = row1
                    # Non aggiornare last_row_index qui, lo abbiamo già letto prima
            except Exception as e_write_header:
                print(f"Errore CRITICO durante la scrittura/verifica dell'intestazione iniziale: {e_write_header}")
                sys.exit(f"Impossibile scrivere/verificare l'intestazione nel foglio: {NOME_FOGLIO}. Uscita.")
        else:
             # L'intestazione è stata letta con successo prima
             intestazione_da_usare = intestazione_attuale_foglio
             print(f"\nUtilizzo l'intestazione esistente dal foglio ({len(intestazione_da_usare)} colonne).")

             # Controllo discrepanze (solo informativo)
             colonne_foglio_set = set(intestazione_attuale_foglio)
             colonne_api_set = set(intestazione_teorica_da_api)
             colonne_mancanti_nel_foglio = sorted(list(colonne_api_set - colonne_foglio_set))
             colonne_extra_nel_foglio = sorted(list(colonne_foglio_set - colonne_api_set - {'Data_Ora'})) # Ignora Data_Ora qui

             if colonne_mancanti_nel_foglio:
                 print(f"WARN: Colonne nei dati API ma non nel foglio: {colonne_mancanti_nel_foglio} (NON verranno scritte)")
                 if any("Pioggia Ora (mm)" in col for col in colonne_mancanti_nel_foglio):
                      print("      -> NOTA: La colonna calcolata 'Pioggia Ora (mm)' non è presente nell'intestazione del foglio. Potrebbe essere necessario aggiungerla manualmente o ricreare l'intestazione.")
             if colonne_extra_nel_foglio:
                 print(f"WARN: Colonne nel foglio ma non nei dati API attuali: {colonne_extra_nel_foglio} (Verrà scritto 'N/A')")


        # Costruisci la riga finale basandoti sull'intestazione DA USARE
        riga_dati_finale = []
        if not intestazione_da_usare:
             print("ERRORE: Impossibile determinare l'intestazione da usare. Non posso preparare i dati.")
             sys.exit("Uscita a causa di intestazione mancante.")

        for header_col in intestazione_da_usare:
            if header_col == 'Data_Ora':
                # Usa la formatted_time calcolata all'inizio della sezione API
                riga_dati_finale.append(formatted_time)
            else:
                # Prendi il valore dai dati API; se manca, metti 'N/A'
                valore = dati_api_per_riga.get(header_col, 'N/A')
                # Assicurati che i float siano scritti come numeri, non stringhe
                if isinstance(valore, float):
                     # Formatta con punto decimale per coerenza
                     riga_dati_finale.append(f'{valore:.2f}'.replace('.', ',')) # Usa la virgola per Sheets Italia
                     # riga_dati_finale.append(valore) # Google Sheets dovrebbe interpretarlo correttamente
                elif isinstance(valore, int):
                     riga_dati_finale.append(valore)
                else:
                     # Lascia N/A o altri valori non numerici come stringhe
                     riga_dati_finale.append(str(valore))

        print(f"\nRiga di dati pronta per l'inserimento ({len(riga_dati_finale)} elementi, in base all'intestazione '{'nuova' if not intestazione_attuale_foglio else 'esistente'}'):")
        preview_limit = 15
        print(riga_dati_finale[:preview_limit], "..." if len(riga_dati_finale) > preview_limit else "")

        # Scrittura nel foglio
        try:
            # Aggiunge la riga DOPO l'ultima riga esistente (sia essa intestazione o dati)
            sheet.append_row(riga_dati_finale, value_input_option='USER_ENTERED')
            print(f"\nDati aggiunti con successo al foglio Google '{NOME_FOGLIO}'")
        except gspread.exceptions.APIError as e_append_api:
            print(f"ERRORE API durante l'aggiunta della riga di dati al foglio: {e_append_api}")
            print(f"Dettagli: {e_append_api.response.text}")
            traceback.print_exc()
            sys.exit(f"Impossibile scrivere i dati nel foglio: {NOME_FOGLIO}. Uscita.")
        except Exception as e_append:
            print(f"ERRORE generico durante l'aggiunta della riga di dati al foglio: {e_append}")
            traceback.print_exc()
            sys.exit(f"Impossibile scrivere i dati nel foglio: {NOME_FOGLIO}. Uscita.")

    except requests.exceptions.Timeout:
        print(f"Errore: Timeout durante la richiesta API a {API_URL}.")
        sys.exit(1)
    except requests.exceptions.RequestException as e_api:
        print(f"Errore nella richiesta API a {API_URL}: {e_api}")
        sys.exit(1)
    except json.JSONDecodeError as e_json:
        print(f"Errore nel parsing JSON dalla risposta API. Risposta ricevuta (primi 500 caratteri):\n{response.text[:500] if response else 'Nessuna risposta'}")
        print(f"Errore JSON: {e_json}")
        sys.exit(1)
    except Exception as e_main:
        print(f"Errore generico non gestito durante l'esecuzione: {e_main}")
        traceback.print_exc()
        sys.exit(1)

    # --- Calcolo Ora Fine Script ---
    try:
        end_time_local = datetime.now(italian_tz) # Usa lo stesso fuso orario
        end_formatted_time = end_time_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')
        print(f"\n--- Fine Script Dati Meteo ({end_formatted_time}) ---")
    except Exception: # Se pytz ha fallito all'inizio, fallirà anche qui
        end_time_local = datetime.now()
        end_formatted_time = end_time_local.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n--- Fine Script Dati Meteo ({end_formatted_time} - Fuso orario non applicato) ---")

if __name__ == "__main__":
    estrai_dati_meteo()
