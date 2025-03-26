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
NOME_FILE_CREDENZIALI = "credentials.json" # Assicurati sia nel posto giusto
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
API_URL = "https://retemir.regione.marche.it/api/stations/rt-data"
STAZIONI_INTERESSATE = [
    "Misa", "Pianello di Ostra", "Nevola", "Barbara",
    "Serra dei Conti", "Arcevia", "Corinaldo", "Ponte Garibaldi",
]
# VERIFICA QUESTI TIPO_SENS! Qual è quello per la Pioggia Totale Giornaliera? (Es. 5?)
SENSORI_INTERESSATI_TIPOSENS = [0, 1, 5, 6, 9, 10, 100, 101]
# Stringa o criteri per identificare la pioggia totale giornaliera nella descrizione
# Potrebbe essere necessario cambiarla dopo aver ispezionato l'output API!
DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS = ["Pioggia TOT Oggi"] # Prova iniziale

# --- FINE CONFIGURAZIONE ---

def estrai_dati_meteo():
    """
    Estrae i dati meteo dall'API, calcola la pioggia oraria basandosi sull'ultimo dato
    nel Google Sheet e aggiunge la nuova riga di dati al foglio.
    Progettato per essere eseguito una volta per invocazione (es. tramite GitHub Actions).
    Logica intestazioni modificata per NON cancellare i dati esistenti.
    """
    print(f"--- Inizio Script Dati Meteo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
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
            # Crealo e ottieni l'oggetto foglio
            spreadsheet = client.create(NOME_FOGLIO)
            sheet = spreadsheet.sheet1
            # Potresti voler dare permessi di scrittura all'utente/gruppo se necessario
            # spreadsheet.share('tuo-utente@example.com', perm_type='user', role='writer')
            print(f"Foglio Google '{NOME_FOGLIO}' creato con successo.")
            # Lascia che l'intestazione venga scritta più avanti se il foglio è nuovo
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
    try:
        # Prova a ottenere il numero di righe non vuote
        all_values = sheet.get_all_values() # Meno efficiente ma più robusto per capire se è vuoto
        last_row_index = len(all_values)

        if last_row_index > 0:
             intestazione_attuale_foglio = all_values[0]
             print(f"Intestazione letta dal foglio (Riga 1): {intestazione_attuale_foglio}")
        else:
             print("Foglio vuoto. L'intestazione verrà creata.")

        if last_row_index > 1:  # Se esiste almeno una riga di dati dopo l'intestazione
            last_row_data = all_values[-1] # Prendi l'ultima riga
            print(f"Ultima riga di dati letta (Indice logico {last_row_index}): {last_row_data}")

            # Mappa nome colonna -> indice per sicurezza
            header_map = {name: idx for idx, name in enumerate(intestazione_attuale_foglio)}

            for col_name, idx in header_map.items():
                # Identifica le colonne di pioggia totale usando le keyword definite
                is_pioggia_tot_col = any(keyword.lower() in col_name.lower() for keyword in DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS)

                if is_pioggia_tot_col and "Pioggia Ora" not in col_name: # Assicurati non sia la colonna calcolata
                    # Estrai il nome stazione (assumendo formato "Nome Stazione - Descrizione (Unità)")
                    try:
                        station_name = col_name.split(" - ")[0].strip()
                    except IndexError:
                        print(f"WARN: Impossibile estrarre nome stazione da '{col_name}'")
                        continue

                    try:
                        # Cerca di leggere il valore precedente dalla colonna corrispondente
                        if idx < len(last_row_data):
                            prev_value_str = last_row_data[idx]
                            # Converte in float, gestendo sia '.' che ',' e stringhe vuote/N/A
                            if isinstance(prev_value_str, str) and prev_value_str.strip() != '' and prev_value_str.upper() != 'N/A':
                                prev_value_float = float(prev_value_str.replace(',', '.'))
                                valori_precedenti_pioggia[station_name] = prev_value_float
                                print(f"  -> Letto valore pioggia precedente per '{station_name}' da colonna '{col_name}': {prev_value_float}")
                            elif isinstance(prev_value_str, (int, float)):
                                valori_precedenti_pioggia[station_name] = float(prev_value_str)
                                print(f"  -> Letto valore pioggia precedente (numerico) per '{station_name}' da colonna '{col_name}': {valori_precedenti_pioggia[station_name]}")
                            else:
                                valori_precedenti_pioggia[station_name] = None
                                print(f"  -> Valore pioggia precedente non valido/vuoto per '{station_name}' in colonna '{col_name}'.")
                        else:
                             valori_precedenti_pioggia[station_name] = None
                             print(f"  -> Indice colonna '{col_name}' ({idx}) fuori dai limiti per l'ultima riga ({len(last_row_data)} elementi).")

                    except (ValueError, TypeError) as e_parse:
                        print(f"  -> Errore conversione valore precedente per '{station_name}' ('{prev_value_str}'): {e_parse}. Impostato a None.")
                        valori_precedenti_pioggia[station_name] = None # Nessun valore precedente valido
        else:
             print("Nessuna riga di dati trovata nel foglio (solo intestazione o vuoto). Calcolo pioggia oraria partirà da zero/N/A.")

    except gspread.exceptions.APIError as api_err:
         print(f"Errore API Google Sheets durante lettura dati: {api_err}")
         # Potrebbe indicare problemi di permessi o foglio non valido
         # Considera se uscire o continuare assumendo foglio vuoto
         # sys.exit(1)
    except Exception as e_read:
        print(f"Attenzione: Errore imprevisto durante lettura dal foglio. Calcolo pioggia oraria potrebbe non funzionare. Errore: {e_read}")
        # Resetta intestazione letta se la lettura fallisce in modo grave
        intestazione_attuale_foglio = []

    # --- Estrazione Dati API ---
    current_time = datetime.now()
    formatted_time = current_time.strftime('%d/%m/%Y %H:%M') # Orario di esecuzione script
    print(f"\nEstrazione dati API alle {formatted_time}...")
    try:
        response = requests.get(API_URL, verify=False, timeout=30)
        response.raise_for_status()
        dati_meteo = response.json()
        print("Dati API ricevuti con successo.")

        # Rinnova token se scaduto (opzionale, gspread di solito lo gestisce)
        if credenziali.access_token_expired:
            print("Token Google scaduto, rinnovo...")
            client.login()

        dati_api_per_riga = {} # Dati attuali letti dall'API (chiave = nome colonna)
        intestazioni_da_api = {} # Tiene traccia delle colonne presenti nei dati API attuali

        for stazione in dati_meteo:
            nome_stazione = stazione.get("nome")
            if nome_stazione in STAZIONI_INTERESSATE:
                print(f"\nProcessando stazione: {nome_stazione}")
                # DEBUG: Stampa tutti i sensori per questa stazione per capire la descrizione della pioggia
                print(f"  DEBUG: Sensori disponibili per {nome_stazione}:")
                for sensore_debug in stazione.get("analog", []):
                     print(f"    - tipoSens: {sensore_debug.get('tipoSens')}, descr: '{sensore_debug.get('descr')}', valore: {sensore_debug.get('valore')}, unmis: '{sensore_debug.get('unmis')}'")

                pioggia_tot_oggi_attuale = None # Resetta per ogni stazione

                for sensore in stazione.get("analog", []):
                    tipoSens = sensore.get("tipoSens")
                    if tipoSens in SENSORI_INTERESSATI_TIPOSENS:
                        descr_sensore = sensore.get("descr", "").strip()
                        valore_sensore = sensore.get("valore")
                        unita_misura = sensore.get("unmis", "").strip() if sensore.get("unmis") else ""

                        # Costruisci nome colonna standard
                        intestazione_sensore = f"{nome_stazione} - {descr_sensore} ({unita_misura})"
                        intestazioni_da_api[intestazione_sensore] = True # Segna che questa colonna esiste nei dati API

                        # Memorizza il valore del sensore corrente, convertendo subito
                        valore_convertito = 'N/A' # Default
                        if valore_sensore is not None:
                            try:
                                 valore_convertito = float(str(valore_sensore).replace(',','.'))
                                 print(f"  - {descr_sensore}: {valore_convertito} {unita_misura}")
                            except (ValueError, TypeError):
                                 valore_convertito = 'N/A' # Non numerico o errore
                                 print(f"  - {descr_sensore}: {valore_sensore} {unita_misura} (Valore non numerico, interpretato come N/A)")
                        else:
                             print(f"  - {descr_sensore}: Valore mancante (None) dall'API. Impostato a N/A")

                        dati_api_per_riga[intestazione_sensore] = valore_convertito

                        # Identifica se questo è il sensore di pioggia totale giornaliera
                        # USA LE KEYWORD definite all'inizio
                        is_pioggia_tot_giornaliera = any(keyword.lower() in descr_sensore.lower() for keyword in DESCRIZIONE_PIOGGIA_TOT_GIORNALIERA_KEYWORDS)

                        if is_pioggia_tot_giornaliera:
                            if isinstance(valore_convertito, float):
                                pioggia_tot_oggi_attuale = valore_convertito
                                print(f"    -> Identificato come Pioggia Totale Giornaliera: {pioggia_tot_oggi_attuale}")
                            else:
                                pioggia_tot_oggi_attuale = None # Valore non valido per il calcolo
                                print(f"    -> Identificato come Pioggia Totale Giornaliera, ma valore non è float ({valore_convertito}).")


                # --- Calcolo Pioggia Oraria per la stazione corrente ---
                pioggia_ora_key = f"{nome_stazione} - Pioggia Ora (mm)"
                intestazioni_da_api[pioggia_ora_key] = True # Assicura che la colonna calcolata sia considerata

                pioggia_ora_calcolata = 'N/A' # Default
                valore_precedente_float = valori_precedenti_pioggia.get(nome_stazione) # Preso dal foglio all'inizio

                print(f"  Calcolo Pioggia Oraria per {nome_stazione}:")
                print(f"    - Pioggia Tot Oggi Attuale: {pioggia_tot_oggi_attuale}")
                print(f"    - Valore Precedente (da foglio): {valore_precedente_float}")

                if pioggia_tot_oggi_attuale is not None: # Solo se abbiamo un valore attuale valido
                    current_hour = current_time.hour

                    if valore_precedente_float is None:
                        # Nessun dato precedente valido o prima esecuzione per questa stazione
                        if current_hour == 0:
                             # Alla mezzanotte (prima ora), il valore attuale è la pioggia di quell'ora
                             pioggia_ora_calcolata = pioggia_tot_oggi_attuale
                             print(f"    -> Nessun prec., ora 00: Pioggia Ora = Attuale ({pioggia_ora_calcolata})")
                        else:
                             # Non possiamo calcolare la differenza, mettiamo 0.0 (più utile di N/A per somme)
                             pioggia_ora_calcolata = 0.0
                             print(f"    -> Nessun prec., ora {current_hour}: Pioggia Ora = 0.0")
                    elif current_hour == 0:
                         # E' mezzanotte (o poco dopo), il valore letto è la pioggia della prima ora del nuovo giorno.
                         # Non sottrarre dal giorno precedente.
                         pioggia_ora_calcolata = pioggia_tot_oggi_attuale
                         print(f"    -> Mezzanotte con prec.: Pioggia Ora = Attuale ({pioggia_ora_calcolata})")
                    elif pioggia_tot_oggi_attuale < valore_precedente_float:
                         # Reset del contatore API? Mettiamo 0.0 per l'incremento orario.
                         pioggia_ora_calcolata = 0.0
                         print(f"    WARN: Rilevato calo pioggia per {nome_stazione} (Prec: {valore_precedente_float}, Attuale: {pioggia_tot_oggi_attuale}). Pioggia Ora = 0.0")
                    else:
                         # Calcolo standard della differenza oraria
                         pioggia_ora_calcolata = round(pioggia_tot_oggi_attuale - valore_precedente_float, 2)
                         print(f"    -> Calcolo standard: {pioggia_tot_oggi_attuale} - {valore_precedente_float} = {pioggia_ora_calcolata}")

                    # Memorizza il risultato (assicurati sia float se numerico)
                    if isinstance(pioggia_ora_calcolata, (int, float)):
                         dati_api_per_riga[pioggia_ora_key] = float(pioggia_ora_calcolata)
                    else:
                         dati_api_per_riga[pioggia_ora_key] = 'N/A'
                else:
                    # Se pioggia_tot_oggi_attuale non era valido, anche pioggia oraria è N/A
                    dati_api_per_riga[pioggia_ora_key] = 'N/A'
                    print(f"    -> Pioggia Tot Oggi non valida, Pioggia Ora = N/A")

                print(f"  -> Risultato Pioggia Ora Calcolata: {dati_api_per_riga[pioggia_ora_key]} mm")


        if not dati_api_per_riga:
            print("\nNessun dato valido estratto per le stazioni/sensori selezionati dall'API.")
            sys.exit(0) # Esce senza errori, ma non scrive nulla

        # --- Preparazione e Scrittura Riga nel Foglio ---

        # Costruisci l'elenco teorico delle intestazioni basato sui dati API di QUESTA esecuzione
        # Manteniamo l'ordine alfabetico per coerenza *teorica*
        intestazione_teorica_da_api = ['Data_Ora'] + sorted(intestazioni_da_api.keys())

        # Verifica se l'intestazione nel foglio esiste
        if not intestazione_attuale_foglio:
            print("\nFoglio vuoto o intestazione non leggibile. Scrittura nuova intestazione...")
            try:
                # Scrivi l'intestazione basata sui dati appena letti
                sheet.append_row(intestazione_teorica_da_api, value_input_option='USER_ENTERED')
                intestazione_da_usare = intestazione_teorica_da_api # Da ora in poi usa questa
                print(f"Intestazione scritta: {intestazione_da_usare}")
                # Se abbiamo appena scritto l'intestazione, non c'è una riga precedente di dati reali
                # Quindi non possiamo aggiungere la riga corrente in questa stessa esecuzione
                # se non forzando valori precedenti a None (già fatto all'inizio)
                # Alternativa: fare un `return` o `sys.exit(0)` qui per aspettare la prossima run.
                # Ma proviamo a scrivere comunque la prima riga di dati.
            except Exception as e_write_header:
                print(f"Errore CRITICO durante la scrittura dell'intestazione iniziale: {e_write_header}")
                sys.exit(f"Impossibile scrivere l'intestazione nel foglio: {NOME_FOGLIO}. Uscita.")
        else:
             # L'intestazione esiste già nel foglio. USA QUELLA per determinare l'ordine.
             intestazione_da_usare = intestazione_attuale_foglio
             print(f"\nUtilizzo l'intestazione esistente dal foglio: {intestazione_da_usare}")

             # LOGGA differenze tra intestazione foglio e dati API attuali (ma non modificare il foglio)
             colonne_foglio = set(intestazione_attuale_foglio)
             colonne_api = set(intestazione_teorica_da_api) # Include 'Data_Ora' e le colonne calcolate

             colonne_mancanti_nel_foglio = sorted(list(colonne_api - colonne_foglio))
             colonne_extra_nel_foglio = sorted(list(colonne_foglio - colonne_api)) # Sensori non più inviati?

             if colonne_mancanti_nel_foglio:
                 print(f"WARN: Le seguenti colonne esistono nei dati API ma non nell'intestazione del foglio: {colonne_mancanti_nel_foglio}")
                 print("      Questi dati NON verranno scritti. Aggiungere manualmente le colonne al foglio se necessario.")
             if colonne_extra_nel_foglio:
                 print(f"WARN: Le seguenti colonne esistono nell'intestazione del foglio ma non nei dati API attuali: {colonne_extra_nel_foglio}")
                 print("      Verrà scritto 'N/A' per queste colonne.")

        # Costruisci la riga di dati FINALE nell'ordine definito da `intestazione_da_usare`
        riga_dati_finale = []
        for header_col in intestazione_da_usare:
            if header_col == 'Data_Ora':
                riga_dati_finale.append(formatted_time)
            else:
                # Prendi il valore dai dati processati dall'API, usa 'N/A' se mancante per questa colonna specifica
                valore = dati_api_per_riga.get(header_col, 'N/A')

                # Formatta correttamente per Google Sheets
                if isinstance(valore, float):
                     # Usa il punto come separatore decimale per USER_ENTERED
                     riga_dati_finale.append(valore)
                # elif isinstance(valore, int): # Anche gli interi vanno bene come numeri
                #      riga_dati_finale.append(valore)
                else:
                     # Invia 'N/A' o altri non-numerici come stringhe
                     riga_dati_finale.append(str(valore))

        print(f"\nRiga di dati pronta per l'inserimento ({len(riga_dati_finale)} elementi):")
        # Stampa solo i primi N elementi per brevità se la riga è lunga
        preview_limit = 15
        print(riga_dati_finale[:preview_limit], "..." if len(riga_dati_finale) > preview_limit else "")

        # Aggiungi la riga di dati al foglio
        try:
            sheet.append_row(riga_dati_finale, value_input_option='USER_ENTERED')
            print(f"\nDati aggiunti con successo al foglio Google '{NOME_FOGLIO}'")
        except Exception as e_append:
            print(f"ERRORE durante l'aggiunta della riga di dati al foglio: {e_append}")
            # Potresti voler implementare un tentativo di retry qui
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
        traceback.print_exc() # Stampa lo stack trace completo per il debug
        sys.exit(1)

    print(f"--- Fine Script Dati Meteo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")


if __name__ == "__main__":
    estrai_dati_meteo()
