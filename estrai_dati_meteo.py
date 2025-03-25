import requests
import json
# import time # Non più necessario per sleep
from datetime import datetime # Rimosso timedelta non più usato qui
import sys
import urllib3
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os # Utile per gestire percorsi file

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def estrai_dati_meteo():
    """
    Estrae i dati meteo dall'API, calcola la pioggia oraria basandosi sull'ultimo dato
    nel Google Sheet e aggiunge la nuova riga di dati al foglio.
    Progettato per essere eseguito una volta per invocazione (es. tramite GitHub Actions).
    """
    # --- Setup ---
    nome_foglio = "Dati Meteo Stazioni"
    # Assicurati che il file credentials.json sia nel path corretto rispetto allo script
    # Potrebbe essere utile usare un percorso assoluto o relativo allo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    credenziali_path = os.path.join(script_dir, "credentials.json")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # API information
    api_url = "https://retemir.regione.marche.it/api/stations/rt-data"
    stazioni_interessate = [
        "Misa", "Pianello di Ostra", "Nevola", "Barbara",
        "Serra dei Conti", "Arcevia", "Corinaldo", "Ponte Garibaldi",
    ]
    sensori_interessati_tipoSens = [0, 1, 5, 6, 9, 10, 100, 101]

    print(f"Inizio estrazione dati meteo per il foglio: {nome_foglio}")
    current_time = datetime.now()
    formatted_time = current_time.strftime('%d/%m/%Y %H:%M') # Orario di esecuzione script

    # --- Autenticazione Google Sheets ---
    try:
        credenziali = ServiceAccountCredentials.from_json_keyfile_name(credenziali_path, scope)
        client = gspread.authorize(credenziali)
        try:
            foglio = client.open(nome_foglio).sheet1
            print(f"Foglio Google '{nome_foglio}' aperto con successo.")
        except gspread.SpreadsheetNotFound:
            # Se il foglio non esiste, crealo. Non ci sarà storico per la pioggia.
            print(f"Foglio Google '{nome_foglio}' non trovato. Verrà creato.")
            foglio = client.create(nome_foglio).sheet1
            print(f"Foglio Google '{nome_foglio}' creato con successo.")
            # Aggiungi un'intestazione di base se è nuovo
            foglio.append_row(["Data_Ora"]) # Intestazione minima iniziale
    except Exception as e:
        print(f"Errore critico nell'autenticazione o apertura/creazione Google Sheets: {e}")
        sys.exit(1)

    # --- Lettura Ultimi Dati dal Foglio (per calcolo pioggia) ---
    valori_precedenti_pioggia = {}
    intestazione_attuale = []
    try:
        last_row_index = foglio.row_count
        if last_row_index > 0:
             intestazione_attuale = foglio.row_values(1)
             print(f"Intestazioni esistenti: {intestazione_attuale}")

        if last_row_index > 1:  # Se esiste almeno una riga di dati dopo l'intestazione
            last_row_data = foglio.row_values(last_row_index)
            print(f"Ultima riga di dati letta (indice {last_row_index}): {last_row_data}")

            for i, col_name in enumerate(intestazione_attuale):
                if "Pioggia TOT Oggi" in col_name:
                    station_name = col_name.split(" - ")[0]
                    try:
                        # Cerca di leggere il valore precedente dalla colonna corrispondente
                        prev_value_str = last_row_data[i]
                        # Converte in float, gestendo sia '.' che ',' come separatore per sicurezza
                        prev_value_float = float(str(prev_value_str).replace(',', '.'))
                        valori_precedenti_pioggia[station_name] = prev_value_float
                        print(f"  -> Letto valore precedente per '{station_name}': {prev_value_float}")
                    except (ValueError, IndexError, TypeError):
                        # Se il valore non è valido, non è un numero o la colonna non esiste
                        print(f"  -> Valore precedente non valido o assente per '{station_name}' nell'ultima riga.")
                        valori_precedenti_pioggia[station_name] = None # Nessun valore precedente valido
        else:
             print("Nessuna riga di dati trovata nel foglio (solo intestazione o vuoto). Calcolo pioggia oraria partirà da zero/N/A.")

    except gspread.exceptions.APIError as api_err:
         # Gestisce l'errore specifico se il foglio è completamente vuoto (neanche l'intestazione)
         if 'exceeds grid limits' in str(api_err) or 'range' in str(api_err).lower() and 'not found' in str(api_err).lower():
             print("Foglio completamente vuoto. Intestazioni verranno create.")
             intestazione_attuale = []
         else:
              print(f"Errore API Google Sheets durante lettura ultima riga/intestazioni: {api_err}")
              # Potresti voler uscire o continuare con valori precedenti vuoti
              # sys.exit(1)
    except Exception as e:
        print(f"Attenzione: Impossibile leggere l'ultima riga o le intestazioni dal foglio. Calcolo pioggia oraria partirà da zero/N/A. Errore: {e}")
        intestazione_attuale = [] # Resetta l'intestazione se la lettura fallisce


    # --- Estrazione Dati API ---
    print(f"\nEstrazione dati API alle {formatted_time}...")
    try:
        response = requests.get(api_url, verify=False, timeout=30)
        response.raise_for_status()
        dati_meteo = response.json()

        # Rinnova token se scaduto (gspread di solito lo gestisce, ma è una buona pratica)
        if credenziali.access_token_expired:
            client.login()
            # Riapri il foglio potrebbe non essere necessario se client si aggiorna
            # foglio = client.open(nome_foglio).sheet1

        intestazioni_nuove_trovate = {} # Usato per costruire la nuova riga e verificare cambiamenti header
        dati_per_riga = {} # Conterrà i valori per la riga da inserire

        for stazione in dati_meteo:
            nome_stazione = stazione.get("nome")
            if nome_stazione in stazioni_interessate:
                print(f"\nProcessando stazione: {nome_stazione}")
                # Timestamp dall'API (opzionale, usiamo formatted_time per coerenza di riga)
                # timestamp_stazione = stazione.get("lastUpdateTime")

                pioggia_tot_oggi_attuale = None # Memorizza il valore attuale per il calcolo orario

                for sensore in stazione.get("analog", []):
                    tipoSens = sensore.get("tipoSens")
                    if tipoSens in sensori_interessati_tipoSens:
                        descr_sensore = sensore.get("descr", "").strip()
                        valore_sensore = sensore.get("valore")
                        unita_misura = sensore.get("unmis", "").strip() if sensore.get("unmis") else ""

                        intestazione_sensore = f"{nome_stazione} - {descr_sensore} ({unita_misura})"
                        intestazioni_nuove_trovate[intestazione_sensore] = True # Segna che questa colonna esiste nei dati attuali

                        # Memorizza il valore del sensore corrente
                        # Cerca di convertire subito in float se possibile, gestendo errori
                        try:
                             valore_convertito = float(str(valore_sensore).replace(',','.'))
                             dati_per_riga[intestazione_sensore] = valore_convertito
                             print(f"  - {descr_sensore}: {valore_convertito} {unita_misura}")
                        except (ValueError, TypeError, AttributeError):
                             dati_per_riga[intestazione_sensore] = 'N/A' # Non numerico o None
                             print(f"  - {descr_sensore}: {valore_sensore} {unita_misura} (Interpretato come N/A)")


                        # Se è il sensore di pioggia totale giornaliera, usalo per il calcolo orario
                        if "Pioggia TOT Oggi" in descr_sensore:
                            if isinstance(dati_per_riga[intestazione_sensore], float):
                                pioggia_tot_oggi_attuale = dati_per_riga[intestazione_sensore]
                            else:
                                pioggia_tot_oggi_attuale = None # Valore non valido per il calcolo


                # --- Calcolo Pioggia Oraria per la stazione corrente ---
                pioggia_ora_key = f"{nome_stazione} - Pioggia Ora (mm)"
                intestazioni_nuove_trovate[pioggia_ora_key] = True # Aggiungi colonna pioggia oraria
                pioggia_ora_calcolata = 'N/A' # Default

                if pioggia_tot_oggi_attuale is not None:
                    valore_precedente_float = valori_precedenti_pioggia.get(nome_stazione) # Preso dal foglio
                    current_hour = current_time.hour

                    if valore_precedente_float is None:
                        # Nessun dato precedente valido o prima esecuzione
                        if current_hour == 0:
                             # Alla mezzanotte, il valore attuale E' la pioggia dall'inizio del giorno (prima ora)
                             pioggia_ora_calcolata = pioggia_tot_oggi_attuale
                        else:
                             # Non possiamo calcolare la differenza per la prima ora (a meno che non sia mezzanotte)
                             # Potremmo mettere 0.0 o N/A. 0.0 è più utile per somme.
                             pioggia_ora_calcolata = 0.0
                    elif current_hour == 0:
                         # E' mezzanotte (o poco dopo), il valore letto è la pioggia della prima ora.
                         pioggia_ora_calcolata = pioggia_tot_oggi_attuale
                    elif pioggia_tot_oggi_attuale < valore_precedente_float:
                         # Reset del contatore API o errore? In questo caso la pioggia *oraria* è 0
                         # o potenzialmente il valore attuale se il reset è avvenuto nell'ora.
                         # Per sicurezza, mettiamo 0.0 per l'incremento orario.
                         pioggia_ora_calcolata = 0.0
                         print(f"WARN: Rilevato reset o calo pioggia per {nome_stazione} (prec: {valore_precedente_float}, attuale: {pioggia_tot_oggi_attuale}). Pioggia oraria impostata a 0.")
                    else:
                         # Calcolo standard della differenza oraria
                         pioggia_ora_calcolata = round(pioggia_tot_oggi_attuale - valore_precedente_float, 2) # Arrotonda a 2 decimali

                    # Assicurati che sia float se numerico
                    if isinstance(pioggia_ora_calcolata, (int, float)):
                         dati_per_riga[pioggia_ora_key] = float(pioggia_ora_calcolata)
                    else: # Altrimenti resta 'N/A'
                         dati_per_riga[pioggia_ora_key] = 'N/A'

                else:
                    # Se pioggia_tot_oggi_attuale non era valido, anche pioggia oraria è N/A
                    dati_per_riga[pioggia_ora_key] = 'N/A'

                print(f"  -> Pioggia Ora Calcolata: {dati_per_riga[pioggia_ora_key]} mm")


        if not dati_per_riga:
            print("Nessun dato valido estratto per le stazioni selezionate.")
            sys.exit(0) # Esce senza errori, ma non scrive nulla

        # --- Preparazione e Scrittura Riga nel Foglio ---

        # Costruisci l'intestazione completa basata sui dati trovati ORA
        nuova_intestazione_ordinata = ['Data_Ora'] + sorted(intestazioni_nuove_trovate.keys())

        # Verifica se l'intestazione nel foglio corrisponde a quella attuale
        if not intestazione_attuale or nuova_intestazione_ordinata != intestazione_attuale:
            print("Intestazioni cambiate o foglio vuoto/danneggiato. Aggiornando/Ricreando intestazioni...")
            try:
                # Soluzione semplice: cancella e riscrivi intestazione. ATTENZIONE: Perde dati se le colonne cambiano ordine!
                # Una soluzione più complessa gestirebbe aggiunta/rimozione colonne senza cancellare.
                foglio.clear()
                foglio.append_row(nuova_intestazione_ordinata, value_input_option='USER_ENTERED')
                intestazione_attuale = nuova_intestazione_ordinata # Aggiorna riferimento interno
                print("Intestazioni scritte.")
                # Dopo aver cambiato le intestazioni, la prossima esecuzione leggerà questa nuova struttura.
            except Exception as e:
                print(f"Errore CRITICO durante la scrittura delle nuove intestazioni: {e}")
                sys.exit(f"Impossibile scrivere le intestazioni nel foglio: {nome_foglio}. Uscita.")

        # Costruisci la riga di dati nell'ordine definito dall'intestazione ATTUALE del foglio
        riga_dati_finale = []
        for header_col in intestazione_attuale:
            if header_col == 'Data_Ora':
                riga_dati_finale.append(formatted_time)
            else:
                # Prendi il valore dai dati processati, usa 'N/A' se mancante per questa colonna specifica
                valore = dati_per_riga.get(header_col, 'N/A')

                # Assicurati che i float siano float, il resto stringa ('N/A' incluso)
                if isinstance(valore, (int, float)):
                     riga_dati_finale.append(float(valore))
                else:
                     riga_dati_finale.append(str(valore)) # Invia 'N/A' o altri non-numerici come stringhe


        # Aggiungi la riga di dati al foglio
        try:
            foglio.append_row(riga_dati_finale, value_input_option='USER_ENTERED')
            print(f"\nDati meteo aggiunti con successo al foglio Google '{nome_foglio}'")
        except Exception as e:
            print(f"Errore durante l'aggiunta della riga di dati al foglio: {e}")
            # Potresti voler implementare un tentativo di retry qui
            sys.exit(f"Impossibile scrivere i dati nel foglio: {nome_foglio}. Uscita.")


    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta API: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Errore nel parsing JSON dalla risposta API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Errore generico durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc() # Stampa più dettagli sull'errore
        sys.exit(1)

if __name__ == "__main__":
    estrai_dati_meteo()
