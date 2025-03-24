import requests
import json
import time
from datetime import datetime, timedelta
import sys
import urllib3
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def estrai_dati_meteo():
    """
    Extracts weather data from API and appends it to a Google Sheet.
    This function runs continuously, updating the sheet every hour.
    """
    # Google Sheets setup
    nome_foglio = "Dati Meteo Stazioni"
    credenziali_path = "credentials.json"  # Inserisci il percorso del tuo file di credenziali
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    try:
        credenziali = ServiceAccountCredentials.from_json_keyfile_name(credenziali_path, scope)
        client = gspread.authorize(credenziali)
        try:
            foglio = client.open(nome_foglio).sheet1
            print(f"Foglio Google '{nome_foglio}' aperto con successo.")
        except gspread.SpreadsheetNotFound:
            foglio = client.create(nome_foglio).sheet1
            print(f"Foglio Google '{nome_foglio}' creato con successo.")
    except Exception as e:
        print(f"Errore nell'autenticazione a Google Sheets: {e}")
        sys.exit(1)

    # API information
    api_url = "https://retemir.regione.marche.it/api/stations/rt-data"
    stazioni_interessate = [
        "Misa",
        "Pianello di Ostra",
        "Nevola",
        "Barbara",
        "Serra dei Conti",
        "Arcevia",
        "Corinaldo" # Added Corinaldo
    ]
    sensori_interessati_tipoSens = [0, 1, 5, 6, 9, 10, 100]

    intestazione_attuale = []
    try:
        intestazione_attuale = foglio.row_values(1)
        print(f"Intestazioni esistenti: {intestazione_attuale}")
    except:
        intestazione_attuale = []

    valori_precedenti_pioggia = {}

    print(f"Iniziando il monitoraggio dei dati meteo. I dati saranno salvati nel foglio Google: {nome_foglio}")
    print(f"Il programma aggiornerà i dati ogni ora. Premi Ctrl+C per terminare.")

    try:
        while True:
            current_time = datetime.now()
            formatted_time = current_time.strftime('%d/%m/%Y %H:%M')
            print(f"\nEstrazione dati alle {formatted_time}...")

            try:
                response = requests.get(api_url, verify=False, timeout=30)
                response.raise_for_status()
                dati_meteo = response.json()

                if credenziali.access_token_expired:
                    client.login()
                    foglio = client.open(nome_foglio).sheet1

                intestazioni_per_stazione = {}
                dati_per_stazione = {}
                pioggia_ora_per_stazione = {}  # Changed to hourly rain

                for stazione in dati_meteo:
                    nome_stazione = stazione.get("nome")
                    if nome_stazione in stazioni_interessate:
                        timestamp_stazione = stazione.get("lastUpdateTime")

                        try:
                            if isinstance(timestamp_stazione, str) and len(timestamp_stazione) >= 16:
                                timestamp_formattato = timestamp_stazione
                            else:
                                dt_obj = datetime.fromisoformat(timestamp_stazione.replace('Z', '+00:00'))
                                timestamp_formattato = dt_obj.strftime('%d/%m/%Y %H:%M')
                        except (ValueError, AttributeError):
                            timestamp_formattato = formatted_time

                        print(f"Dati per la stazione: {nome_stazione}")

                        for sensore in stazione.get("analog", []):
                            tipoSens = sensore.get("tipoSens")
                            if tipoSens in sensori_interessati_tipoSens:
                                descr_sensore = sensore.get("descr", "").strip()
                                valore_sensore = sensore.get("valore")
                                unita_misura = sensore.get("unmis", "").strip() if sensore.get("unmis") else ""

                                intestazione = f"{nome_stazione} - {descr_sensore} ({unita_misura})"
                                intestazioni_per_stazione[intestazione] = True
                                dati_per_stazione[intestazione] = valore_sensore

                                if "Pioggia TOT Oggi" in descr_sensore and valore_sensore is not None:
                                    pioggia_key = f"{nome_stazione} - Pioggia Ora (mm)" # Changed key
                                    valore_precedente = valori_precedenti_pioggia.get(nome_stazione, 0)

                                    if isinstance(valore_sensore, (int, float)) and isinstance(valore_precedente, (int, float)):
                                        current_hour = current_time.hour
                                        
                                        # Reset at the start of each hour *after* the first hour
                                        if current_hour == 0 :
                                            pioggia_ora = valore_sensore
                                        
                                        elif valore_sensore < valore_precedente:
                                           pioggia_ora = valore_sensore
                                        
                                        else:
                                            pioggia_ora = valore_sensore - valore_precedente  # Hourly difference

                                    else:
                                        pioggia_ora = 0

                                    valori_precedenti_pioggia[nome_stazione] = valore_sensore
                                    intestazioni_per_stazione[pioggia_key] = True
                                    pioggia_ora_per_stazione[pioggia_key] = pioggia_ora # store hourly rain

                                    print(f"  - {descr_sensore}: {valore_sensore} {unita_misura}")
                                    print(f"  - Pioggia Ora: {pioggia_ora} mm")  # Corrected label
                                else:
                                    print(f"  - {descr_sensore}: {valore_sensore} {unita_misura}")

                        print("-" * 30)

                dati_per_stazione.update(pioggia_ora_per_stazione) # add hourly rain data

                if not dati_per_stazione:
                    print("Nessun dato disponibile per le stazioni selezionate. Riproverò al prossimo aggiornamento.")
                    time.sleep(60 * 60)  # Wait 1 hour
                    continue

                nuova_intestazione = ['Data_Ora']
                nuova_intestazione.extend(sorted(intestazioni_per_stazione.keys()))

                headers_changed = False
                if not intestazione_attuale or nuova_intestazione != intestazione_attuale:
                    headers_changed = True
                    print("Le intestazioni sono cambiate o il foglio è vuoto. Aggiornando intestazioni...")
                    foglio.clear()
                    foglio.append_row(nuova_intestazione)
                    intestazione_attuale = nuova_intestazione

                riga_dati = [formatted_time]
                for intestazione in nuova_intestazione[1:]:
                    valore = dati_per_stazione.get(intestazione, 'N/A')
                    riga_dati.append(valore)

                foglio.append_row(riga_dati)
                print(f"Dati meteo aggiunti al foglio Google '{nome_foglio}'")

            except requests.exceptions.RequestException as e:
                print(f"Errore nella richiesta API: {e}")
            except json.JSONDecodeError as e:
                print(f"Errore nel parsing JSON: {e}")
            except Exception as e:
                print(f"Errore generico: {e}")

            # Wait for 1 hour before the next update
            next_update = datetime.now() + timedelta(hours=1)
            print(f"Prossimo aggiornamento alle {next_update.strftime('%H:%M')}")
            time.sleep(60 * 60)  # 1 hour in seconds

    except KeyboardInterrupt:
        print("\nProgramma terminato dall'utente.")
        sys.exit(0)

if __name__ == "__main__":
    estrai_dati_meteo()