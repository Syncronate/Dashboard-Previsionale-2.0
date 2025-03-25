import requests
import json
import time
from datetime import datetime, timedelta
import sys
import urllib3
import gspread
from google.oauth2 import service_account
import os

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def estrai_dati_meteo():
    """
    Extracts weather data from API and appends it to a Google Sheet.
    Modified to run once per execution for GitHub Actions.
    """
    # Google Sheets setup
    nome_foglio = "Dati Meteo Stazioni"
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # State to track previous rainfall values (with persistence between runs)
    rainfall_state_file = 'rainfall_state.json'

    try:
        # Load previous rainfall state
        previous_rainfall_values = {}
        if os.path.exists(rainfall_state_file):
            with open(rainfall_state_file, 'r') as f:
                previous_rainfall_values = json.load(f)

        # Load credentials from environment variable
        credentials_json = os.environ.get("GCP_CREDENTIALS_JSON")
        if not credentials_json:
            raise EnvironmentError("GCP_CREDENTIALS_JSON environment variable not found.")
        
        credentials_info = json.loads(credentials_json)
        scoped_credentials = service_account.Credentials.from_service_account_info(
            credentials_info, scopes=scope
        )
        client = gspread.authorize(scoped_credentials)

        try:
            foglio = client.open(nome_foglio).sheet1
            print(f"Google Sheet '{nome_foglio}' opened successfully.")
        except gspread.SpreadsheetNotFound:
            foglio = client.create(nome_foglio).sheet1
            print(f"Google Sheet '{nome_foglio}' created successfully.")
    except Exception as e:
        print(f"Error authenticating to Google Sheets: {e}")
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
        "Corinaldo",
        "Ponte Garibaldi",
    ]
    sensori_interessati_tipoSens = [0, 1, 5, 6, 9, 10, 100, 101]

    try:
        # Fetch existing headers
        try:
            intestazione_attuale = foglio.row_values(1)
            print(f"Existing headers: {intestazione_attuale}")
        except:
            intestazione_attuale = []

        # Prepare for data extraction
        current_time = datetime.now()
        formatted_time = current_time.strftime('%d/%m/%Y %H:%M')
        print(f"\nExtracting data at {formatted_time}...")

        # Fetch weather data
        try:
            response = requests.get(api_url, verify=False, timeout=30)
            response.raise_for_status()
            dati_meteo = response.json()

            intestazioni_per_stazione = {}
            dati_per_stazione = {}
            pioggia_ora_per_stazione = {}

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

                    print(f"Data for station: {nome_stazione}")

                    for sensore in stazione.get("analog", []):
                        tipoSens = sensore.get("tipoSens")
                        if tipoSens in sensori_interessati_tipoSens:
                            descr_sensore = sensore.get("descr", "").strip()
                            valore_sensore = sensore.get("valore")
                            unita_misura = sensore.get("unmis", "").strip() if sensore.get("unmis") else ""

                            intestazione = f"{nome_stazione} - {descr_sensore} ({unita_misura})"
                            intestazioni_per_stazione[intestazione] = True
                            dati_per_stazione[intestazione] = valore_sensore

                            # Improved rainfall calculation
                            if "Pioggia TOT Oggi" in descr_sensore and valore_sensore is not None:
                                pioggia_key = f"{nome_stazione} - Pioggia Ora (mm)"
                                
                                # Get previous rainfall value, defaulting to 0 if not found
                                valore_precedente = previous_rainfall_values.get(nome_stazione, 0)

                                # Validate input types
                                if isinstance(valore_sensore, (int, float)) and isinstance(valore_precedente, (int, float)):
                                    # Calculate rainfall amount
                                    if valore_sensore >= valore_precedente:
                                        # Normal case: current value is greater or equal to previous
                                        pioggia_ora = valore_sensore - valore_precedente
                                    else:
                                        # Case where counter might have reset or been recalibrated
                                        # Use the current total rainfall as the amount
                                        pioggia_ora = valore_sensore
                                else:
                                    # Handle case of invalid types
                                    pioggia_ora = 0

                                # Update previous rainfall value for next iteration
                                previous_rainfall_values[nome_stazione] = valore_sensore

                                intestazioni_per_stazione[pioggia_key] = True
                                pioggia_ora_per_stazione[pioggia_key] = round(pioggia_ora, 2)

            # Update local state file with current rainfall values
            with open(rainfall_state_file, 'w') as f:
                json.dump(previous_rainfall_values, f)

            dati_per_stazione.update(pioggia_ora_per_stazione)

            if not dati_per_stazione:
                print("No data available for selected stations.")
                return

            # Prepare headers
            nuova_intestazione = ['Data_Ora']
            nuova_intestazione.extend(sorted(intestazioni_per_stazione.keys()))

            # Update headers if needed
            if not intestazione_attuale or nuova_intestazione != intestazione_attuale:
                print("Updating headers...")
                foglio.clear()
                foglio.append_row(nuova_intestazione)

            # Prepare and append data row
            riga_dati = [formatted_time]
            for intestazione in nuova_intestazione[1:]:
                valore = dati_per_stazione.get(intestazione, 'N/A')
                riga_dati.append(valore)

            foglio.append_row(riga_dati)
            print(f"Weather data added to Google Sheet '{nome_foglio}'")

        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        except Exception as e:
            print(f"Generic error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    estrai_dati_meteo()
