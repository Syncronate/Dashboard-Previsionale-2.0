import os
import json
# import base64 # Non più necessario per le credenziali se usi il file
from datetime import datetime, timedelta
import pytz
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build # NUOVA IMPORTAZIONE
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Costanti (adattale se necessario) ---
MODELS_DIR = "models"
MODEL_BASE_NAME = "modello_lstm_20250623_2039" # <<< MODIFICATO
GSHEET_ID = os.environ.get("GSHEET_ID")
GSHEET_DATA_SHEET_NAME = os.environ.get("GSHEET_DATA_SHEET_NAME", "Sheet1")
GSHEET_PREDICTIONS_SHEET_NAME = os.environ.get("GSHEET_PREDICTIONS_SHEET_NAME", "Previsioni Idrometri") # <<< MODIFICATO
GSHEET_DATE_COL_INPUT = 'Data_Ora'
GSHEET_DATE_FORMAT_INPUT = '%d/%m/%Y %H:%M'
HUMIDITY_COL_NAME_INPUT = "Umidita' Sensore 3452 (Montemurello)"

italy_tz = pytz.timezone('Europe/Rome')

# --- Definizione Modello LSTM (invariata) ---
class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(out.size(0), self.output_window, self.output_size)
        return out

# --- Funzioni Utilità (load_model_and_scalers, fetch_input_data_from_gsheet, predict_with_model - invariate) ---
def load_model_and_scalers(model_base_name, models_dir):
    """Carica il modello, la configurazione e gli scaler."""
    config_path = os.path.join(models_dir, f"{model_base_name}.json")
    model_path = os.path.join(models_dir, f"{model_base_name}.pth")
    scaler_features_path = os.path.join(models_dir, f"{model_base_name}_features.joblib")
    scaler_targets_path = os.path.join(models_dir, f"{model_base_name}_targets.joblib")

    if not all(os.path.exists(p) for p in [config_path, model_path, scaler_features_path, scaler_targets_path]):
        raise FileNotFoundError(f"Uno o più file per il modello '{model_base_name}' non trovati in '{models_dir}'.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if "feature_columns" not in config:
        config["feature_columns"] = [
            'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
            'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
            HUMIDITY_COL_NAME_INPUT,
            'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
        ]
        print(f"Warning: 'feature_columns' non trovate nella config. Usate quelle definite nello script: {config['feature_columns']}")

    input_size = len(config["feature_columns"])
    output_size = len(config["target_columns"])

    model = HydroLSTM(
        input_size,
        config["hidden_size"],
        output_size,
        config["output_window"],
        config["num_layers"],
        config["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler_features = joblib.load(scaler_features_path)
    scaler_targets = joblib.load(scaler_targets_path)

    print(f"Modello '{model_base_name}' e scaler caricati su {device}.")
    return model, scaler_features, scaler_targets, config, device

def fetch_input_data_from_gsheet(gc, sheet_id, data_sheet_name, config, column_mapping):
    """Recupera e prepara i dati di input dal Google Sheet."""
    input_window_steps = config["input_window"]
    model_feature_columns = config["feature_columns"]

    sh = gc.open_by_key(sheet_id)
    worksheet = sh.worksheet(data_sheet_name)

    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        raise ValueError(f"Foglio Google '{data_sheet_name}' vuoto o con solo intestazione.")

    headers_gsheet = all_values[0]
    # Recupera più dati per avere margine per ffill/bfill e per input_window
    num_rows_to_fetch = input_window_steps + 20 # Aumentato per sicurezza
    start_index = max(1, len(all_values) - num_rows_to_fetch)
    data_rows = all_values[start_index:]
    df_gsheet_raw = pd.DataFrame(data_rows, columns=headers_gsheet)

    required_gsheet_cols_from_mapping = [col for col in column_mapping.keys() if col != GSHEET_DATE_COL_INPUT]

    cols_to_select_in_gsheet = []
    if GSHEET_DATE_COL_INPUT in headers_gsheet:
        cols_to_select_in_gsheet.append(GSHEET_DATE_COL_INPUT)
    else:
        raise ValueError(f"Colonna data input '{GSHEET_DATE_COL_INPUT}' non trovata nel foglio GSheet '{data_sheet_name}'.")

    for gsheet_col_name in required_gsheet_cols_from_mapping:
        if gsheet_col_name in headers_gsheet:
            cols_to_select_in_gsheet.append(gsheet_col_name)
        else:
            print(f"Attenzione: Colonna GSheet '{gsheet_col_name}' specificata nel mapping ma non trovata nel foglio '{data_sheet_name}'.")

    df_subset = df_gsheet_raw[list(set(cols_to_select_in_gsheet))].copy()
    df_mapped = df_subset.rename(columns=column_mapping)
    date_col_model_name = column_mapping.get(GSHEET_DATE_COL_INPUT, GSHEET_DATE_COL_INPUT)
    latest_valid_timestamp = None

    for col in df_mapped.columns:
        if col == date_col_model_name:
            try:
                df_mapped[col] = pd.to_datetime(df_mapped[col], format=GSHEET_DATE_FORMAT_INPUT, errors='coerce')
                if not df_mapped[col].empty and df_mapped[col].notna().any():
                    if df_mapped[col].dt.tz is None:
                        df_mapped[col] = df_mapped[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                    else:
                        df_mapped[col] = df_mapped[col].dt.tz_convert(italy_tz)
            except Exception as e_date:
                print(f"Errore conversione data per colonna '{col}': {e_date}. Sarà NaT.")
                df_mapped[col] = pd.NaT
        elif col in model_feature_columns: # Processa solo colonne feature del modello per la conversione numerica
            try:
                # Sostituisci virgola con punto per decimali e gestisci 'N/A' o stringhe vuote
                if pd.api.types.is_object_dtype(df_mapped[col]) or pd.api.types.is_string_dtype(df_mapped[col]):
                    col_str = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    df_mapped[col] = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
            except Exception as e_num:
                print(f"Errore conversione numerica per colonna '{col}': {e_num}. Sarà NaN.")
                df_mapped[col] = np.nan

    if date_col_model_name in df_mapped.columns and pd.api.types.is_datetime64_any_dtype(df_mapped[date_col_model_name]):
        df_mapped = df_mapped.sort_values(by=date_col_model_name)
        valid_dates = df_mapped[date_col_model_name].dropna()
        if not valid_dates.empty:
            latest_valid_timestamp = valid_dates.iloc[-1]
    else:
        raise ValueError(f"Colonna data '{date_col_model_name}' non trovata o non valida dopo mappatura/conversione.")

    df_features_selected = pd.DataFrame()
    for m_col in model_feature_columns:
        if m_col in df_mapped.columns:
            df_features_selected[m_col] = df_mapped[m_col]
        else:
            print(f"Attenzione: Colonna feature modello '{m_col}' non presente in df_mapped. Sarà riempita con NaN.")
            df_features_selected[m_col] = np.nan # Crea la colonna con NaN

    # Assicura che le colonne siano nell'ordine corretto e fai fill
    df_features_filled = df_features_selected[model_feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)

    if len(df_features_filled) < input_window_steps:
        raise ValueError(f"Dati insufficienti nel GSheet ({len(df_features_filled)} righe valide) per l'input del modello (richiesti {input_window_steps}).")

    input_data = df_features_filled.iloc[-input_window_steps:]

    for col in input_data.columns:
        if not pd.api.types.is_numeric_dtype(input_data[col]):
            raise TypeError(f"La colonna '{col}' nei dati di input finali non è numerica (tipo: {input_data[col].dtype}).")

    print(f"Dati di input recuperati e processati. Shape: {input_data.shape}")
    print(f"Ultimo timestamp valido usato per i dati di input: {latest_valid_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if latest_valid_timestamp else 'N/A'}")
    return input_data.values, latest_valid_timestamp

def predict_with_model(model, input_data_np, scaler_features, scaler_targets, device):
    """Esegue la previsione."""
    input_normalized = scaler_features.transform(input_data_np)
    input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_np = output_tensor.cpu().numpy().squeeze(0)
    predictions_scaled_back = scaler_targets.inverse_transform(output_np)
    return predictions_scaled_back

# --- MODIFICATA ---
def append_predictions_to_gsheet(gc, sheet_id_str, predictions_sheet_name, predictions_np, target_columns, prediction_start_time, config):
    """Aggiunge le previsioni e un grafico a un foglio Google."""
    sh = gc.open_by_key(sheet_id_str)
    worksheet_created = False
    try:
        worksheet = sh.worksheet(predictions_sheet_name)
        print(f"Foglio '{predictions_sheet_name}' trovato. Cancello il contenuto precedente.")
        worksheet.clear() # Cancella tutto il contenuto per ricominciare
        worksheet_created = False # Non è stata creata ora, ma esisteva e l'abbiamo svuotata
    except gspread.exceptions.WorksheetNotFound:
        print(f"Foglio '{predictions_sheet_name}' non trovato. Creazione in corso...")
        worksheet = sh.add_worksheet(title=predictions_sheet_name, rows=config["output_window"] + 10, cols=len(target_columns) + 10)
        worksheet_created = True

    # Intestazione
    header_parts_targets = [f"Previsto: {target_col.split('[')[0].strip()}" for target_col in target_columns]

    # ***** INIZIO BLOCCO MODIFICATO *****
    # Per maggiore chiarezza nel foglio di calcolo, cambiamo il nome della colonna
    # da "Passo (Relativo)" a "Ora Previsione", dato che mostreremo solo l'orario.
    # Se vuoi mantenere il vecchio nome, commenta la riga "header_row" modificata
    # e de-commenta quella originale.
    header_row = ["Timestamp Esecuzione", "Timestamp Inizio Serie", "Ora Previsione"] + header_parts_targets
    # header_row = ["Timestamp Esecuzione", "Timestamp Inizio Serie", "Passo (Relativo)"] + header_parts_targets # RIGA ORIGINALE
    # ***** FINE BLOCCO MODIFICATO *****

    worksheet.append_row(header_row, value_input_option='USER_ENTERED')
    print(f"Intestazione aggiunta al foglio '{predictions_sheet_name}'.")

    output_window_steps = config["output_window"]
    timestamp_esecuzione_dt = datetime.now(italy_tz)
    timestamp_esecuzione_str = timestamp_esecuzione_dt.strftime('%Y-%m-%d %H:%M:%S %Z')

    if prediction_start_time is None:
        print("Attenzione: prediction_start_time è None. Userò il timestamp di esecuzione meno la durata dell'input window come stima.")
        input_duration_minutes = config["input_window"] * 30 # Assumendo 30 min per step di input
        prediction_start_time = timestamp_esecuzione_dt - timedelta(minutes=input_duration_minutes)

    prediction_start_time_str = prediction_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')

    rows_to_append = []
    for step_idx in range(output_window_steps):
        current_prediction_time_dt = prediction_start_time + timedelta(minutes=30 * (step_idx + 1))
        # ***** INIZIO BLOCCO MODIFICATO *****
        # Modifichiamo il formato della colonna per mostrare solo hh:mm.
        row_data_for_step = [
            timestamp_esecuzione_str,
            prediction_start_time_str,
            current_prediction_time_dt.strftime('%H:%M') # MODIFICA PRINCIPALE: formato cambiato a hh:mm
        ]
        # ***** FINE BLOCCO MODIFICATO *****

        for target_idx in range(predictions_np.shape[1]):
            predicted_value = predictions_np[step_idx, target_idx]
            formatted_value_str = f"{predicted_value:.3f}".replace('.', ',')
            row_data_for_step.append(formatted_value_str)
        rows_to_append.append(row_data_for_step)

    if rows_to_append:
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        print(f"Aggiunte {len(rows_to_append)} righe di previsione al foglio '{predictions_sheet_name}'.")

        # --- AGGIUNTA GRAFICO ---
        try:
            print("Tentativo di aggiungere/aggiornare il grafico...")
            service = build('sheets', 'v4', credentials=gc.auth)
            spreadsheet_id = sh.id
            sheet_id_numeric = worksheet.id

            sheet_info = service.spreadsheets().get(spreadsheetId=spreadsheet_id, fields='sheets(properties,charts)').execute()
            delete_chart_requests = []
            for s_info in sheet_info.get('sheets', []):
                if s_info.get('properties', {}).get('sheetId') == sheet_id_numeric:
                    for chart in s_info.get('charts', []):
                        delete_chart_requests.append({
                            "deleteEmbeddedObject": {
                                "objectId": chart['chartId']
                            }
                        })
                    break
            if delete_chart_requests:
                service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={"requests": delete_chart_requests}
                ).execute()
                print(f"Eliminati {len(delete_chart_requests)} grafici esistenti dal foglio '{predictions_sheet_name}'.")

            domain_column_index = 2 # Colonna C: "Ora Previsione"
            first_series_column_index = 3
            start_row_index_api = 1
            end_row_index_api = start_row_index_api + len(rows_to_append)
            chart_series_requests = []

            for i, target_name in enumerate(header_parts_targets):
                series_column_idx_api = first_series_column_index + i
                chart_series_requests.append({
                    "series": {
                        "sourceRange": {
                            "sources": [{
                                "sheetId": sheet_id_numeric,
                                "startRowIndex": start_row_index_api,
                                "endRowIndex": end_row_index_api,
                                "startColumnIndex": series_column_idx_api,
                                "endColumnIndex": series_column_idx_api + 1
                            }]
                        }
                    },
                    "targetAxis": "LEFT_AXIS",
                })

            chart_request = {
                "addChart": {
                    "chart": {
                        "spec": {
                            "title": f"Previsioni Modello ({timestamp_esecuzione_str})",
                            "basicChart": {
                                "chartType": "LINE",
                                "legendPosition": "BOTTOM_LEGEND",
                                "axis": [
                                    { "position": "BOTTOM_AXIS", "title": "Ora della Previsione" }, # Titolo asse X aggiornato
                                    { "position": "LEFT_AXIS", "title": "Valore Previsto" }
                                ],
                                "domains": [{
                                    "domain": {
                                        "sourceRange": {
                                            "sources": [{
                                                "sheetId": sheet_id_numeric,
                                                "startRowIndex": start_row_index_api,
                                                "endRowIndex": end_row_index_api,
                                                "startColumnIndex": domain_column_index,
                                                "endColumnIndex": domain_column_index + 1
                                            }]
                                        }
                                    }
                                }],
                                "series": chart_series_requests,
                                "headerCount": 0
                            }
                        },
                        "position": {
                            "overlayPosition": {
                                "anchorCell": {
                                    "sheetId": sheet_id_numeric,
                                    "rowIndex": 0,
                                    "columnIndex": len(header_row) + 1
                                },
                                "offsetXPixels": 10, "offsetYPixels": 10,
                                "widthPixels": 650, "heightPixels": 400
                            }
                        }
                    }
                }
            }

            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": [chart_request]}
            ).execute()
            print(f"Grafico aggiunto/aggiornato con successo al foglio '{predictions_sheet_name}'.")

        except Exception as e_chart:
            print(f"Errore durante la creazione/aggiornamento del grafico: {e_chart}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Nessuna riga di previsione da aggiungere al foglio '{predictions_sheet_name}'.")

def main():
    print(f"Avvio simulazione script alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if not GSHEET_ID:
        print("Errore: GSHEET_ID non impostato come variabile d'ambiente.")
        return

    try:
        credentials_path = "credentials.json"
        if not os.path.exists(credentials_path):
            gcp_sa_key_b64 = os.environ.get("GCP_SA_KEY_BASE64_FALLBACK")
            if gcp_sa_key_b64:
                import base64
                print("Uso GCP_SA_KEY_BASE64_FALLBACK per le credenziali (test locale).")
                credentials_json_str = base64.b64decode(gcp_sa_key_b64).decode('utf-8')
                credentials_dict = json.loads(credentials_json_str)
                credentials = Credentials.from_service_account_info(credentials_dict,
                    scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
            else:
                 raise FileNotFoundError(f"File '{credentials_path}' non trovato e GCP_SA_KEY_BASE64_FALLBACK non impostato.")
        else:
            credentials = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            )
        gc = gspread.authorize(credentials)
        print("Autenticazione Google Sheets riuscita.")

        model, scaler_features, scaler_targets, config, device = load_model_and_scalers(MODEL_BASE_NAME, MODELS_DIR)

        column_mapping_gsheet_to_model = {
            'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
            'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
            'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
            'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
            HUMIDITY_COL_NAME_INPUT: HUMIDITY_COL_NAME_INPUT,
            'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
            GSHEET_DATE_COL_INPUT: GSHEET_DATE_COL_INPUT
        }

        missing_features_in_mapping = []
        for fc in config["feature_columns"]:
            if fc not in column_mapping_gsheet_to_model.values():
                 missing_features_in_mapping.append(fc)

        if missing_features_in_mapping:
            print(f"ATTENZIONE CRITICA: Le seguenti feature del modello NON sono mappate da alcuna colonna GSheet nel 'column_mapping_gsheet_to_model': {missing_features_in_mapping}")
            print("Queste colonne saranno riempite con NaN e poi con 0, il che potrebbe portare a previsioni errate.")

        input_data_np, last_input_timestamp = fetch_input_data_from_gsheet(
            gc, GSHEET_ID, GSHEET_DATA_SHEET_NAME, config, column_mapping_gsheet_to_model
        )

        predictions_np = predict_with_model(model, input_data_np, scaler_features, scaler_targets, device)
        print(f"Previsioni generate. Shape: {predictions_np.shape}")

        prediction_start_for_series = last_input_timestamp

        append_predictions_to_gsheet(
            gc, GSHEET_ID, GSHEET_PREDICTIONS_SHEET_NAME, predictions_np,
            config["target_columns"], prediction_start_for_series, config
        )

        print(f"Simulazione completata con successo alle {datetime.now(italy_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    except FileNotFoundError as e:
        print(f"Errore File: {e}")
    except ValueError as e:
        print(f"Errore Valore: {e}")
    except TypeError as e:
        print(f"Errore Tipo: {e}")
    except gspread.exceptions.APIError as e:
        print(f"Errore API Google Sheets (gspread): {e}")
    except Exception as e:
        print(f"Errore imprevisto durante la simulazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
