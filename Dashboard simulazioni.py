import streamlit as st
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta, time # Aggiunto time
import joblib
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go
import time as pytime # Rinominato per evitare conflitti con datetime.time
import gspread # Importa gspread
from google.oauth2.service_account import Credentials # Importa Credentials
import re # Importa re per espressioni regolari
import json # Importa json per leggere/scrivere file di configurazione
import glob # Per cercare i file dei modelli
import traceback # Per stampare errori dettagliati
import random
import mimetypes # Per indovinare i tipi MIME per i download
from streamlit_js_eval import streamlit_js_eval # Per forzare refresh periodico
import pytz # Per gestione timezone
import math # Per calcoli matematici (potenza)

# --- Configurazione Pagina Streamlit ---
st.set_page_config(page_title="Modello Predittivo Idrologico", layout="wide")

# --- Costanti Globali ---
MODELS_DIR = "models"
DEFAULT_DATA_PATH = "dati_idro.csv"
GSHEET_ID = "1pQI6cKrrT-gcVAfl-9ZhUx5b3J-edZRRj6nzDcCBRcA" # ID Foglio per Dashboard e Simulazione Base
GSHEET_DATE_COL = 'Data_Ora'
GSHEET_DATE_FORMAT = '%d/%m/%Y %H:%M'
GSHEET_RELEVANT_COLS = [
    GSHEET_DATE_COL,
    'Arcevia - Pioggia Ora (mm)', 'Barbara - Pioggia Ora (mm)', 'Corinaldo - Pioggia Ora (mm)',
    'Misa - Pioggia Ora (mm)', 'Umidita\' Sensore 3452 (Montemurello)', # Assicurati che sia nel foglio!
    'Serra dei Conti - Livello Misa (mt)', 'Pianello di Ostra - Livello Misa (m)',
    'Nevola - Livello Nevola (mt)', 'Misa - Livello Misa (mt)',
    'Ponte Garibaldi - Livello Misa 2 (mt)'
]
DASHBOARD_REFRESH_INTERVAL_SECONDS = 300
DASHBOARD_HISTORY_ROWS = 48 # 24 ore (48 step da 30 min) - Usato per vista default
DEFAULT_THRESHOLDS = { # Soglie USATE NELLA DASHBOARD
    'Arcevia - Pioggia Ora (mm)': 2.0, 'Barbara - Pioggia Ora (mm)': 2.0, 'Corinaldo - Pioggia Ora (mm)': 2.0,
    'Misa - Pioggia Ora (mm)': 2.0, 'Umidita\' Sensore 3452 (Montemurello)': 95.0,
    'Serra dei Conti - Livello Misa (mt)': 1.7, 'Pianello di Ostra - Livello Misa (m)': 2.0,
    'Nevola - Livello Nevola (mt)': 2.5, 'Misa - Livello Misa (mt)': 2.0,
    'Ponte Garibaldi - Livello Misa 2 (mt)': 2.2
}

# --- NUOVE COSTANTI PER SOGLIE SIMULAZIONE (ATTENZIONE/ALLERTA) ---
SIMULATION_THRESHOLDS = {
    # Livelli GSheet Style
    'Serra dei Conti - Livello Misa (mt)': {'attenzione': 1.2, 'allerta': 1.7},
    'Pianello di Ostra - Livello Misa (m)': {'attenzione': 1.5, 'allerta': 2.0},
    'Nevola - Livello Nevola (mt)': {'attenzione': 2.0, 'allerta': 2.5},
    'Misa - Livello Misa (mt)': {'attenzione': 1.5, 'allerta': 2.0},
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'attenzione': 1.5, 'allerta': 2.2},
    # Livelli CSV/Internal Style
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)': {'attenzione': 1.2, 'allerta': 1.7}, # Aggiornato per coerenza
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)': {'attenzione': 1.5, 'allerta': 2.0}, # Aggiornato per coerenza
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)': {'attenzione': 2.0, 'allerta': 2.5}, # Aggiornato per coerenza
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)': {'attenzione': 1.5, 'allerta': 2.0}, # Aggiornato per coerenza
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'attenzione': 1.5, 'allerta': 2.2}  # Aggiornato per coerenza
}

# --- NUOVE COSTANTI PER LA FRASE DI ATTRIBUZIONE ---
ATTRIBUTION_PHRASE = "Previsione progettata ed elaborata da Alberto Bussaglia del Comune di Senigallia. La previsione è di tipo probabilistico e non rappresenta una certezza.<br> Per maggiori informazioni, o per ottenere il consenso all'utilizzo/pubblicazione, contattare l'autore."
ATTRIBUTION_PHRASE_FILENAME_SUFFIX = "_da_Alberto_Bussaglia_Area_PC"

italy_tz = pytz.timezone('Europe/Rome')
HUMIDITY_COL_NAME = "Umidita' Sensore 3452 (Montemurello)"

# --- NUOVA COSTANTE: SCALE DI DEFLUSSO ---
RATING_CURVES = {
    '1112': [ # Misa - Livello Misa (mt) / Bettolelle
        {'min_H': 0.73, 'max_H': 1.3,  'a': 32.6,   'b': 0.72,  'c': 2.45, 'd': 0.0},
        {'min_H': 1.31, 'max_H': 3.93, 'a': 30.19,  'b': 0.82,  'c': 1.78, 'd': 0.1},
        {'min_H': 3.94, 'max_H': 6.13, 'a': 15.51,  'b': 0.85,  'c': 2.4,  'd': 0.11}
    ],
    '1008': [ # Serra dei Conti - Livello Misa (mt)
        {'min_H': 0.45, 'max_H': 0.73, 'a': 24.52,  'b': 0.45,  'c': 1.86, 'd': 0.0},
        {'min_H': 0.74, 'max_H': 2.9,  'a': 28.77,  'b': 0.55,  'c': 1.48, 'd': 0.0}
    ],
    '1283': [ # Nevola - Livello Nevola (mt) / Corinaldo
        {'min_H': 0.85, 'max_H': 1.2,  'a': 0.628,  'b': 0.849, 'c': 0.957, 'd': 0.0},
        {'min_H': 1.21, 'max_H': 1.46, 'a': 22.094, 'b': 1.2,   'c': 1.295, 'd': 0.231},
        {'min_H': 1.47, 'max_H': 2.84, 'a': 48.606, 'b': 1.46,  'c': 1.18,  'd': 4.091},
        {'min_H': 2.85, 'max_H': 5.0,  'a': 94.37,  'b': 2.84,  'c': 1.223, 'd': 75.173}
    ],
    # Aggiungere qui eventuali altre scale per Pianello (3072) o Garibaldi (3405) se disponibili
}
# --- FINE NUOVA COSTANTE ---

# --- AGGIORNAMENTO STATION_COORDS CON sensor_code ---
STATION_COORDS = {
    # Pioggia e Umidità (invariati, senza sensor_code idrometrico)
    'Arcevia - Pioggia Ora (mm)': {'lat': 43.5228, 'lon': 12.9388, 'name': 'Arcevia (Pioggia)', 'type': 'Pioggia', 'location_id': 'Arcevia'},
    'Barbara - Pioggia Ora (mm)': {'lat': 43.5808, 'lon': 13.0277, 'name': 'Barbara (Pioggia)', 'type': 'Pioggia', 'location_id': 'Barbara'},
    'Corinaldo - Pioggia Ora (mm)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Pioggia)', 'type': 'Pioggia', 'location_id': 'Corinaldo'},
    'Misa - Pioggia Ora (mm)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Pioggia)', 'type': 'Pioggia', 'location_id': 'Bettolelle'},
    HUMIDITY_COL_NAME: {'lat': 43.6, 'lon': 13.0, 'name': 'Montemurello (Umidità)', 'type': 'Umidità', 'location_id': 'Montemurello'},

    # Livelli GSheet Style (con sensor_code per quelli in RATING_CURVES)
    'Serra dei Conti - Livello Misa (mt)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti', 'sensor_code': '1008'},
    'Nevola - Livello Nevola (mt)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo', 'sensor_code': '1283'},
    'Misa - Livello Misa (mt)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle', 'sensor_code': '1112'},
    # Livelli GSheet Style (senza scala di deflusso specificata o con codici non in RATING_CURVES)
    'Pianello di Ostra - Livello Misa (m)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra', 'sensor_code': '3072'}, # Aggiunto codice per potenziale uso futuro
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi', 'sensor_code': '3405'}, # Aggiunto codice per potenziale uso futuro

    # Livelli CSV/Internal Style (con sensor_code) - Assicurarsi che corrispondano a GSheet se usati
     'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti', 'sensor_code': '1008'},
     'Livello Idrometrico Sensore 1112 [m] (Bettolelle)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle', 'sensor_code': '1112'},
     'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo', 'sensor_code': '1283'},
    # Livelli CSV/Internal Style (senza scala di deflusso specificata)
     'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra', 'sensor_code': '3072'},
     'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi', 'sensor_code': '3405'}
}
# --- FINE AGGIORNAMENTO ---

# --- NUOVA FUNZIONE CALCOLO PORTATA ---
def calculate_discharge(sensor_code, H):
    """
    Calcola la portata (Q) in m3/s dato il codice sensore e il livello H (m).
    Utilizza le scale di deflusso definite in RATING_CURVES.

    Args:
        sensor_code (str): Codice identificativo del sensore ('1112', '1008', '1283').
        H (float): Livello idrometrico in metri.

    Returns:
        float or None: La portata calcolata (Q) in m3/s, o None se H non è valido,
                       il sensore non ha una scala di deflusso definita, o H è
                       fuori dagli intervalli di validità. np.nan è anche possibile
                       in caso di errori matematici.
    """
    if sensor_code not in RATING_CURVES:
        return None
    if H is None or not isinstance(H, (int, float, np.number)) or np.isnan(H):
        return None

    try:
        H = float(H) # Assicura che sia float
        rules = RATING_CURVES[sensor_code]

        for rule in rules:
            min_H, max_H = rule['min_H'], rule['max_H']
            if min_H <= H <= max_H:
                a, b, c, d = rule['a'], rule['b'], rule['c'], rule['d']
                base = H - b
                if base < 0:
                    # Se H è sotto lo zero idrometrico 'b' MA dentro l'intervallo min/max,
                    # restituiamo 'd'. Altrimenti NaN.
                    return float(d) if H >= min_H else np.nan

                # Calcola la portata
                Q = a * math.pow(base, c) + d
                return float(Q)

        # H è fuori da tutti gli intervalli definiti
        return None # Restituisce None se H non cade in nessun intervallo

    except (ValueError, TypeError, OverflowError) as e:
        # print(f"Errore calcolo portata per sensore {sensor_code}, H={H}: {e}") # Debug
        return np.nan # Restituisce NaN in caso di errori matematici

# Vettorizza la funzione per applicarla facilmente agli array numpy o serie pandas
calculate_discharge_vectorized = np.vectorize(calculate_discharge, otypes=[float])
# --- FINE NUOVA FUNZIONE ---


# --- Definizioni Classi Modello (Dataset, LSTM, Encoder, Decoder, Seq2Seq) ---
# (Nessuna modifica necessaria qui)
class TimeSeriesDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tuple(torch.tensor(t, dtype=torch.float32) for t in tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

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

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(forecast_input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_forecast_step, hidden, cell):
        output, (hidden, cell) = self.lstm(x_forecast_step, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2SeqHydro(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps
        self.device = device

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0):
        batch_size = x_past.shape[0]
        forecast_window = x_future_forecast.shape[1]
        target_output_size = self.decoder.output_size

        if forecast_window < self.output_window:
             missing_steps = self.output_window - forecast_window
             last_forecast_step = x_future_forecast[:, -1:, :]
             padding = last_forecast_step.repeat(1, missing_steps, 1)
             x_future_forecast = torch.cat([x_future_forecast, padding], dim=1)
             # print(f"Warning: forecast_window ({forecast_window}) < output_window ({self.output_window}). Padding forecast input.") # Debug
             forecast_window = self.output_window

        outputs = torch.zeros(batch_size, self.output_window, target_output_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :]

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell
            )
            outputs[:, t, :] = decoder_output_step
            if t < self.output_window - 1:
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]

        return outputs

# --- Funzioni Utilità Modello/Dati ---
# (Nessuna modifica necessaria qui)
def prepare_training_data(df, feature_columns, target_columns, input_window_hours, output_window_hours, val_split=20):
    """Prepara i dati per l'allenamento del modello LSTM standard. Input/Output in ORE."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati LSTM Standard...")
    try:
        missing_features = [col for col in feature_columns if col not in df.columns]
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_features: st.error(f"Feature mancanti: {missing_features}"); return None, None, None, None, None, None
        if missing_targets: st.error(f"Target mancanti: {missing_targets}"); return None, None, None, None, None, None
        if df[feature_columns + target_columns].isnull().any().any(): st.warning("NaN prima creazione sequenze LSTM.")
    except Exception as e: st.error(f"Errore colonne prepare_training_data: {e}"); return None, None, None, None, None, None

    X, y = [], []
    total_len = len(df)
    input_steps = int(input_window_hours * 2) # Converti ore in passi da 30 min
    output_steps = int(output_window_hours * 2) # Converti ore in passi da 30 min
    required_len = input_steps + output_steps

    if total_len < required_len:
         st.error(f"Dati LSTM insufficienti ({total_len}) per sequenze ({required_len}).")
         return None, None, None, None, None, None

    for i in range(total_len - required_len + 1):
        X.append(df.iloc[i : i + input_steps][feature_columns].values)
        y.append(df.iloc[i + input_steps : i + required_len][target_columns].values)

    if not X or not y: st.error("Errore creazione X/y LSTM."); return None, None, None, None, None, None
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)

    if X.size == 0 or y.size == 0: st.error("X o y vuoti LSTM."); return None, None, None, None, None, None

    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    num_seq, seq_in, num_feat = X.shape
    num_seq_y, seq_out, num_targ = y.shape

    if seq_in != input_steps or seq_out != output_steps:
        st.error(f"Shape sequenze LSTM errata: In={seq_in}({input_steps}), Out={seq_out}({output_steps})")
        return None, None, None, None, None, None

    X_flat = X.reshape(-1, num_feat); y_flat = y.reshape(-1, num_targ)
    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except Exception as e_scale: st.error(f"Errore scaling LSTM: {e_scale}"); return None, None, None, None, None, None
    X_scaled = X_scaled_flat.reshape(num_seq, seq_in, num_feat)
    y_scaled = y_scaled_flat.reshape(num_seq_y, seq_out, num_targ)

    split_idx = int(len(X_scaled) * (1 - val_split / 100))
    if split_idx <= 0 or split_idx >= len(X_scaled):
         st.warning(f"Split indice LSTM ({split_idx}) non valido. Adattamento.")
         if len(X_scaled) < 2: st.error("Dataset LSTM troppo piccolo."); return None, None, None, None, None, None
         split_idx = max(1, min(len(X_scaled) - 1, split_idx))

    X_train = X_scaled[:split_idx]; y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]; y_val = y_scaled[split_idx:]

    if X_train.size == 0 or y_train.size == 0: st.error("Set Training LSTM vuoto."); return None, None, None, None, None, None
    if X_val.size == 0 or y_val.size == 0:
         st.warning("Set Validazione LSTM vuoto.")
         X_val = np.empty((0, seq_in, num_feat), dtype=np.float32)
         y_val = np.empty((0, seq_out, num_targ), dtype=np.float32)

    print(f"Dati LSTM pronti: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

def prepare_training_data_seq2seq(df, past_feature_cols, forecast_feature_cols, target_cols,
                                 input_window_steps, forecast_window_steps, output_window_steps, val_split=20):
    """Prepara dati per modello Seq2Seq con 3 scaler separati."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq...")
    print(f"Input Steps: {input_window_steps}, Forecast Steps: {forecast_window_steps}, Output Steps: {output_window_steps}")

    all_needed_cols = list(set(past_feature_cols + forecast_feature_cols + target_cols))
    try:
        for col in all_needed_cols:
            if col not in df.columns: raise ValueError(f"Colonna '{col}' Seq2Seq mancante.")
        if df[all_needed_cols].isnull().any().any():
             st.warning("NaN prima creazione sequenze Seq2Seq. Applico ffill/bfill.")
             df_filled = df.copy()
             df_filled[all_needed_cols] = df_filled[all_needed_cols].fillna(method='ffill').fillna(method='bfill')
             if df_filled[all_needed_cols].isnull().any().any():
                  st.error("NaN RESIDUI dopo fillna Seq2Seq."); return None, None, None, None, None, None, None, None, None
             df_to_use = df_filled
        else: df_to_use = df
    except ValueError as e: st.error(f"Errore colonne prepare_seq2seq: {e}"); return None, None, None, None, None, None, None, None, None

    X_encoder, X_decoder, y_target = [], [], []
    total_len = len(df_to_use)
    required_len = input_window_steps + max(forecast_window_steps, output_window_steps)

    if total_len < required_len:
        st.error(f"Dati Seq2Seq insuff. ({total_len}) per sequenze ({required_len}).")
        return None, None, None, None, None, None, None, None, None

    print(f"Creazione sequenze Seq2Seq: {total_len - required_len + 1} possibili...")
    for i in range(total_len - required_len + 1):
        enc_end = i + input_window_steps
        X_encoder.append(df_to_use.iloc[i : enc_end][past_feature_cols].values)
        dec_start = enc_end
        dec_end_forecast = dec_start + forecast_window_steps
        X_decoder.append(df_to_use.iloc[dec_start : dec_end_forecast][forecast_feature_cols].values)
        target_end = dec_start + output_window_steps
        y_target.append(df_to_use.iloc[dec_start : target_end][target_cols].values)

    if not X_encoder or not X_decoder or not y_target:
        st.error("Errore creazione sequenze X/Y Seq2Seq."); return None, None, None, None, None, None, None, None, None

    X_encoder = np.array(X_encoder, dtype=np.float32)
    X_decoder = np.array(X_decoder, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    print(f"Shapes Seq2Seq NumPy: Enc={X_encoder.shape}, Dec={X_decoder.shape}, Tar={y_target.shape}")

    if X_encoder.shape[1] != input_window_steps or X_decoder.shape[1] != forecast_window_steps or y_target.shape[1] != output_window_steps:
         st.error("Errore shape sequenze Seq2Seq."); return None, None, None, None, None, None, None, None, None
    if X_encoder.shape[2] != len(past_feature_cols) or X_decoder.shape[2] != len(forecast_feature_cols) or y_target.shape[2] != len(target_cols):
         st.error("Errore numero feature/target Seq2Seq."); return None, None, None, None, None, None, None, None, None

    scaler_past_features = MinMaxScaler(); scaler_forecast_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    num_sequences = X_encoder.shape[0]
    X_enc_flat = X_encoder.reshape(-1, len(past_feature_cols))
    X_dec_flat = X_decoder.reshape(-1, len(forecast_feature_cols))
    y_tar_flat = y_target.reshape(-1, len(target_cols))

    try:
        X_enc_scaled_flat = scaler_past_features.fit_transform(X_enc_flat)
        X_dec_scaled_flat = scaler_forecast_features.fit_transform(X_dec_flat)
        y_tar_scaled_flat = scaler_targets.fit_transform(y_tar_flat)
    except Exception as e_scale: st.error(f"Errore scaling Seq2Seq: {e_scale}"); return None, None, None, None, None, None, None, None, None

    X_enc_scaled = X_enc_scaled_flat.reshape(num_sequences, input_window_steps, len(past_feature_cols))
    X_dec_scaled = X_dec_scaled_flat.reshape(num_sequences, forecast_window_steps, len(forecast_feature_cols))
    y_tar_scaled = y_tar_scaled_flat.reshape(num_sequences, output_window_steps, len(target_cols))

    split_idx = int(num_sequences * (1 - val_split / 100))
    if split_idx <= 0 or split_idx >= num_sequences:
         st.warning(f"Split indice Seq2Seq ({split_idx}) non valido. Adattamento.")
         if num_sequences < 2: st.error("Dataset Seq2Seq troppo piccolo."); return None, None, None, None, None, None, None, None, None
         split_idx = max(1, min(num_sequences - 1, split_idx))

    X_enc_train, X_dec_train, y_tar_train = X_enc_scaled[:split_idx], X_dec_scaled[:split_idx], y_tar_scaled[:split_idx]
    X_enc_val, X_dec_val, y_tar_val = X_enc_scaled[split_idx:], X_dec_scaled[split_idx:], y_tar_scaled[split_idx:]

    if X_enc_train.size == 0 or X_dec_train.size == 0 or y_tar_train.size == 0:
        st.error("Set Training Seq2Seq vuoto."); return None, None, None, None, None, None, None, None, None
    if X_enc_val.size == 0 or X_dec_val.size == 0 or y_tar_val.size == 0:
        st.warning("Set Validazione Seq2Seq vuoto.")
        X_enc_val = np.empty((0, input_window_steps, len(past_feature_cols)), dtype=np.float32)
        X_dec_val = np.empty((0, forecast_window_steps, len(forecast_feature_cols)), dtype=np.float32)
        y_tar_val = np.empty((0, output_window_steps, len(target_cols)), dtype=np.float32)

    print(f"Dati Seq2Seq pronti: Train={len(X_enc_train)}, Val={len(X_enc_val)}")
    return (X_enc_train, X_dec_train, y_tar_train,
            X_enc_val, X_dec_val, y_tar_val,
            scaler_past_features, scaler_forecast_features, scaler_targets)


# --- Funzioni Caricamento Modello/Scaler (Aggiornate) ---
# (Nessuna modifica necessaria qui)
@st.cache_data(show_spinner="Ricerca modelli disponibili...")
def find_available_models(models_dir=MODELS_DIR):
    """ Trova modelli validi (pth, json, scalers) e determina il tipo (LSTM/Seq2Seq). """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE find_available_models")
    available = {}
    if not os.path.isdir(models_dir): return available

    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
    for pth_path in pth_files:
        base = os.path.splitext(os.path.basename(pth_path))[0]
        cfg_path = os.path.join(models_dir, f"{base}.json")

        if not os.path.exists(cfg_path): continue

        model_info = {"config_name": base, "pth_path": pth_path, "config_path": cfg_path}
        model_type = "LSTM"; valid_model = False

        try:
            with open(cfg_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
            name = config_data.get("display_name", base)
            model_type = config_data.get("model_type", "LSTM")

            if model_type == "Seq2Seq":
                req_keys = ["input_window_steps", "output_window_steps", "forecast_window_steps", "hidden_size", "num_layers", "dropout", "all_past_feature_columns", "forecast_input_columns", "target_columns"]
                s_past_p = os.path.join(models_dir, f"{base}_past_features.joblib")
                s_fore_p = os.path.join(models_dir, f"{base}_forecast_features.joblib")
                s_targ_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if all(k in config_data for k in req_keys) and os.path.exists(s_past_p) and os.path.exists(s_fore_p) and os.path.exists(s_targ_p):
                    model_info.update({ "scaler_past_features_path": s_past_p, "scaler_forecast_features_path": s_fore_p, "scaler_targets_path": s_targ_p, "model_type": "Seq2Seq" })
                    valid_model = True
            else: # LSTM standard
                req_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"] # feature_columns opzionale
                scf_p = os.path.join(models_dir, f"{base}_features.joblib")
                sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if all(k in config_data for k in req_keys) and os.path.exists(scf_p) and os.path.exists(sct_p):
                     model_info.update({ "scaler_features_path": scf_p, "scaler_targets_path": sct_p, "model_type": "LSTM" })
                     valid_model = True

        except Exception as e_cfg:
            st.warning(f"Modello '{base}' ignorato: errore config '{cfg_path}': {e_cfg}")
            valid_model = False

        if valid_model: available[name] = model_info
        else: print(f"Modello '{base}' ignorato: file mancanti o config JSON incompleta ({model_type}).")

    return available

@st.cache_data
def load_model_config(_config_path):
    """Carica config e valida chiavi base."""
    try:
        with open(_config_path, 'r', encoding='utf-8') as f: config = json.load(f)
        return config
    except Exception as e: st.error(f"Errore caricamento config '{_config_path}': {e}"); return None

@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    """Carica modello Pytorch (LSTM o Seq2Seq) in base al config."""
    if not config: st.error("Config non valida per caricamento modello."); return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get("model_type", "LSTM")

    try:
        model = None
        if model_type == "Seq2Seq":
            enc_in = len(config["all_past_feature_columns"]); dec_in = len(config["forecast_input_columns"])
            dec_out = len(config["target_columns"]); hidden = config["hidden_size"]
            layers = config["num_layers"]; drop = config["dropout"]; out_win = config["output_window_steps"]
            encoder = EncoderLSTM(enc_in, hidden, layers, drop).to(device)
            decoder = DecoderLSTM(dec_in, hidden, dec_out, layers, drop).to(device)
            model = Seq2SeqHydro(encoder, decoder, out_win, device).to(device)
        else: # LSTM standard
            f_cols_lstm = config.get("feature_columns")
            if not f_cols_lstm:
                 st.warning(f"Config LSTM '{config.get('display_name', 'N/A')}' senza feature_columns. Uso globali.")
                 f_cols_lstm = st.session_state.get("feature_columns", [])
                 if not f_cols_lstm: raise ValueError("Feature globali non definite per modello LSTM.")
                 config["feature_columns"] = f_cols_lstm # Aggiorna config in memoria
            input_size_lstm = len(f_cols_lstm); target_size_lstm = len(config["target_columns"])
            out_win_lstm = config["output_window"]
            model = HydroLSTM(input_size_lstm, config["hidden_size"], target_size_lstm, out_win_lstm, config["num_layers"], config["dropout"]).to(device)

        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"Modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'):
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")

        model.eval()
        print(f"Modello '{config.get('display_name', 'N/A')}' ({model_type}) caricato su {device}.")
        return model, device

    except Exception as e:
        st.error(f"Errore caricamento modello '{config.get('display_name', 'N/A')}' ({model_type}): {e}")
        st.error(traceback.format_exc()); return None, None

@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(config, model_info):
    """Carica 1, 2 o 3 scaler in base al tipo di modello nel config."""
    if not config or not model_info: st.error("Config/info mancanti per caricare scaler."); return None
    model_type = config.get("model_type", "LSTM")

    def _load_joblib(path):
         if isinstance(path, str):
              if not os.path.exists(path): raise FileNotFoundError(f"Scaler '{path}' non trovato.")
              return joblib.load(path)
         elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
         else: raise TypeError(f"Percorso scaler non valido: {type(path)}")

    try:
        if model_type == "Seq2Seq":
            sp_path = model_info["scaler_past_features_path"]; sf_path = model_info["scaler_forecast_features_path"]
            st_path = model_info["scaler_targets_path"]
            scaler_past = _load_joblib(sp_path); scaler_forecast = _load_joblib(sf_path); scaler_targets = _load_joblib(st_path)
            print(f"Scaler Seq2Seq caricati."); return {"past": scaler_past, "forecast": scaler_forecast, "targets": scaler_targets}
        else: # LSTM
            sf_path = model_info["scaler_features_path"]; st_path = model_info["scaler_targets_path"]
            scaler_features = _load_joblib(sf_path); scaler_targets = _load_joblib(st_path)
            print(f"Scaler LSTM caricati."); return scaler_features, scaler_targets

    except Exception as e:
        st.error(f"Errore caricamento scaler ({model_type}): {e}")
        st.error(traceback.format_exc())
        if model_type == "Seq2Seq": return None
        else: return None, None

# --- Funzioni Predict (Standard e Seq2Seq) ---
# (Nessuna modifica necessaria qui)
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    """Esegue predizione per modello LSTM standard."""
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict LSTM: Modello/scaler/config mancanti."); return None

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM": st.error(f"predict chiamata su modello non LSTM ({model_type})"); return None

    input_steps = config["input_window"]; output_steps = config["output_window"]
    target_cols = config["target_columns"]; f_cols_cfg = config.get("feature_columns", [])

    if input_data.shape[0] != input_steps: st.error(f"Predict LSTM: Input righe {input_data.shape[0]} != {input_steps}."); return None
    expected_features = getattr(scaler_features, 'n_features_in_', len(f_cols_cfg) if f_cols_cfg else None)
    if expected_features is not None and input_data.shape[1] != expected_features: st.error(f"Predict LSTM: Input colonne {input_data.shape[1]} != {expected_features}."); return None

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(inp_tens)

        if output.shape != (1, output_steps, len(target_cols)): st.error(f"Predict LSTM: Shape output {output.shape} inattesa."); return None

        out_np = output.cpu().numpy().squeeze(0)
        expected_targets = getattr(scaler_targets, 'n_features_in_', len(target_cols))
        if out_np.shape[1] != expected_targets: st.error(f"Predict LSTM: Output colonne {out_np.shape[1]} != targets scaler/config ({expected_targets})."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds

    except Exception as e: st.error(f"Errore predict LSTM: {e}"); st.error(traceback.format_exc()); return None

def predict_seq2seq(model, past_data, future_forecast_data, scalers, config, device):
    """Esegue predizione per modello Seq2Seq."""
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]): st.error("Predict Seq2Seq: Input mancanti."); return None

    model_type = config.get("model_type")
    if model_type != "Seq2Seq": st.error(f"predict_seq2seq chiamata su modello non Seq2Seq"); return None

    input_steps = config["input_window_steps"]; forecast_steps = config["forecast_window_steps"]
    output_steps = config["output_window_steps"]; past_cols = config["all_past_feature_columns"]
    forecast_cols = config["forecast_input_columns"]; target_cols = config["target_columns"]

    scaler_past = scalers.get("past"); scaler_forecast = scalers.get("forecast"); scaler_targets = scalers.get("targets")
    if not all([scaler_past, scaler_forecast, scaler_targets]): st.error("Predict Seq2Seq: Scaler mancanti."); return None

    if past_data.shape != (input_steps, len(past_cols)): st.error(f"Predict Seq2Seq: Shape passati {past_data.shape} errata."); return None
    if future_forecast_data.shape != (forecast_steps, len(forecast_cols)): st.error(f"Predict Seq2Seq: Shape forecast {future_forecast_data.shape} errata."); return None
    if forecast_steps < output_steps: st.warning(f"Predict Seq2Seq: Finestra forecast ({forecast_steps}) < output ({output_steps}). Padding.")

    model.eval()
    try:
        past_norm = scaler_past.transform(past_data); future_norm = scaler_forecast.transform(future_forecast_data)
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device); future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)

        with torch.no_grad(): output = model(past_tens, future_tens)

        if output.shape != (1, output_steps, len(target_cols)): st.error(f"Predict Seq2Seq: Shape output {output.shape} inattesa."); return None

        out_np = output.cpu().numpy().squeeze(0)
        expected_targets = getattr(scaler_targets, 'n_features_in_', len(target_cols))
        if out_np.shape[1] != expected_targets: st.error(f"Predict Seq2Seq: Output colonne {out_np.shape[1]} != targets scaler/config ({expected_targets})."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds

    except Exception as e: st.error(f"Errore predict Seq2Seq: {e}"); st.error(traceback.format_exc()); return None

# --- Funzione Grafici Previsioni (usata in Simulazione - include già Q) ---
def plot_predictions(predictions, config, start_time=None):
    """
    Genera grafici Plotly INDIVIDUALI per le previsioni (LSTM o Seq2Seq).
    Aggiunge linee orizzontali per le soglie H e un secondo asse Y per la portata Q.
    """
    if config is None or predictions is None: return []

    model_type = config.get("model_type", "LSTM")
    target_cols = config["target_columns"]
    output_steps = predictions.shape[0]
    attribution_text = ATTRIBUTION_PHRASE
    figs = []

    for i, sensor in enumerate(target_cols):
        fig = go.Figure()
        if start_time:
            time_steps = [start_time + timedelta(minutes=30 * (step + 1)) for step in range(output_steps)]
            x_axis, x_title = time_steps, "Data e Ora Previste"; x_tick_format = "%d/%m %H:%M"
        else:
            time_steps = np.arange(1, output_steps + 1) * 0.5
            x_axis, x_title = time_steps, "Ore Future (passi da 30 min)"; x_tick_format = None
        x_axis_np = np.array(x_axis)

        station_name_graph = get_station_label(sensor, short=False)
        plot_title = f'Previsione {model_type} - {station_name_graph}<br><span style="font-size:10px;">{attribution_text}</span>'

        unit_match = re.search(r'\((.*?)\)|\[(.*?)\]', sensor)
        y_axis_unit = "m" if unit_match else ""
        if unit_match: y_axis_unit = (unit_match.group(1) or unit_match.group(2) or "m").strip()

        # Traccia Livello H (Asse Y1)
        fig.add_trace(go.Scatter(
            x=x_axis_np, y=predictions[:, i], mode='lines+markers',
            name=f'Livello H ({y_axis_unit})', yaxis='y1'
        ))

        # Soglie H (da SIMULATION_THRESHOLDS)
        threshold_info = SIMULATION_THRESHOLDS.get(sensor, {})
        soglia_attenzione = threshold_info.get('attenzione')
        soglia_allerta = threshold_info.get('allerta')
        if soglia_attenzione is not None: fig.add_hline(y=soglia_attenzione, line_dash="dash", line_color="orange", annotation_text=f"Att.H({soglia_attenzione:.2f})", annotation_position="bottom right", layer="below")
        if soglia_allerta is not None: fig.add_hline(y=soglia_allerta, line_dash="dash", line_color="red", annotation_text=f"All.H({soglia_allerta:.2f})", annotation_position="top right", layer="below")

        # Traccia Portata Q (Asse Y2, Condizionale)
        sensor_info = STATION_COORDS.get(sensor)
        sensor_code_plot = sensor_info.get('sensor_code') if sensor_info else None
        has_discharge_data = False
        predicted_Q_values = None
        if sensor_code_plot and sensor_code_plot in RATING_CURVES:
            predicted_H_values = predictions[:, i]
            predicted_Q_values = calculate_discharge_vectorized(sensor_code_plot, predicted_H_values)
            valid_Q_mask = pd.notna(predicted_Q_values) & (predicted_Q_values > 0) # Mostra solo Q > 0
            if np.any(valid_Q_mask):
                has_discharge_data = True
                fig.add_trace(go.Scatter(
                    x=x_axis_np[valid_Q_mask], y=predicted_Q_values[valid_Q_mask], mode='lines',
                    name='Portata Q (m³/s)', line=dict(color='firebrick', dash='dot'), yaxis='y2'
                ))

        # Aggiornamento Layout
        fig.update_layout(
            title=plot_title, height=400, margin=dict(l=60, r=60, t=70, b=50),
            hovermode="x unified", template="plotly_white",
            xaxis_title=x_title, yaxis=dict(title=f"Livello H ({y_axis_unit})", titlefont_color="#1f77b4", tickfont_color="#1f77b4", side="left", rangemode='tozero')
        )
        if x_tick_format: fig.update_xaxes(tickformat=x_tick_format)

        if has_discharge_data:
            fig.update_layout(
                 yaxis2=dict(title="Portata Q (m³/s)", titlefont_color="firebrick", tickfont_color="firebrick", overlaying="y", side="right", rangemode='tozero', showgrid=False),
                 legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) # Legenda sotto se c'è Q
            )
        else: # Legenda di default se solo H
             fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1))

        figs.append(fig)
    return figs

# --- Funzioni Fetch Dati Google Sheet (Dashboard e Simulazione) ---
# (Nessuna modifica necessaria qui)
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format,
                                fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS):
    """Recupera e pulisce i dati da GSheet per la Dashboard."""
    mode = "tutti i dati" if fetch_all else f"ultime {num_rows_default} righe"
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data ({mode})")
    actual_fetch_time = datetime.now(italy_tz)
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", actual_fetch_time
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials); sh = gc.open_by_key(sheet_id); worksheet = sh.sheet1
        all_values = worksheet.get_all_values()

        if not all_values or len(all_values) < 2: return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time

        headers = all_values[0]
        if fetch_all: data_rows = all_values[1:]
        else: start_index = max(1, len(all_values) - num_rows_default); data_rows = all_values[start_index:]

        missing_cols = [col for col in relevant_columns if col not in headers]
        if missing_cols: return None, f"Errore: Colonne GSheet mancanti: {', '.join(missing_cols)}", actual_fetch_time

        df = pd.DataFrame(data_rows, columns=headers)
        cols_to_select = [c for c in relevant_columns if c in df.columns]
        missing_in_df_after_create = [c for c in relevant_columns if c not in df.columns]
        if missing_in_df_after_create: st.warning(f"Colonne GSheet '{missing_in_df_after_create}' non trovate nel DF, ignorate.")
        df = df[cols_to_select]

        error_parsing = []
        for col in df.columns:
            if col == date_col:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format if date_format else None, errors='coerce', infer_datetime_format=True)
                    if df[col].isnull().any(): error_parsing.append(f"Formato data non valido '{col}'")
                    if not df[col].empty and df[col].notna().any():
                        if df[col].dt.tz is None: df[col] = df[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                        else: df[col] = df[col].dt.tz_convert(italy_tz)
                except Exception as e_date: error_parsing.append(f"Errore data '{col}': {e_date}"); df[col] = pd.NaT
            else:
                try:
                    if col in df.columns:
                        df_col_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                        df[col] = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e_num: error_parsing.append(f"Errore numerico '{col}': {e_num}"); df[col] = np.nan

        if date_col in df.columns: df = df.sort_values(by=date_col, na_position='first').reset_index(drop=True)
        else: st.warning("Colonna data GSheet non trovata per ordinamento.")

        numeric_cols_in_df = df.select_dtypes(include=np.number).columns
        nan_numeric_count = df[numeric_cols_in_df].isnull().sum().sum()
        if nan_numeric_count > 0: st.caption(f"Nota Dashboard: Trovati {nan_numeric_count} valori mancanti/non numerici GSheet.")

        error_message = "Attenzione conversione dati GSheet: " + " | ".join(error_parsing) if error_parsing else None
        return df, error_message, actual_fetch_time

    except gspread.exceptions.APIError as api_e:
        try: error_details = api_e.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}";
        except: error_msg = str(api_e)
        return None, f"Errore API Google Sheets: {error_msg}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}').", actual_fetch_time
    except Exception as e: return None, f"Errore imprevisto recupero GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time

@st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet per simulazione...")
def fetch_sim_gsheet_data(sheet_id_fetch, n_rows_steps, date_col_gs, date_format_gs, col_mapping, required_model_cols_fetch, impute_dict):
    """ Recupera, mappa, pulisce e ordina n_rows_steps di dati da GSheet per simulazione. """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_sim_gsheet_data (Steps: {n_rows_steps})")
    actual_fetch_time = datetime.now(italy_tz)
    last_valid_timestamp = None
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", None
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials); sh = gc.open_by_key(sheet_id_fetch); worksheet = sh.sheet1
        all_data_gs = worksheet.get_all_values()

        if not all_data_gs or len(all_data_gs) < 2 : return None, "Errore: Foglio GSheet vuoto o senza dati.", None

        headers_gs = all_data_gs[0]
        start_index_gs = max(1, len(all_data_gs) - n_rows_steps)
        data_rows_gs = all_data_gs[start_index_gs:]
        if len(data_rows_gs) < n_rows_steps: st.warning(f"Dati GSheet insuff. ({len(data_rows_gs)}/{n_rows_steps}).")

        df_gsheet_raw = pd.DataFrame(data_rows_gs, columns=headers_gs)
        required_gsheet_cols_from_mapping = list(col_mapping.keys()) + ([date_col_gs] if date_col_gs not in col_mapping.keys() else [])
        required_gsheet_cols_from_mapping = list(set(required_gsheet_cols_from_mapping))
        missing_gsheet_cols_in_sheet = [c for c in required_gsheet_cols_from_mapping if c not in df_gsheet_raw.columns]
        if missing_gsheet_cols_in_sheet: return None, f"Errore: Colonne GSheet ({', '.join(missing_gsheet_cols_in_sheet)}) mancanti.", None

        cols_to_select_gsheet = [c for c in required_gsheet_cols_from_mapping if c in df_gsheet_raw.columns]
        df_subset = df_gsheet_raw[cols_to_select_gsheet].copy()
        df_mapped = df_subset.rename(columns=col_mapping)

        for model_col, impute_val in impute_dict.items():
             if model_col not in df_mapped.columns: df_mapped[model_col] = impute_val

        final_missing_model_cols = [c for c in required_model_cols_fetch if c not in df_mapped.columns]
        if final_missing_model_cols: return None, f"Errore: Colonne modello ({', '.join(final_missing_model_cols)}) mancanti dopo map/impute.", None

        date_col_model_name = col_mapping.get(date_col_gs, date_col_gs)
        if date_col_model_name not in df_mapped.columns: st.warning(f"Colonna data '{date_col_model_name}' non presente."); date_col_model_name = None

        numeric_model_cols = [c for c in required_model_cols_fetch if c != date_col_model_name]
        for col in numeric_model_cols:
            if col not in df_mapped.columns: continue
            try:
                if pd.api.types.is_object_dtype(df_mapped[col]) or pd.api.types.is_string_dtype(df_mapped[col]):
                    col_str = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    df_mapped[col] = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
            except Exception as e_clean_num: st.warning(f"Problema pulizia GSheet col '{col}': {e_clean_num}. Trattata NaN."); df_mapped[col] = np.nan

        if date_col_model_name:
             try:
                 df_mapped[date_col_model_name] = pd.to_datetime(df_mapped[date_col_model_name], format=date_format_gs, errors='coerce', infer_datetime_format=True)
                 if df_mapped[date_col_model_name].isnull().any(): st.warning(f"Date non valide GSheet sim ('{date_col_model_name}').")
                 if not df_mapped[date_col_model_name].empty and df_mapped[date_col_model_name].notna().any():
                     if df_mapped[date_col_model_name].dt.tz is None: df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                     else: df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_convert(italy_tz)
                 df_mapped = df_mapped.sort_values(by=date_col_model_name, na_position='first')
                 valid_dates = df_mapped[date_col_model_name].dropna()
                 if not valid_dates.empty: last_valid_timestamp = valid_dates.iloc[-1]
             except Exception as e_date_clean: st.warning(f"Errore data GSheet sim '{date_col_model_name}': {e_date_clean}."); date_col_model_name = None

        try:
            cols_present_final = [c for c in required_model_cols_fetch if c in df_mapped.columns]
            df_final = df_mapped[cols_present_final].copy()
        except KeyError as e_key: return None, f"Errore selezione colonne finali sim: '{e_key}' mancante.", None

        numeric_cols_to_fill = df_final.select_dtypes(include=np.number).columns
        nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
        if nan_count_before > 0:
             st.warning(f"{nan_count_before} NaN GSheet sim. Applico ffill/bfill.")
             df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(method='ffill').fillna(method='bfill')
             if df_final[numeric_cols_to_fill].isnull().sum().sum() > 0:
                 st.error(f"NaN residui ({df_final[numeric_cols_to_fill].isnull().sum().sum()}) GSheet sim. Riempio con 0.")
                 df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(0)

        if len(df_final) != n_rows_steps: st.warning(f"Attenzione: Righe finali ({len(df_final)}) != richieste ({n_rows_steps}) GSheet.")

        print(f"Dati sim GSheet pronti ({len(df_final)} righe). Ultimo TS: {last_valid_timestamp}")
        try: df_final = df_final[required_model_cols_fetch]
        except KeyError as e_final_order:
            missing_final = [c for c in required_model_cols_fetch if c not in df_final.columns]
            return None, f"Errore: Colonne modello ({missing_final}) mancanti prima del return.", None
        return df_final, None, last_valid_timestamp

    except gspread.exceptions.APIError as api_e_sim:
        try: error_details = api_e_sim.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e_sim)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}";
        except: error_msg = str(api_e_sim)
        return None, f"Errore API Google Sheets: {error_msg}", None
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google sim non trovato (ID: '{sheet_id_fetch}').", None
    except Exception as e_sim_fetch: st.error(traceback.format_exc()); return None, f"Errore imprevisto import GSheet sim: {type(e_sim_fetch).__name__} - {e_sim_fetch}", None

# --- Funzioni Allenamento (Standard e Seq2Seq) ---
# (Nessuna modifica necessaria qui)
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window_steps,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2,
                save_strategy='migliore', preferred_device='auto'):
    """Allena modello LSTM standard e mostra progresso."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training LSTM Standard...")
    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"Training LSTM userà: {device}")

    model = HydroLSTM(input_size, hidden_size, output_size, output_window_steps, num_layers, dropout).to(device)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val) if (X_val is not None and X_val.size > 0) else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=device.type == 'cuda') if val_dataset else None
    criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    train_losses, val_losses = [], []; best_val_loss = float('inf'); best_model_state_dict = None
    progress_bar = st.progress(0.0, text="Training LSTM: Inizio...")
    status_text = st.empty(); loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        valid_v_loss = [v for v in v_loss if v is not None] if v_loss else []
        if valid_v_loss: fig.add_trace(go.Scatter(y=valid_v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss (LSTM)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training LSTM per {epochs} epoche su **{device}**...")
    start_training_time = pytime.time()

    for epoch in range(epochs):
        epoch_start_time = pytime.time(); model.train(); train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            outputs = model(X_batch); loss = criterion(outputs, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader); train_losses.append(train_loss)

        val_loss = None
        if val_loader:
            model.eval(); current_val_loss = 0.0
            with torch.no_grad():
                for X_batch_val, y_batch_val in val_loader:
                    X_batch_val, y_batch_val = X_batch_val.to(device, non_blocking=True), y_batch_val.to(device, non_blocking=True)
                    outputs_val = model(X_batch_val); loss_val = criterion(outputs_val, y_batch_val)
                    current_val_loss += loss_val.item()
            if len(val_loader) > 0: val_loss = current_val_loss / len(val_loader)
            else: val_loss = float('inf')
            val_losses.append(val_loss); scheduler.step(val_loss)
            if val_loss < best_val_loss: best_val_loss = val_loss; best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: val_losses.append(None)

        progress_percentage = (epoch + 1) / epochs
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None and val_loss != float('inf') else "N/A"
        epoch_time = pytime.time() - epoch_start_time
        status_text.text(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) | Train Loss: {train_loss:.6f} | Val Loss: {val_loss_str} | LR: {current_lr:.6f}')
        progress_bar.progress(progress_percentage, text=f"Training LSTM: Epoca {epoch+1}/{epochs}")
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

    total_training_time = pytime.time() - start_training_time
    st.write(f"Training LSTM completato in {total_training_time:.1f} secondi.")

    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict:
            try:
                best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
                model.load_state_dict(best_model_state_dict_on_device); final_model_to_return = model
                st.success(f"Strategia 'migliore': Caricato modello LSTM Val Loss ({best_val_loss:.6f}).")
            except Exception as e_load_best: st.error(f"Errore caricamento stato LSTM migliore: {e_load_best}. Restituito finale.")
        elif val_loader: st.warning("Strat 'migliore' LSTM: No miglioramento. Restituito finale.")
        else: st.warning("Strat 'migliore' LSTM: No validazione. Restituito finale.")
    elif save_strategy == 'finale': st.info("Strat 'finale' LSTM: Restituito modello ultima epoca.")

    return final_model_to_return, train_losses, val_losses

def train_model_seq2seq(X_enc_train, X_dec_train, y_tar_train, X_enc_val, X_dec_val, y_tar_val,
                        encoder, decoder, output_window_steps, epochs=50, batch_size=32, learning_rate=0.001,
                        save_strategy='migliore', preferred_device='auto', teacher_forcing_ratio_schedule=None):
    """Allena modello Seq2Seq e mostra progresso."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training Seq2Seq...")
    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"Training Seq2Seq userà: {device}")

    model = Seq2SeqHydro(encoder, decoder, output_window_steps, device).to(device)
    train_dataset = TimeSeriesDataset(X_enc_train, X_dec_train, y_tar_train)
    val_dataset = TimeSeriesDataset(X_enc_val, X_dec_val, y_tar_val) if (X_enc_val is not None and X_enc_val.size > 0) else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=device.type == 'cuda') if val_dataset else None
    criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    train_losses, val_losses = [], []; best_val_loss = float('inf'); best_model_state_dict = None
    progress_bar = st.progress(0.0, text="Training Seq2Seq: Inizio..."); status_text = st.empty(); loss_chart_placeholder = st.empty()

    def update_loss_chart_seq2seq(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        valid_v_loss = [v for v in v_loss if v is not None] if v_loss else []
        if valid_v_loss: fig.add_trace(go.Scatter(y=valid_v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss (Seq2Seq)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training Seq2Seq per {epochs} epoche su **{device}**...")
    start_training_time = pytime.time()

    for epoch in range(epochs):
        epoch_start_time = pytime.time(); model.train(); train_loss = 0.0
        current_tf_ratio = 0.5
        if teacher_forcing_ratio_schedule and isinstance(teacher_forcing_ratio_schedule, list) and len(teacher_forcing_ratio_schedule) == 2:
            start_tf, end_tf = teacher_forcing_ratio_schedule
            if epochs > 1: current_tf_ratio = max(end_tf, start_tf - (start_tf - end_tf) * epoch / (epochs - 1))
            else: current_tf_ratio = start_tf

        for i, (x_enc_b, x_dec_b, y_tar_b) in enumerate(train_loader):
            x_enc_b = x_enc_b.to(device, non_blocking=True); x_dec_b = x_dec_b.to(device, non_blocking=True); y_tar_b = y_tar_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            batch_s = x_enc_b.shape[0]; out_win = model.output_window; target_size = model.decoder.output_size
            outputs_train = torch.zeros(batch_s, out_win, target_size).to(device)
            encoder_hidden, encoder_cell = model.encoder(x_enc_b); decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
            decoder_input_step = x_dec_b[:, 0:1, :]
            for t in range(out_win):
                decoder_output_step, decoder_hidden, decoder_cell = model.decoder(decoder_input_step, decoder_hidden, decoder_cell)
                outputs_train[:, t, :] = decoder_output_step
                use_teacher_forcing = random.random() < current_tf_ratio
                if t < out_win - 1: decoder_input_step = x_dec_b[:, t+1:t+2, :] # Sempre input forecast reale in training
            loss = criterion(outputs_train, y_tar_b); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader); train_losses.append(train_loss)

        val_loss = None
        if val_loader:
            model.eval(); current_val_loss = 0.0
            with torch.no_grad():
                for x_enc_vb, x_dec_vb, y_tar_vb in val_loader:
                    x_enc_vb = x_enc_vb.to(device, non_blocking=True); x_dec_vb = x_dec_vb.to(device, non_blocking=True); y_tar_vb = y_tar_vb.to(device, non_blocking=True)
                    outputs_val = model(x_enc_vb, x_dec_vb) # Chiama forward senza TF
                    loss_val = criterion(outputs_val, y_tar_vb); current_val_loss += loss_val.item()
            if len(val_loader) > 0: val_loss = current_val_loss / len(val_loader)
            else: val_loss = float('inf')
            val_losses.append(val_loss); scheduler.step(val_loss)
            if val_loss < best_val_loss: best_val_loss = val_loss; best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: val_losses.append(None)

        progress_percentage = (epoch + 1) / epochs
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None and val_loss != float('inf') else "N/A"
        epoch_time = pytime.time() - epoch_start_time
        tf_ratio_str = f"{current_tf_ratio:.2f}" if teacher_forcing_ratio_schedule else "N/A"
        status_text.text(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) | Train Loss: {train_loss:.6f} | Val Loss: {val_loss_str} | TF Ratio: {tf_ratio_str} | LR: {current_lr:.6f}')
        progress_bar.progress(progress_percentage, text=f"Training Seq2Seq: Epoca {epoch+1}/{epochs}")
        update_loss_chart_seq2seq(train_losses, val_losses, loss_chart_placeholder)

    total_training_time = pytime.time() - start_training_time
    st.write(f"Training Seq2Seq completato in {total_training_time:.1f} secondi.")

    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict:
            try:
                best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
                model.load_state_dict(best_model_state_dict_on_device); final_model_to_return = model
                st.success(f"Strat 'migliore': Caricato modello Seq2Seq Val Loss ({best_val_loss:.6f}).")
            except Exception as e_load_best: st.error(f"Errore caricamento stato Seq2Seq migliore: {e_load_best}. Restituito finale.")
        elif val_loader: st.warning("Strat 'migliore' Seq2Seq: No miglioramento. Restituito finale.")
        else: st.warning("Strat 'migliore' Seq2Seq: No validazione. Restituito finale.")
    elif save_strategy == 'finale': st.info("Strat 'finale' Seq2Seq: Restituito modello ultima epoca.")

    return final_model_to_return, train_losses, val_losses


# --- Funzioni Helper Download ---
# (Nessuna modifica necessaria qui)
def get_table_download_link(df, filename="data.csv", link_text="Scarica CSV"):
    """Genera link download per DataFrame come CSV (UTF-8 BOM)."""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        csv_buffer.seek(0); b64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8-sig')).decode()
        return f'<a href="data:text/csv;charset=utf-8-sig;base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.warning(f"Errore link CSV: {e}"); return "<i>Errore link CSV</i>"

def get_binary_file_download_link(file_object, filename, text):
    """Genera link download per oggetto file binario."""
    try:
        file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode()
        mime_type, _ = mimetypes.guess_type(filename); mime_type = mime_type or 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
    except Exception as e: st.warning(f"Errore link binario {filename}: {e}"); return f"<i>Errore link {filename}</i>"

def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    """Genera link download per grafico Plotly (HTML, PNG se kaleido installato)."""
    href_html = ""; href_png = ""
    safe_filename_base = re.sub(r'[^\w-]', '_', filename_base)
    try: # HTML
        html_filename = f"{safe_filename_base}.html"; buf_html = io.StringIO()
        fig.write_html(buf_html, include_plotlyjs='cdn'); buf_html.seek(0)
        b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="{html_filename}">{text_html}</a>'
    except Exception as e_html: print(f"Errore download HTML {safe_filename_base}: {e_html}"); href_html = "<i>Errore HTML</i>"
    try: # PNG (richiede kaleido)
        import importlib
        if importlib.util.find_spec("kaleido"):
            png_filename = f"{safe_filename_base}.png"; buf_png = io.BytesIO()
            fig.write_image(buf_png, format="png", scale=2); buf_png.seek(0)
            b64_png = base64.b64encode(buf_png.getvalue()).decode()
            href_png = f'<a href="data:image/png;base64,{b64_png}" download="{png_filename}">{text_png}</a>'
        else: print("Nota: 'kaleido' non installato, download PNG disabilitato.")
    except Exception as e_png: print(f"Errore download PNG {safe_filename_base}: {e_png}"); href_png = "<i>Errore PNG</i>"
    return f"{href_html} {href_png}".strip()

def get_download_link_for_file(filepath, link_text=None):
    """Genera link download per un file su disco."""
    if not os.path.exists(filepath): return f"<i>File non trovato: {os.path.basename(filepath)}</i>"
    filename = os.path.basename(filepath); link_text = link_text or f"Scarica {filename}"
    try:
        with open(filepath, "rb") as f: file_content = f.read()
        b64 = base64.b64encode(file_content).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(filepath); mime_type = mime_type or 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.error(f"Errore link per {filename}: {e}"); return f"<i>Errore link</i>"

# --- Funzione Estrazione ID GSheet ---
# (Nessuna modifica necessaria qui)
def extract_sheet_id(url):
    """Estrae l'ID del foglio Google da un URL."""
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

# --- Funzione Estrazione Etichetta Stazione ---
# (Nessuna modifica necessaria qui)
def get_station_label(col_name, short=False):
    """Restituisce etichetta leggibile per colonna sensore."""
    if col_name in STATION_COORDS:
        info = STATION_COORDS[col_name]
        loc_id = info.get('location_id')
        s_type = info.get('type', '')
        name = info.get('name', loc_id or col_name)
        if short and loc_id:
            sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == loc_id]
            if len(sensors_at_loc) > 1:
                type_abbr = {'Pioggia': 'P', 'Livello': 'L', 'Umidità': 'U'}.get(s_type, '?')
                label = f"{loc_id} ({type_abbr})"
            else: label = loc_id
            label = re.sub(r'\s*\(.*?\)\s*', '', label).strip()
            return label[:20] + ('...' if len(label) > 20 else '')
        else: return name
    label = col_name
    label = re.sub(r'\s*\[.*?\]|\s*\(.*?\)', '', label).strip()
    label = label.replace('Sensore ', '').replace('Livello Idrometrico ', '').replace(' - Pioggia Ora', '')
    label = label.replace(' - Livello Misa', '').replace(' - Livello Nevola', '')
    parts = label.split(' '); label = ' '.join(parts[:2])
    return label[:20] + ('...' if len(label) > 20 else '') if short else label

# --- Inizializzazione Session State ---
# (Nessuna modifica necessaria qui)
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scalers' not in st.session_state: st.session_state.active_scalers = None
if 'df' not in st.session_state: st.session_state.df = None
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [
         'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)', 'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)', HUMIDITY_COL_NAME,
         'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)', 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)', 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
         'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)', 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)' ]
if 'date_col_name_csv' not in st.session_state: st.session_state.date_col_name_csv = 'Data e Ora'
if 'dashboard_thresholds' not in st.session_state: st.session_state.dashboard_thresholds = DEFAULT_THRESHOLDS.copy()
if 'last_dashboard_data' not in st.session_state: st.session_state.last_dashboard_data = None
if 'last_dashboard_error' not in st.session_state: st.session_state.last_dashboard_error = None
if 'last_dashboard_fetch_time' not in st.session_state: st.session_state.last_dashboard_fetch_time = None
if 'active_alerts' not in st.session_state: st.session_state.active_alerts = []
if 'seq2seq_past_data_gsheet' not in st.session_state: st.session_state.seq2seq_past_data_gsheet = None
if 'seq2seq_last_ts_gsheet' not in st.session_state: st.session_state.seq2seq_last_ts_gsheet = None
if 'imported_sim_data_gs_df_lstm' not in st.session_state: st.session_state.imported_sim_data_gs_df_lstm = None
if 'imported_sim_start_time_gs_lstm' not in st.session_state: st.session_state.imported_sim_start_time_gs_lstm = None
if 'current_page' not in st.session_state: st.session_state.current_page = 'Dashboard'
if 'dash_custom_range_check' not in st.session_state: st.session_state.dash_custom_range_check = False
if 'dash_start_date' not in st.session_state: st.session_state.dash_start_date = datetime.now(italy_tz).date() - timedelta(days=7)
if 'dash_end_date' not in st.session_state: st.session_state.dash_end_date = datetime.now(italy_tz).date()
if 'full_gsheet_data_cache' not in st.session_state: st.session_state.full_gsheet_data_cache = None


# ==============================================================================
# --- LAYOUT STREAMLIT PRINCIPALE ---
# ==============================================================================
st.title('Modello Predittivo Idrologico')
st.caption('Applicazione per monitoraggio, simulazione e addestramento di modelli LSTM e Seq2Seq.')

# --- Sidebar ---
with st.sidebar:
    st.header('Impostazioni')

    # --- Caricamento Dati Storici CSV ---
    st.subheader('Dati Storici (per Analisi/Training)')
    uploaded_data_file = st.file_uploader('Carica CSV Dati Storici', type=['csv'], key="data_uploader", help="File CSV con dati storici, separatore ';', decimale ','")
    df = None; df_load_error = None; data_source_info = ""
    data_path_to_load = None; is_uploaded = False
    if uploaded_data_file is not None:
        data_path_to_load = uploaded_data_file; is_uploaded = True; data_source_info = f"File caricato: **{uploaded_data_file.name}**"
    elif os.path.exists(DEFAULT_DATA_PATH):
        if 'df' not in st.session_state or st.session_state.df is None or not st.session_state.get('uploaded_file_processed', False):
            data_path_to_load = DEFAULT_DATA_PATH; is_uploaded = False; data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
            st.session_state['uploaded_file_processed'] = False
        else: data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"; df = st.session_state.df
    elif 'df' not in st.session_state or st.session_state.df is None: df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato e nessun file caricato."

    process_file = False
    if is_uploaded and st.session_state.get('uploaded_file_processed') != uploaded_data_file.name: process_file = True; print("Processando nuovo file...")
    elif data_path_to_load == DEFAULT_DATA_PATH and not st.session_state.get('uploaded_file_processed'): process_file = True; print("Processando file default...")

    if process_file and data_path_to_load:
        with st.spinner("Caricamento e processamento dati CSV..."):
            try:
                read_args = {'sep': ';', 'decimal': ',', 'low_memory': False}; encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']; df_temp = None
                file_obj = data_path_to_load
                for enc in encodings_to_try:
                     try:
                         if hasattr(file_obj, 'seek'): file_obj.seek(0)
                         df_temp = pd.read_csv(file_obj, encoding=enc, **read_args); print(f"CSV letto con {enc}"); break
                     except UnicodeDecodeError: continue
                     except Exception as read_e: raise read_e
                if df_temp is None: raise ValueError(f"Impossibile leggere CSV con encodings: {encodings_to_try}")

                date_col_csv = st.session_state.date_col_name_csv
                if date_col_csv not in df_temp.columns: raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante.")
                try: df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
                except ValueError:
                     try: df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], errors='coerce'); st.caption(f"Formato data CSV non standard ('{date_col_csv}'). Tentativo inferenza.")
                     except Exception as e_date_csv_infer: raise ValueError(f"Errore conversione data CSV '{date_col_csv}': {e_date_csv_infer}")
                df_temp = df_temp.dropna(subset=[date_col_csv])
                if df_temp.empty: raise ValueError("Nessuna riga valida dopo pulizia data CSV.")
                df_temp = df_temp.sort_values(by=date_col_csv).reset_index(drop=True)

                features_to_clean = [col for col in df_temp.columns if col != date_col_csv]
                for col in features_to_clean:
                    if pd.api.types.is_object_dtype(df_temp[col]) or pd.api.types.is_string_dtype(df_temp[col]):
                         col_str = df_temp[col].astype(str).str.strip().replace(['N/A', '', '-', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                         if '.' in col_str.unique() and ',' in col_str.unique(): col_str = col_str.str.replace('.', '', regex=False)
                         col_str = col_str.str.replace(',', '.', regex=False)
                         df_temp[col] = pd.to_numeric(col_str, errors='coerce')
                    elif pd.api.types.is_numeric_dtype(df_temp[col]): pass
                    else: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

                numeric_cols = df_temp.select_dtypes(include=np.number).columns
                n_nan_before_fill = df_temp[numeric_cols].isnull().sum().sum()
                if n_nan_before_fill > 0:
                     st.caption(f"{n_nan_before_fill} NaN CSV. Eseguito ffill/bfill.")
                     df_temp.loc[:, numeric_cols] = df_temp[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                     n_nan_after_fill = df_temp[numeric_cols].isnull().sum().sum()
                     if n_nan_after_fill > 0:
                         st.warning(f"NaN residui ({n_nan_after_fill}) CSV. Riempiti con 0.")
                         df_temp.loc[:, numeric_cols] = df_temp[numeric_cols].fillna(0)

                st.session_state.df = df_temp
                if is_uploaded: st.session_state['uploaded_file_processed'] = uploaded_data_file.name
                else: st.session_state['uploaded_file_processed'] = True
                st.success(f"Dati CSV caricati ({len(st.session_state.df)} righe)."); df = st.session_state.df

            except Exception as e:
                df = None; st.session_state.df = None; st.session_state['uploaded_file_processed'] = None
                df_load_error = f'Errore caricamento/processamento CSV: {e}'; st.error(f"Errore CSV: {df_load_error}")
                print(traceback.format_exc())

    if df is None and 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        if data_source_info == "":
            if st.session_state.get('uploaded_file_processed') and isinstance(st.session_state.get('uploaded_file_processed'), str): data_source_info = f"File caricato: **{st.session_state['uploaded_file_processed']}** (da sessione)"
            elif st.session_state.get('uploaded_file_processed') == True: data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"

    if data_source_info: st.caption(data_source_info)
    if df_load_error and not data_source_info: st.error(df_load_error)
    data_ready_csv = df is not None
    st.divider()

    # --- Selezione Modello Predittivo ---
    st.subheader("Modello Predittivo (per Simulazione)")
    available_models_dict = find_available_models(MODELS_DIR)
    model_display_names = sorted(list(available_models_dict.keys()))
    MODEL_CHOICE_UPLOAD = "Carica File Manualmente (Solo LSTM)"; MODEL_CHOICE_NONE = "-- Nessun Modello Selezionato --"
    selection_options = [MODEL_CHOICE_NONE] + model_display_names
    current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

    if current_selection_name not in selection_options and current_selection_name != MODEL_CHOICE_UPLOAD:
        st.session_state.active_model_name = MODEL_CHOICE_NONE; current_selection_name = MODEL_CHOICE_NONE
        st.session_state.active_config = None; st.session_state.active_model = None; st.session_state.active_device = None; st.session_state.active_scalers = None

    try: current_index = selection_options.index(current_selection_name)
    except ValueError: current_index = 0

    selected_model_display_name = st.selectbox("Modello Pre-Addestrato:", selection_options, index=current_index, key="model_selector_predefined", help="Scegli un modello pre-addestrato (LSTM o Seq2Seq).")

    with st.expander("Carica Modello LSTM Manualmente", expanded=(current_selection_name == MODEL_CHOICE_UPLOAD)):
         is_upload_active = False
         m_f = st.file_uploader('File Modello (.pth)', type=['pth'], key="up_pth")
         sf_f = st.file_uploader('File Scaler Features (.joblib)', type=['joblib'], key="up_scf_lstm")
         st_f = st.file_uploader('File Scaler Target (.joblib)', type=['joblib'], key="up_sct_lstm")
         if m_f and sf_f and st_f:
            st.caption("Configura parametri modello LSTM caricato:")
            col1_up, col2_up = st.columns(2)
            with col1_up:
                iw_up = st.number_input("Input Window (steps)", 1, 500, 48, 1, key="up_in_steps_lstm")
                ow_up = st.number_input("Output Window (steps)", 1, 500, 24, 1, key="up_out_steps_lstm")
                hs_up = st.number_input("Hidden Size", 8, 1024, 128, 8, key="up_hid_lstm")
            with col2_up:
                nl_up = st.number_input("Numero Layers", 1, 8, 2, 1, key="up_lay_lstm")
                dr_up = st.slider("Dropout", 0.0, 0.9, 0.2, 0.05, key="up_drop_lstm")

            available_cols_up = list(df.columns.drop(st.session_state.date_col_name_csv, errors='ignore')) if data_ready_csv else st.session_state.feature_columns
            default_targets_up = [c for c in available_cols_up if 'Livello' in c or '[m]' in c][:1]
            default_features_up = [c for c in st.session_state.feature_columns if c in available_cols_up]
            targets_up = st.multiselect("Target Columns (Output)", available_cols_up, default=default_targets_up, key="up_targets_lstm")
            features_up = st.multiselect("Feature Columns (Input)", available_cols_up, default=default_features_up, key="up_features_lstm")

            if targets_up and features_up:
                if st.button("Carica Modello Manuale", key="load_manual_lstm", type="secondary"): is_upload_active = True; selected_model_display_name = MODEL_CHOICE_UPLOAD
            else: st.caption("Seleziona features e target per il modello caricato.")
         else: st.caption("Carica tutti e tre i file (.pth, scaler features, scaler target).")

    config_to_load = None; model_to_load = None; device_to_load = None; scalers_to_load = None
    load_error_sidebar = False; model_type_loaded = None

    if is_upload_active:
        st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
        temp_cfg_up = {"input_window": iw_up, "output_window": ow_up, "hidden_size": hs_up, "num_layers": nl_up, "dropout": dr_up, "feature_columns": features_up, "target_columns": targets_up, "name": "uploaded_lstm_model", "display_name": "Modello LSTM Caricato", "model_type": "LSTM"}
        model_to_load, device_to_load = load_specific_model(m_f, temp_cfg_up)
        temp_model_info_up = {"scaler_features_path": sf_f, "scaler_targets_path": st_f}
        scalers_tuple_up = load_specific_scalers(temp_cfg_up, temp_model_info_up)
        if model_to_load and scalers_tuple_up and isinstance(scalers_tuple_up, tuple) and len(scalers_tuple_up)==2:
            config_to_load = temp_cfg_up; scalers_to_load = scalers_tuple_up; model_type_loaded = "LSTM"
        else: load_error_sidebar = True; st.error("Errore caricamento modello/scaler LSTM manuale.")

    elif selected_model_display_name != MODEL_CHOICE_NONE:
        if selected_model_display_name != st.session_state.get('active_model_name'):
            st.session_state.active_model_name = selected_model_display_name
            if selected_model_display_name in available_models_dict:
                model_info = available_models_dict[selected_model_display_name]
                config_to_load = load_model_config(model_info["config_path"])
                if config_to_load:
                    config_to_load["pth_path"] = model_info["pth_path"]; config_to_load["config_name"] = model_info["config_name"]; config_to_load["display_name"] = selected_model_display_name
                    model_type_loaded = model_info.get("model_type", "LSTM"); config_to_load["model_type"] = model_type_loaded
                    model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
                    scalers_to_load = load_specific_scalers(config_to_load, model_info)
                    if not (model_to_load and scalers_to_load): load_error_sidebar = True; config_to_load = None
                else: load_error_sidebar = True
            else: st.error(f"Modello '{selected_model_display_name}' non trovato."); load_error_sidebar = True
        elif st.session_state.get('active_model') and st.session_state.get('active_config'):
             config_to_load = st.session_state.active_config; model_to_load = st.session_state.active_model
             device_to_load = st.session_state.active_device; scalers_to_load = st.session_state.active_scalers
             model_type_loaded = config_to_load.get("model_type", "LSTM")

    elif selected_model_display_name == MODEL_CHOICE_NONE and st.session_state.get('active_model_name') != MODEL_CHOICE_NONE:
         st.session_state.active_model_name = MODEL_CHOICE_NONE
         st.session_state.active_config = None; st.session_state.active_model = None; st.session_state.active_device = None; st.session_state.active_scalers = None
         config_to_load = None; model_to_load = None

    if config_to_load and model_to_load and device_to_load and scalers_to_load:
        scalers_valid = (isinstance(scalers_to_load, dict) and all(scalers_to_load.values())) or \
                        (isinstance(scalers_to_load, tuple) and all(s is not None for s in scalers_to_load))
        if scalers_valid:
            st.session_state.active_config = config_to_load; st.session_state.active_model = model_to_load
            st.session_state.active_device = device_to_load; st.session_state.active_scalers = scalers_to_load
        else:
             st.error("Errore caricamento scaler associati."); load_error_sidebar = True
             st.session_state.active_config = None; st.session_state.active_model = None; st.session_state.active_device = None; st.session_state.active_scalers = None

    active_config_sess = st.session_state.get('active_config'); active_model_sess = st.session_state.get('active_model')
    active_model_name_sess = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

    if active_model_sess and active_config_sess:
        cfg = active_config_sess; model_type_sess = cfg.get('model_type', 'LSTM'); display_feedback_name = cfg.get("display_name", active_model_name_sess)
        st.success(f"Modello Attivo: **{display_feedback_name}** ({model_type_sess})")
        if model_type_sess == "Seq2Seq": st.caption(f"Input: {cfg['input_window_steps']}s | Forecast: {cfg['forecast_window_steps']}s | Output: {cfg['output_window_steps']}s")
        else: st.caption(f"Input: {cfg['input_window']}s | Output: {cfg['output_window']}s")
    elif load_error_sidebar and active_model_name_sess not in [MODEL_CHOICE_NONE]: st.error(f"Caricamento modello '{active_model_name_sess}' fallito.")
    elif active_model_name_sess == MODEL_CHOICE_UPLOAD and not active_model_sess: st.info("Completa caricamento manuale LSTM.")
    elif active_model_name_sess == MODEL_CHOICE_NONE: st.info("Nessun modello selezionato per simulazione.")

    st.divider()

    # --- Configurazione Soglie Dashboard ---
    st.subheader("Configurazione Soglie Dashboard")
    with st.expander("Modifica Soglie di Allerta (per Dashboard)"):
        temp_thresholds = st.session_state.dashboard_thresholds.copy()
        monitorable_cols_thresh = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
        cols_thresh_ui = st.columns(2); col_idx_thresh = 0
        for col in monitorable_cols_thresh:
            with cols_thresh_ui[col_idx_thresh % 2]:
                label_short = get_station_label(col, short=False)
                is_level = 'Livello' in col or '(m)' in col or '(mt)' in col; is_humidity = 'Umidit' in col
                step = 0.1 if is_level else (1.0 if is_humidity else 0.5); fmt = "%.2f" if is_level else "%.1f"; min_v = 0.0
                current_threshold = st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, 0.0))
                new_threshold = st.number_input(label=f"{label_short}", value=float(current_threshold), min_value=min_v, step=step, format=fmt, key=f"thresh_{col}", help=f"Soglia allerta per: {col}")
                if new_threshold != current_threshold: temp_thresholds[col] = new_threshold
            col_idx_thresh += 1
        if st.button("Salva Soglie", key="save_thresholds", type="secondary"):
            st.session_state.dashboard_thresholds = temp_thresholds.copy(); st.success("Soglie aggiornate!"); pytime.sleep(0.5); st.rerun()

    st.divider()

    # --- Menu Navigazione ---
    st.header('Menu Navigazione')
    model_ready = active_model_sess is not None and active_config_sess is not None
    radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
    radio_captions = []; disabled_options = []; default_page_idx = 0

    for i, opt in enumerate(radio_options):
        caption = ""; disabled = False
        if opt == 'Dashboard': caption = "Monitoraggio GSheet"
        elif opt == 'Simulazione':
            if not model_ready: caption = "Richiede Modello"; disabled = True
            else: caption = f"Previsioni ({active_config_sess.get('model_type', 'LSTM')})"
        elif opt == 'Analisi Dati Storici':
            if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
            else: caption = "Esplora dati CSV"
        elif opt == 'Allenamento Modello':
            if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
            else: caption = "Allena modello"
        radio_captions.append(caption); disabled_options.append(disabled)
        if opt == 'Dashboard': default_page_idx = i

    current_page = st.session_state.get('current_page', radio_options[default_page_idx])
    try:
        current_page_index = radio_options.index(current_page)
        if disabled_options[current_page_index]:
            st.warning(f"Pagina '{current_page}' non disponibile ({radio_captions[current_page_index]}). Reindirizzato a Dashboard.")
            current_page = radio_options[default_page_idx]; st.session_state['current_page'] = current_page
    except ValueError: current_page = radio_options[default_page_idx]; st.session_state['current_page'] = current_page

    chosen_page = st.radio('Scegli funzionalità:', options=radio_options, captions=radio_captions, index=radio_options.index(current_page), key='page_selector_radio')

    if chosen_page != current_page:
        chosen_page_index = radio_options.index(chosen_page)
        if not disabled_options[chosen_page_index]:
            st.session_state['current_page'] = chosen_page; st.rerun()
        else: st.error(f"Pagina '{chosen_page}' non disponibile. {radio_captions[chosen_page_index]}.")
    page = current_page

# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================
active_config = st.session_state.get('active_config')
active_model = st.session_state.get('active_model')
active_device = st.session_state.get('active_device')
active_scalers = st.session_state.get('active_scalers')
active_model_type = active_config.get("model_type", "LSTM") if active_config else "LSTM"
model_ready = active_model is not None and active_config is not None
df_current_csv = st.session_state.get('df', None)
data_ready_csv = df_current_csv is not None
date_col_name_csv = st.session_state.date_col_name_csv

# --- PAGINA DASHBOARD ---
if page == 'Dashboard':
    st.header('Dashboard Monitoraggio Idrologico')
    if "GOOGLE_CREDENTIALS" not in st.secrets: st.error("Errore Config: Credenziali Google mancanti."); st.stop()

    st.checkbox("Visualizza intervallo date personalizzato", key="dash_custom_range_check", value=st.session_state.dash_custom_range_check)

    df_dashboard = None; error_msg = None; actual_fetch_time = None; data_source_mode = ""

    if st.session_state.dash_custom_range_check:
        st.subheader("Seleziona Intervallo Temporale")
        col_date1, col_date2 = st.columns(2)
        with col_date1: start_date_select = st.date_input("Data Inizio", key="dash_start_date", value=st.session_state.dash_start_date)
        with col_date2: end_date_select = st.date_input("Data Fine", key="dash_end_date", value=st.session_state.dash_end_date)
        if start_date_select != st.session_state.dash_start_date: st.session_state.dash_start_date = start_date_select
        if end_date_select != st.session_state.dash_end_date: st.session_state.dash_end_date = end_date_select
        if start_date_select > end_date_select: st.error("Data Fine < Data Inizio."); st.stop()

        start_dt_filter = italy_tz.localize(datetime.combine(start_date_select, time.min))
        end_dt_filter = italy_tz.localize(datetime.combine(end_date_select, time.max))

        now_ts = pytime.time(); cache_time_key_full = int(now_ts // (DASHBOARD_REFRESH_INTERVAL_SECONDS * 2))
        df_dashboard_full, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(cache_time_key_full, GSHEET_ID, GSHEET_RELEVANT_COLS, GSHEET_DATE_COL, GSHEET_DATE_FORMAT, fetch_all=True)

        if df_dashboard_full is not None:
            st.session_state.full_gsheet_data_cache = df_dashboard_full; st.session_state.last_dashboard_error = error_msg; st.session_state.last_dashboard_fetch_time = actual_fetch_time
            try:
                if GSHEET_DATE_COL in df_dashboard_full.columns:
                     if not pd.api.types.is_datetime64_any_dtype(df_dashboard_full[GSHEET_DATE_COL]): df_dashboard_full[GSHEET_DATE_COL] = pd.to_datetime(df_dashboard_full[GSHEET_DATE_COL], errors='coerce')
                     if df_dashboard_full[GSHEET_DATE_COL].dt.tz is None: df_dashboard_full[GSHEET_DATE_COL] = df_dashboard_full[GSHEET_DATE_COL].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                     else: df_dashboard_full[GSHEET_DATE_COL] = df_dashboard_full[GSHEET_DATE_COL].dt.tz_convert(italy_tz)
                     df_dashboard = df_dashboard_full[(df_dashboard_full[GSHEET_DATE_COL] >= start_dt_filter) & (df_dashboard_full[GSHEET_DATE_COL] <= end_dt_filter)].copy()
                     data_source_mode = f"Dati da {start_date_select.strftime('%d/%m/%Y')} a {end_date_select.strftime('%d/%m/%Y')}"
                     if df_dashboard.empty: st.warning(f"Nessun dato trovato nell'intervallo selezionato.")
                else: st.error("Colonna data GSheet mancante per filtraggio."); df_dashboard = None
            except Exception as e_filter: st.error(f"Errore filtraggio dati: {e}"); df_dashboard = None
        else:
            df_dashboard_full = st.session_state.get('full_gsheet_data_cache')
            if df_dashboard_full is not None:
                 st.warning("Recupero GSheet fallito, uso dati cache.")
                 try:
                      if GSHEET_DATE_COL in df_dashboard_full.columns:
                         if df_dashboard_full[GSHEET_DATE_COL].dt.tz is None: df_dashboard_full[GSHEET_DATE_COL] = df_dashboard_full[GSHEET_DATE_COL].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                         else: df_dashboard_full[GSHEET_DATE_COL] = df_dashboard_full[GSHEET_DATE_COL].dt.tz_convert(italy_tz)
                         df_dashboard = df_dashboard_full[(df_dashboard_full[GSHEET_DATE_COL] >= start_dt_filter) & (df_dashboard_full[GSHEET_DATE_COL] <= end_dt_filter)].copy()
                         data_source_mode = f"Dati (cache) da {start_date_select.strftime('%d/%m/%Y')} a {end_date_select.strftime('%d/%m/%Y')}"
                 except Exception: pass
            else: st.error(f"Recupero GSheet fallito: {error_msg}")
            st.session_state.last_dashboard_error = error_msg

    else: # --- Ultime 24 Ore ---
        now_ts = pytime.time(); cache_time_key_recent = int(now_ts // DASHBOARD_REFRESH_INTERVAL_SECONDS)
        df_dashboard, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(cache_time_key_recent, GSHEET_ID, GSHEET_RELEVANT_COLS, GSHEET_DATE_COL, GSHEET_DATE_FORMAT, fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS)
        st.session_state.last_dashboard_data = df_dashboard; st.session_state.last_dashboard_error = error_msg
        if df_dashboard is not None or error_msg is None: st.session_state.last_dashboard_fetch_time = actual_fetch_time
        data_source_mode = f"Ultime {DASHBOARD_HISTORY_ROWS // 2} ore circa"

    # --- Visualizzazione Stato e Refresh ---
    col_status, col_refresh_btn = st.columns([4, 1])
    with col_status:
        last_fetch_dt_sess = st.session_state.get('last_dashboard_fetch_time')
        if last_fetch_dt_sess:
             fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess; fetch_secs_ago = int(fetch_time_ago.total_seconds())
             status_text = f"{data_source_mode}. Aggiornato alle {last_fetch_dt_sess.strftime('%H:%M:%S')} ({fetch_secs_ago}s fa)."
             if not st.session_state.dash_custom_range_check: status_text += f" Refresh auto ogni {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
             st.caption(status_text)
        else: st.caption("Recupero GSheet in corso o fallito...")
    with col_refresh_btn:
        if st.button("Aggiorna Ora", key="dash_refresh_button"):
            fetch_gsheet_dashboard_data.clear(); fetch_sim_gsheet_data.clear(); st.session_state.full_gsheet_data_cache = None
            st.success("Cache GSheet pulita. Ricarico..."); pytime.sleep(0.5); st.rerun()

    last_error_sess = st.session_state.get('last_dashboard_error')
    if last_error_sess:
        if any(x in last_error_sess for x in ["API", "Foglio", "Credenziali", "Errore:"]): st.error(f"Errore GSheet: {last_error_sess}")
        else: st.warning(f"Attenzione Dati GSheet: {last_error_sess}")

    # --- Visualizzazione Dati e Grafici ---
    if df_dashboard is not None and not df_dashboard.empty:
        if GSHEET_DATE_COL in df_dashboard.columns:
            latest_row_data = df_dashboard.iloc[-1]
            last_update_time = latest_row_data.get(GSHEET_DATE_COL)
            if pd.notna(last_update_time) and isinstance(last_update_time, pd.Timestamp):
                 time_now_italy = datetime.now(italy_tz)
                 if last_update_time.tzinfo is None: last_update_time = italy_tz.localize(last_update_time)
                 else: last_update_time = last_update_time.tz_convert(italy_tz)
                 time_delta = time_now_italy - last_update_time; minutes_ago = int(time_delta.total_seconds() // 60)
                 time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')
                 if minutes_ago < 2: time_ago_str = "pochi istanti fa"
                 elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
                 else: time_ago_str = f"circa {minutes_ago // 60}h e {minutes_ago % 60}min fa"
                 st.markdown(f"**Ultimo rilevamento nel periodo:** {time_str} ({time_ago_str})")
                 if not st.session_state.dash_custom_range_check and minutes_ago > 90: st.warning(f"Attenzione: Ultimo dato ricevuto > {minutes_ago} minuti fa.")
            else: st.warning("Timestamp ultimo rilevamento non valido.")
        else: st.warning("Colonna data GSheet mancante.")

        st.divider()
        # --- Tabella Valori Attuali (include già Q) ---
        st.subheader("Valori Ultimo Rilevamento, Stato Allerta e Portata")
        cols_to_monitor = [col for col in df_dashboard.columns if col != GSHEET_DATE_COL]
        table_rows = []; current_alerts = []
        for col_name in cols_to_monitor:
            current_value_H = latest_row_data.get(col_name) if 'latest_row_data' in locals() else None
            threshold = st.session_state.dashboard_thresholds.get(col_name)
            alert_active = False; value_numeric = np.nan; value_display = "N/D"; unit = ""
            unit_match = re.search(r'\((.*?)\)|\[(.*?)\]', col_name)
            if unit_match: unit = (unit_match.group(1) or unit_match.group(2) or "").strip()

            if pd.notna(current_value_H) and isinstance(current_value_H, (int, float, np.number)):
                 value_numeric = float(current_value_H)
                 fmt_spec = ".1f" if unit in ['mm', '%'] else ".2f"
                 try: value_display = f"{value_numeric:{fmt_spec}} {unit}".strip()
                 except ValueError: value_display = f"{value_numeric} {unit}".strip()
                 if threshold is not None and isinstance(threshold, (int, float, np.number)) and value_numeric >= float(threshold):
                      alert_active = True; current_alerts.append((col_name, value_numeric, threshold))
            elif pd.notna(current_value_H): value_display = f"{current_value_H} (?)"

            portata_Q = None; portata_display = "N/A"
            sensor_info = STATION_COORDS.get(col_name)
            if sensor_info:
                sensor_code = sensor_info.get('sensor_code')
                if sensor_code and sensor_code in RATING_CURVES:
                    portata_Q = calculate_discharge(sensor_code, value_numeric)
                    if portata_Q is not None and pd.notna(portata_Q): portata_display = f"{portata_Q:.2f}"
                    else: portata_display = "-"

            status = "ALLERTA" if alert_active else ("OK" if pd.notna(value_numeric) else "N/D")
            threshold_display = f"{float(threshold):.1f}" if threshold is not None else "-"
            table_rows.append({
                "Stazione": get_station_label(col_name, short=True), "Valore H": value_display, "Portata Q (m³/s)": portata_display,
                "Soglia H": threshold_display, "Stato": status, "Valore Numerico H": value_numeric, "Soglia Numerica H": float(threshold) if threshold else None })

        df_display = pd.DataFrame(table_rows)
        def highlight_alert(row): return ['background-color: rgba(255, 0, 0, 0.1);'] * len(row) if row['Stato'] == 'ALLERTA' else [''] * len(row)
        st.dataframe( df_display.style.apply(highlight_alert, axis=1),
            column_order=["Stazione", "Valore H", "Portata Q (m³/s)", "Soglia H", "Stato"], hide_index=True, use_container_width=True,
            column_config={ "Valore Numerico H": None, "Soglia Numerica H": None } )
        st.session_state.active_alerts = current_alerts

        st.divider()
        # --- Grafico Comparativo Configurabile (MODIFICATO per Portata Q) ---
        if GSHEET_DATE_COL in df_dashboard.columns:
            st.subheader("Grafico Storico Comparativo (Periodo Selezionato)")
            sensor_options_compare = {get_station_label(col, short=True): col for col in cols_to_monitor}
            default_selection_labels = [label for label, col in sensor_options_compare.items() if 'Livello' in col][:3] or list(sensor_options_compare.keys())[:2]
            selected_labels_compare = st.multiselect("Seleziona sensori da confrontare:", options=list(sensor_options_compare.keys()), default=default_selection_labels, key="compare_select_multi")
            selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare]

            if selected_cols_compare:
                fig_compare = go.Figure()
                x_axis_data = df_dashboard[GSHEET_DATE_COL]
                show_yaxis2 = False # Flag per mostrare asse Q

                # Colori Plotly di default
                default_colors = go.layout.Template().layout.colorway

                # Aggiungi tracce H (asse y1)
                for i, col in enumerate(selected_cols_compare):
                    color_idx = i % len(default_colors)
                    fig_compare.add_trace(go.Scatter(
                        x=x_axis_data, y=df_dashboard[col], mode='lines', name=get_station_label(col, short=True),
                        line=dict(color=default_colors[color_idx]), yaxis='y1'
                    ))

                # Aggiungi tracce Q (asse y2) - Se applicabile
                for i, col in enumerate(selected_cols_compare):
                    sensor_info_comp = STATION_COORDS.get(col)
                    if sensor_info_comp:
                        sensor_code_comp = sensor_info_comp.get('sensor_code')
                        if sensor_code_comp and sensor_code_comp in RATING_CURVES:
                            h_values_comp = df_dashboard[col].astype(float).values # Assicura float
                            q_values_comp = calculate_discharge_vectorized(sensor_code_comp, h_values_comp)
                            valid_q_mask_comp = pd.notna(q_values_comp) & (q_values_comp >= 0) # Maschera per Q valido >= 0

                            if np.any(valid_q_mask_comp):
                                show_yaxis2 = True # Abilita asse Y2
                                color_idx = i % len(default_colors) # Usa stesso colore di H
                                fig_compare.add_trace(go.Scatter(
                                    x=x_axis_data[valid_q_mask_comp], y=q_values_comp[valid_q_mask_comp],
                                    mode='lines', name=f"Q {get_station_label(col, short=True)}", # Aggiungi "Q "
                                    line=dict(color=default_colors[color_idx], dash='dot'), # Stesso colore, linea tratteggiata
                                    yaxis='y2'
                                ))

                # Layout Grafico Comparativo
                title_compare = f"Andamento Storico Comparato H/Q ({data_source_mode})"
                fig_compare.update_layout(
                    title=title_compare, xaxis_title='Data e Ora', height=500, hovermode="x unified", template="plotly_white",
                    yaxis=dict(title="Livello H / Pioggia / Umidità", side="left", rangemode='tozero'), # Titolo Y1 generico
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) # Legenda sotto
                )
                # Aggiungi asse Y2 solo se necessario
                if show_yaxis2:
                    fig_compare.update_layout(
                        yaxis2=dict(title="Portata Q (m³/s)", overlaying="y", side="right", rangemode='tozero', showgrid=False)
                    )

                st.plotly_chart(fig_compare, use_container_width=True)
                compare_filename_base = f"compare_HQ_{'_'.join(sl.replace(' ','_') for sl in selected_labels_compare)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_compare, compare_filename_base), unsafe_allow_html=True)

            st.divider()
            # --- Grafici Individuali (MODIFICATO per Portata Q) ---
            st.subheader("Grafici Storici Individuali H/Q (Periodo Selezionato)")
            num_cols_individual = 3
            graph_cols = st.columns(num_cols_individual)
            col_idx_graph = 0
            x_axis_data_indiv = df_dashboard[GSHEET_DATE_COL]

            for col_name in cols_to_monitor:
                with graph_cols[col_idx_graph % num_cols_individual]:
                    label_individual = get_station_label(col_name, short=True)
                    unit_match_indiv = re.search(r'\((.*?)\)|\[(.*?)\]', col_name)
                    unit_indiv = ""
                    if unit_match_indiv: unit_indiv = f"({(unit_match_indiv.group(1) or unit_match_indiv.group(2) or '').strip()})"

                    yaxis_title_individual_h = f"Valore H {unit_indiv}".strip()
                    fig_individual = go.Figure()

                    # Traccia H (Y1)
                    fig_individual.add_trace(go.Scatter(x=x_axis_data_indiv, y=df_dashboard[col_name], mode='lines', name=f"H {unit_indiv.strip()}", yaxis='y1'))

                    # Soglia H (basata su dashboard thresholds)
                    threshold_individual_h = st.session_state.dashboard_thresholds.get(col_name)
                    if threshold_individual_h is not None and isinstance(threshold_individual_h, (int, float)) and ('Livello' in col_name or '(m)' in col_name or '(mt)' in col_name):
                        fig_individual.add_hline(y=threshold_individual_h, line_dash="dash", line_color="red", annotation_text=f"Soglia H ({float(threshold_individual_h):.1f})", annotation_position="bottom right", layer="below")

                    # Traccia Q (Y2) - Condizionale
                    sensor_info_indiv = STATION_COORDS.get(col_name)
                    has_discharge_indiv = False
                    if sensor_info_indiv:
                        sensor_code_indiv = sensor_info_indiv.get('sensor_code')
                        if sensor_code_indiv and sensor_code_indiv in RATING_CURVES:
                             h_values_indiv = df_dashboard[col_name].astype(float).values
                             q_values_indiv = calculate_discharge_vectorized(sensor_code_indiv, h_values_indiv)
                             valid_q_mask_indiv = pd.notna(q_values_indiv) & (q_values_indiv >= 0)
                             if np.any(valid_q_mask_indiv):
                                 has_discharge_indiv = True
                                 fig_individual.add_trace(go.Scatter(
                                     x=x_axis_data_indiv[valid_q_mask_indiv], y=q_values_indiv[valid_q_mask_indiv],
                                     mode='lines', name='Q (m³/s)', line=dict(color='firebrick', dash='dot'), yaxis='y2'
                                 ))

                    # Layout Grafico Individuale
                    fig_individual.update_layout(
                        title=f"{label_individual}", xaxis_title=None, height=300, hovermode="x unified",
                        margin=dict(t=40, b=30, l=50, r=50), template="plotly_white", # Aumentato margine destro per y2
                        yaxis=dict(title=yaxis_title_individual_h, side="left", rangemode='tozero', titlefont_color="#1f77b4", tickfont_color="#1f77b4"),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) # Legenda sotto
                    )
                    if has_discharge_indiv:
                        fig_individual.update_layout(
                             yaxis2=dict(title="Portata Q (m³/s)", overlaying="y", side="right", rangemode='tozero', showgrid=False, titlefont_color="firebrick", tickfont_color="firebrick")
                        )

                    st.plotly_chart(fig_individual, use_container_width=True)
                    ind_filename_base = f"sensor_HQ_{label_individual.replace(' ','_').replace('(','').replace(')','')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    st.markdown(get_plotly_download_link(fig_individual, ind_filename_base, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)
                col_idx_graph += 1
        else: st.info("Grafici storici non disponibili (colonna data GSheet mancante).")

        st.divider()
        # --- Riepilogo Alert ---
        st.subheader("Riepilogo Allerte (basato su ultimo valore nel periodo)")
        active_alerts_sess = st.session_state.get('active_alerts', [])
        if active_alerts_sess:
             st.warning("**Allerte Attive (Valori H >= Soglia H):**")
             alert_md = ""; sorted_alerts = sorted(active_alerts_sess, key=lambda x: get_station_label(x[0], short=False))
             for col, val, thr in sorted_alerts:
                 label_alert = get_station_label(col, short=False); sensor_type_alert = STATION_COORDS.get(col, {}).get('type', ''); type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                 unit_match_alert = re.search(r'\((.*?)\)|\[(.*?)\]', col); unit_alert = "";
                 if unit_match_alert: unit_alert = f"({(unit_match_alert.group(1) or unit_match_alert.group(2) or '').strip()})"
                 val_fmt = f"{val:.1f}" if unit_alert in ['(mm)', '(%)'] else f"{val:.2f}"
                 thr_fmt = f"{float(thr):.1f}" if isinstance(thr, (int, float, np.number)) else str(thr)
                 alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit_alert}** >= Soglia **{thr_fmt}{unit_alert}**\n"
             st.markdown(alert_md)
        else: st.success("Nessuna soglia H superata nell'ultimo rilevamento del periodo.")

    elif df_dashboard is not None and df_dashboard.empty and st.session_state.dash_custom_range_check: pass # Messaggio già mostrato
    elif df_dashboard is not None and df_dashboard.empty and not st.session_state.dash_custom_range_check: st.warning("Recupero GSheet ok, ma nessun dato nelle ultime ore.")
    elif not st.session_state.last_dashboard_error: st.info("Recupero dati dashboard in corso...")

    # --- Auto Refresh JS ---
    if not st.session_state.dash_custom_range_check:
        try:
            component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"
            js_code = f"(function() {{ const k='{component_key}'; if (window[k]) clearInterval(window[k]); window[k] = setInterval(() => {{ if (window.streamlitHook && typeof window.streamlitHook.rerunScript === 'function') window.streamlitHook.rerunScript(null); }}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000}); }})();"
            streamlit_js_eval(js_expressions=js_code, key=component_key, want_output=False)
        except Exception as e_js: st.caption(f"Nota: Impossibile auto-refresh ({e_js})")
    else:
         try: # Pulisce timer se esiste
              component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"
              js_clear_code = f"(function() {{ const k='{component_key}'; if (window[k]) {{ clearInterval(window[k]); window[k]=null; }} }})();"
              streamlit_js_eval(js_expressions=js_clear_code, key=f"{component_key}_clear", want_output=False)
         except Exception as e_js_clear: st.caption(f"Nota: Impossibile pulire timer auto-refresh ({e_js_clear})")


# --- PAGINA SIMULAZIONE ---
elif page == 'Simulazione':
    st.header(f'Simulazione Idrologica ({active_model_type})')
    if not model_ready: st.warning("Seleziona Modello attivo dalla sidebar."); st.stop()

    st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}** ({active_model_type})")
    target_columns_model = []
    if active_model_type == "Seq2Seq":
        st.caption(f"Input: {active_config['input_window_steps']}s | Forecast: {active_config['forecast_window_steps']}s | Output: {active_config['output_window_steps']}s")
        with st.expander("Dettagli Colonne Modello Seq2Seq"):
             st.markdown("**Feature Storiche:**"); st.caption(f"`{', '.join(active_config['all_past_feature_columns'])}`")
             st.markdown("**Feature Forecast:**"); st.caption(f"`{', '.join(active_config['forecast_input_columns'])}`")
             st.markdown("**Target (Output H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        target_columns_model = active_config['target_columns']
    else: # LSTM
        feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns)
        st.caption(f"Input: {active_config['input_window']} steps | Output: {active_config['output_window']} steps")
        with st.expander("Dettagli Colonne Modello LSTM"):
             st.markdown("**Feature Input:**"); st.caption(f"`{', '.join(feature_columns_current_model)}`")
             st.markdown("**Target (Output H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        target_columns_model = active_config['target_columns']; input_steps = active_config['input_window']

    st.divider()

    # --- SEZIONE INPUT DATI SIMULAZIONE (include già Q nei risultati/grafici tramite plot_predictions) ---
    if active_model_type == "Seq2Seq":
         st.subheader(f"Preparazione Dati Input Simulazione Seq2Seq")
         st.markdown("**1. Recupero Dati Storici (Input Encoder)**")
         st.caption(f"Recupero ultimi {active_config['input_window_steps']} steps da Google Sheet (ID: `{GSHEET_ID}`).")

         past_feature_cols_s2s = active_config["all_past_feature_columns"]; date_col_model_name = st.session_state.date_col_name_csv
         column_mapping_gsheet_to_past_s2s = {
             'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)', 'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)', 'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
             'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)', 'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME,
             'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)', 'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
             'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', 'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
             'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)', GSHEET_DATE_COL: date_col_model_name }
         required_model_cols_for_mapping = past_feature_cols_s2s + [date_col_model_name]
         column_mapping_gsheet_to_past_s2s_filtered = { gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_past_s2s.items() if model_col in required_model_cols_for_mapping }
         missing_past_features_in_map = list(set(past_feature_cols_s2s) - (set(column_mapping_gsheet_to_past_s2s_filtered.values()) - {date_col_model_name}))
         imputed_values_sim_past_s2s = {}
         if missing_past_features_in_map:
             st.caption(f"Feature storiche non mappate da GSheet: `{', '.join(missing_past_features_in_map)}`. Imputate.")
             for missing_f in missing_past_features_in_map: imputed_values_sim_past_s2s[missing_f] = 0.0 # Semplice imputazione 0

         fetch_error_gsheet_s2s = None
         if st.button("Carica/Aggiorna Storico da GSheet", key="fetch_gsh_s2s_base"):
              fetch_sim_gsheet_data.clear()
              with st.spinner("Recupero dati storici..."):
                  required_cols_fetch = past_feature_cols_s2s + [date_col_model_name]
                  imported_df_past, import_err_past, last_ts_past = fetch_sim_gsheet_data(GSHEET_ID, active_config['input_window_steps'], GSHEET_DATE_COL, GSHEET_DATE_FORMAT, column_mapping_gsheet_to_past_s2s_filtered, required_cols_fetch, imputed_values_sim_past_s2s)
              if import_err_past: st.error(f"Recupero storico fallito: {import_err_past}"); st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = import_err_past
              elif imported_df_past is not None:
                  try:
                      final_past_df = imported_df_past[past_feature_cols_s2s]
                      st.success(f"Recuperate {len(final_past_df)} righe storiche."); st.session_state.seq2seq_past_data_gsheet = final_past_df
                      st.session_state.seq2seq_last_ts_gsheet = last_ts_past if last_ts_past else datetime.now(italy_tz)
                      fetch_error_gsheet_s2s = None; st.rerun()
                  except KeyError as e_cols: st.error(f"Errore colonne storiche: {e_cols}"); st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = f"Errore colonne: {e_cols}"
              else: st.error("Recupero storico non riuscito."); fetch_error_gsheet_s2s = "Errore recupero."

         past_data_loaded_s2s = st.session_state.get('seq2seq_past_data_gsheet') is not None
         if past_data_loaded_s2s:
             st.caption("Dati storici caricati."); # with st.expander("Mostra dati storici (Input Encoder)"): st.dataframe(st.session_state.seq2seq_past_data_gsheet.round(3)) # Rimosso per brevità
         elif fetch_error_gsheet_s2s: st.warning(f"Caricamento fallito ({fetch_error_gsheet_s2s}). Clicca per riprovare.")
         else: st.info("Clicca 'Carica/Aggiorna Storico da GSheet'.")

         if past_data_loaded_s2s:
             st.markdown(f"**2. Inserisci Previsioni Future ({active_config['forecast_window_steps']} steps)**")
             forecast_input_cols_s2s = active_config['forecast_input_columns']
             st.caption(f"Inserisci valori previsti per: `{', '.join(forecast_input_cols_s2s)}`")
             forecast_steps_s2s = active_config['forecast_window_steps']; forecast_df_initial = pd.DataFrame(index=range(forecast_steps_s2s), columns=forecast_input_cols_s2s)
             last_known_past_data = st.session_state.seq2seq_past_data_gsheet.iloc[-1]
             for col in forecast_input_cols_s2s:
                  if 'pioggia' in col.lower() or 'cumulata' in col.lower(): forecast_df_initial[col] = 0.0
                  elif col in last_known_past_data.index: forecast_df_initial[col] = last_known_past_data.get(col, 0.0)
                  else: forecast_df_initial[col] = 0.0
             edited_forecast_df = st.data_editor( forecast_df_initial.round(2), key="seq2seq_forecast_editor", num_rows="fixed", use_container_width=True,
                 column_config={ col: st.column_config.NumberColumn( label=get_station_label(col, short=True), help=f"Valore previsto {col}",
                         min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower() or 'livello' in col.lower()) else None, max_value=100.0 if 'umidit' in col.lower() else None,
                         format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f", step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1), required=True
                     ) for col in forecast_input_cols_s2s } )
             forecast_data_valid_s2s = False
             if edited_forecast_df is not None and not edited_forecast_df.isnull().any().any() and edited_forecast_df.shape == (forecast_steps_s2s, len(forecast_input_cols_s2s)):
                 try: edited_forecast_df_float = edited_forecast_df.astype(float); forecast_data_valid_s2s = True
                 except ValueError: st.warning("Valori non numerici nella tabella previsioni.")
             else: st.warning("Completa/correggi tabella previsioni.")

             st.divider()
             st.markdown("**3. Esegui Simulazione Seq2Seq**")
             can_run_s2s_sim = past_data_loaded_s2s and forecast_data_valid_s2s
             if st.button("Esegui Simulazione Seq2Seq", disabled=not can_run_s2s_sim, type="primary", key="run_s2s_sim_button"):
                  if not can_run_s2s_sim: st.error("Mancano dati storici o previsioni valide.")
                  else:
                       predictions_s2s = None; start_pred_time_s2s = None
                       with st.spinner("Simulazione Seq2Seq..."):
                           past_data_np = st.session_state.seq2seq_past_data_gsheet[past_feature_cols_s2s].astype(float).values
                           future_forecast_np = edited_forecast_df[forecast_input_cols_s2s].astype(float).values
                           predictions_s2s = predict_seq2seq(active_model, past_data_np, future_forecast_np, active_scalers, active_config, active_device)
                           start_pred_time_s2s = st.session_state.get('seq2seq_last_ts_gsheet', datetime.now(italy_tz))

                       if predictions_s2s is not None:
                           output_steps_actual = predictions_s2s.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                           st.subheader(f'Risultato Simulazione Seq2Seq: {total_hours_output_actual:.1f} ore')
                           st.caption(f"A partire da: {start_pred_time_s2s.strftime('%d/%m/%Y %H:%M:%S %Z')}")

                           results_df_s2s = pd.DataFrame(predictions_s2s, columns=target_columns_model)
                           q_cols_to_add_s2s = {}; final_display_columns = ['Ora Prevista']
                           for i, target_col_h in enumerate(target_columns_model):
                               label_h = get_station_label(target_col_h, short=True)
                               unit_match_h = re.search(r'\[(.*?)\]|\((.*?)\)', target_col_h); unit_str_h = f"({(unit_match_h.group(1) or unit_match_h.group(2) or 'm')})"
                               new_name_h = f"{label_h} H {unit_str_h}"; final_display_columns.append(new_name_h)
                               sensor_info = STATION_COORDS.get(target_col_h)
                               if sensor_info:
                                   sensor_code_sim = sensor_info.get('sensor_code')
                                   if sensor_code_sim and sensor_code_sim in RATING_CURVES:
                                       q_values_sim = calculate_discharge_vectorized(sensor_code_sim, results_df_s2s[target_col_h].values)
                                       q_col_name = f"Q {label_h} (m³/s)" # Nome breve per Q
                                       q_cols_to_add_s2s[q_col_name] = q_values_sim
                                       final_display_columns.append(q_col_name) # Aggiungi Q all'ordine

                           for q_name, q_data in q_cols_to_add_s2s.items(): results_df_s2s[q_name] = q_data
                           pred_times_s2s = [start_pred_time_s2s + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]
                           results_df_s2s.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_s2s])

                           rename_dict_s2s = {'Ora Prevista': 'Ora Prevista'}
                           for col in target_columns_model: # Rinomina solo H originale
                               label_h = get_station_label(col, short=True)
                               unit_match_h = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str_h = f"({(unit_match_h.group(1) or unit_match_h.group(2) or 'm')})"
                               rename_dict_s2s[col] = f"{label_h} H {unit_str_h}"
                           # Le colonne Q hanno già il nome corretto ("Q ...")
                           results_df_s2s_display = results_df_s2s.rename(columns=rename_dict_s2s)

                           st.dataframe(results_df_s2s_display[final_display_columns].round(3), hide_index=True) # Mostra H e Q nell'ordine giusto
                           st.markdown(get_table_download_link(results_df_s2s, f"simulazione_seq2seq_{datetime.now().strftime('%Y%m%d_%H%M')}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}.csv"), unsafe_allow_html=True)

                           st.subheader('Grafici Previsioni Simulate (Seq2Seq - Individuali H/Q)')
                           figs_sim_s2s = plot_predictions(predictions_s2s, active_config, start_pred_time_s2s) # plot_predictions gestisce H e Q
                           num_graph_cols = min(len(figs_sim_s2s), 3); sim_cols = st.columns(num_graph_cols)
                           for i, fig_sim in enumerate(figs_sim_s2s):
                              with sim_cols[i % num_graph_cols]:
                                   target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                   filename_base_s2s_ind = f"grafico_sim_s2s_{s_name_file}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                                   st.plotly_chart(fig_sim, use_container_width=True)
                                   st.markdown(get_plotly_download_link(fig_sim, filename_base_s2s_ind), unsafe_allow_html=True)

                           st.subheader('Grafico Combinato Livelli H Output (Seq2Seq)')
                           fig_combined_s2s = go.Figure()
                           x_axis_comb, x_title_comb, x_tick_format = (pred_times_s2s, "Data e Ora Previste", "%d/%m %H:%M") if start_pred_time_s2s else ((np.arange(output_steps_actual) + 1) * 0.5, "Ore Future", None)
                           for i, sensor in enumerate(target_columns_model): fig_combined_s2s.add_trace(go.Scatter(x=x_axis_comb, y=predictions_s2s[:, i], mode='lines+markers', name=get_station_label(sensor, short=True)))
                           combined_title_s2s = f'Previsioni Combinate Livello H {active_model_type}<br><span style="font-size:10px;">{ATTRIBUTION_PHRASE}</span>'
                           fig_combined_s2s.update_layout(title=combined_title_s2s, xaxis_title=x_title_comb, yaxis_title="Livello H Previsto (m)", height=500, margin=dict(l=60, r=20, t=70, b=50), hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                           if x_tick_format: fig_combined_s2s.update_xaxes(tickformat=x_tick_format)
                           fig_combined_s2s.update_yaxes(rangemode='tozero'); st.plotly_chart(fig_combined_s2s, use_container_width=True)
                           filename_base_s2s_comb = f"grafico_combinato_H_sim_s2s{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                           st.markdown(get_plotly_download_link(fig_combined_s2s, filename_base_s2s_comb), unsafe_allow_html=True)
                       else: st.error("Predizione simulazione Seq2Seq fallita.")

    else: # LSTM
         st.subheader("Metodo Input Dati Simulazione LSTM")
         sim_method_options = ['Manuale (Valori Costanti)', 'Importa da Google Sheet (Ultime Ore)', 'Orario Dettagliato (Tabella)']
         if data_ready_csv: sim_method_options.append('Usa Ultime Ore da CSV Caricato')
         sim_method = st.radio("Scegli come fornire i dati input:", sim_method_options, key="sim_method_radio_select_lstm", horizontal=True, label_visibility="collapsed")
         predictions_sim_lstm = None; start_pred_time_lstm = None
         feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns)

         if sim_method == 'Manuale (Valori Costanti)':
             st.markdown(f'Inserisci valori costanti per le **{len(feature_columns_current_model)}** feature.')
             st.caption(f"Ripetuti per **{input_steps}** steps ({input_steps*0.5:.1f} ore).")
             temp_sim_values_lstm = {}; cols_manual_lstm = st.columns(3); col_idx_manual = 0; input_valid_manual = True
             for feature in feature_columns_current_model:
                 with cols_manual_lstm[col_idx_manual % 3]:
                     label = get_station_label(feature, short=True); unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', feature); unit_str = f"({(unit_match.group(1) or unit_match.group(2) or '')})" if unit_match else ""
                     fmt = "%.1f" if 'pioggia' in feature.lower() or 'umidit' in feature.lower() else "%.2f"; step = 0.5 if 'pioggia' in feature.lower() else (1.0 if 'umidit' in feature.lower() else 0.1)
                     try: val = st.number_input(f"{label} {unit_str}".strip(), key=f"man_{feature}", format=fmt, step=step); temp_sim_values_lstm[feature] = float(val)
                     except Exception as e: st.error(f"Valore non valido {label}: {e}"); input_valid_manual = False
                 col_idx_manual += 1
             sim_data_input_manual = None
             if input_valid_manual:
                 try: ordered_values = [temp_sim_values_lstm[feature] for feature in feature_columns_current_model]; sim_data_input_manual = np.tile(ordered_values, (input_steps, 1)).astype(float)
                 except Exception as e: st.error(f"Errore creazione dati manuali: {e}"); input_valid_manual = False
             input_ready_manual = sim_data_input_manual is not None and input_valid_manual
             st.divider()
             if st.button('Esegui Simulazione LSTM (Manuale)', type="primary", disabled=(not input_ready_manual), key="sim_run_exec_lstm_manual"):
                 if input_ready_manual:
                      with st.spinner('Simulazione LSTM (Manuale)...'):
                           if isinstance(active_scalers, tuple) and len(active_scalers) == 2:
                               scaler_features_lstm, scaler_targets_lstm = active_scalers
                               predictions_sim_lstm = predict(active_model, sim_data_input_manual, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                               start_pred_time_lstm = datetime.now(italy_tz)
                           else: st.error("Errore: Scaler LSTM non validi."); predictions_sim_lstm = None
                 else: st.error("Dati input manuali non pronti.")

         elif sim_method == 'Importa da Google Sheet (Ultime Ore)':
             st.markdown(f'Importa ultimi **{input_steps}** steps ({input_steps*0.5:.1f} ore) da Google Sheet.')
             st.caption(f"Foglio Google (ID: `{GSHEET_ID}`).")
             date_col_model_name = st.session_state.date_col_name_csv
             column_mapping_gsheet_to_lstm = {
                 'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)', 'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)', 'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                 'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)', 'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME,
                 'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)', 'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
                 'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', 'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                 'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)', GSHEET_DATE_COL: date_col_model_name }
             relevant_model_cols_lstm_map = feature_columns_current_model + [date_col_model_name]
             column_mapping_gsheet_to_lstm_filtered = { gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_lstm.items() if model_col in relevant_model_cols_lstm_map }
             missing_lstm_features_in_map = list(set(feature_columns_current_model) - (set(column_mapping_gsheet_to_lstm_filtered.values()) - {date_col_model_name}))
             imputed_values_sim_lstm_gs = {}
             if missing_lstm_features_in_map:
                 st.caption(f"Feature LSTM non mappate da GSheet: `{', '.join(missing_lstm_features_in_map)}`. Imputate.")
                 for missing_f in missing_lstm_features_in_map: imputed_values_sim_lstm_gs[missing_f] = 0.0
             fetch_error_gsheet_lstm = None
             if st.button("Importa da Google Sheet (LSTM)", key="sim_run_gsheet_import_lstm"):
                 fetch_sim_gsheet_data.clear()
                 with st.spinner("Recupero dati LSTM GSheet..."):
                     required_cols_lstm_fetch = feature_columns_current_model + [date_col_model_name]
                     imported_df_lstm, import_err_lstm, last_ts_lstm = fetch_sim_gsheet_data(GSHEET_ID, input_steps, GSHEET_DATE_COL, GSHEET_DATE_FORMAT, column_mapping_gsheet_to_lstm_filtered, required_cols_lstm_fetch, imputed_values_sim_lstm_gs)
                 if import_err_lstm: st.error(f"Recupero GSheet LSTM fallito: {import_err_lstm}"); st.session_state.imported_sim_data_gs_df_lstm = None; fetch_error_gsheet_lstm = import_err_lstm
                 elif imported_df_lstm is not None:
                     try:
                         final_lstm_df = imported_df_lstm[feature_columns_current_model]
                         st.success(f"Recuperate {len(final_lstm_df)} righe LSTM da GSheet."); st.session_state.imported_sim_data_gs_df_lstm = final_lstm_df
                         st.session_state.imported_sim_start_time_gs_lstm = last_ts_lstm if last_ts_lstm else datetime.now(italy_tz)
                         fetch_error_gsheet_lstm = None; st.rerun()
                     except KeyError as e_cols_lstm: st.error(f"Errore colonne LSTM GSheet: {e_cols_lstm}"); st.session_state.imported_sim_data_gs_df_lstm = None; fetch_error_gsheet_lstm = f"Errore colonne: {e_cols_lstm}"
                 else: st.error("Recupero GSheet LSTM non riuscito."); fetch_error_gsheet_lstm = "Errore sconosciuto."

             imported_df_lstm_gs = st.session_state.get("imported_sim_data_gs_df_lstm", None)
             if imported_df_lstm_gs is not None: st.caption("Dati LSTM importati pronti."); # with st.expander("Mostra dati importati"): st.dataframe(imported_df_lstm_gs.round(3))
             sim_data_input_lstm_gs = None; sim_start_time_lstm_gs = None
             if isinstance(imported_df_lstm_gs, pd.DataFrame):
                 try:
                     sim_data_input_lstm_gs_ordered = imported_df_lstm_gs[feature_columns_current_model]; sim_data_input_lstm_gs = sim_data_input_lstm_gs_ordered.astype(float).values
                     sim_start_time_lstm_gs = st.session_state.get("imported_sim_start_time_gs_lstm", datetime.now(italy_tz))
                     if sim_data_input_lstm_gs.shape[0] != input_steps: st.warning(f"Righe GSheet ({sim_data_input_lstm_gs.shape[0]}) != richieste ({input_steps}). Uso disponibili.")
                     if np.isnan(sim_data_input_lstm_gs).any(): st.error("NaN nei dati GSheet importati."); sim_data_input_lstm_gs = None
                 except Exception as e_prep_gs: st.error(f"Errore prep dati LSTM GSheet: {e_prep_gs}"); sim_data_input_lstm_gs = None
             input_ready_lstm_gs = sim_data_input_lstm_gs is not None
             st.divider()
             if st.button('Esegui Simulazione LSTM (GSheet)', type="primary", disabled=(not input_ready_lstm_gs), key="sim_run_exec_lstm_gsheet"):
                  if input_ready_lstm_gs:
                      with st.spinner('Simulazione LSTM (GSheet)...'):
                            if isinstance(active_scalers, tuple) and len(active_scalers) == 2:
                                scaler_features_lstm, scaler_targets_lstm = active_scalers
                                predictions_sim_lstm = predict(active_model, sim_data_input_lstm_gs, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                                start_pred_time_lstm = sim_start_time_lstm_gs
                            else: st.error("Errore: Scaler LSTM non validi."); predictions_sim_lstm = None
                  else: st.error("Dati input GSheet non pronti.")

         elif sim_method == 'Orario Dettagliato (Tabella)':
             st.markdown(f'Inserisci dati per **{input_steps}** steps ({input_steps*0.5:.1f} ore) precedenti.')
             st.caption(f"Tabella con **{len(feature_columns_current_model)}** feature input.")
             editor_df_initial = pd.DataFrame(index=range(input_steps), columns=feature_columns_current_model)
             if data_ready_csv and len(df_current_csv) >= input_steps:
                 try: editor_df_initial = df_current_csv[feature_columns_current_model].iloc[-input_steps:].reset_index(drop=True).astype(float).round(2); st.caption("Precompilata con ultimi dati CSV.")
                 except Exception as e: st.caption(f"Impossibile precompilare ({e}). Inizializzata a 0."); editor_df_initial = editor_df_initial.fillna(0.0)
             else: editor_df_initial = editor_df_initial.fillna(0.0)
             edited_lstm_df = st.data_editor( editor_df_initial, key="lstm_editor_sim", num_rows="fixed", use_container_width=True,
                 column_config={ col: st.column_config.NumberColumn(label=get_station_label(col, short=True), help=f"Valore storico {col}",
                         min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower() or 'livello' in col.lower()) else None, max_value=100.0 if 'umidit' in col.lower() else None,
                         format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f", step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1), required=True ) for col in feature_columns_current_model } )
             sim_data_input_lstm_editor = None; validation_passed_editor = False
             if edited_lstm_df is not None and not edited_lstm_df.isnull().any().any() and edited_lstm_df.shape == (input_steps, len(feature_columns_current_model)):
                 try: sim_data_input_lstm_editor = edited_lstm_df[feature_columns_current_model].astype(float).values; validation_passed_editor = True
                 except Exception as e: st.error(f"Errore conversione tabella: {e}")
             else: st.warning("Completa/correggi tabella.")
             input_ready_lstm_editor = sim_data_input_lstm_editor is not None and validation_passed_editor
             st.divider()
             if st.button('Esegui Simulazione LSTM (Tabella)', type="primary", disabled=(not input_ready_lstm_editor), key="sim_run_exec_lstm_editor"):
                  if input_ready_lstm_editor:
                      with st.spinner('Simulazione LSTM (Tabella)...'):
                           if isinstance(active_scalers, tuple) and len(active_scalers) == 2:
                               scaler_features_lstm, scaler_targets_lstm = active_scalers
                               predictions_sim_lstm = predict(active_model, sim_data_input_lstm_editor, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                               start_pred_time_lstm = datetime.now(italy_tz)
                           else: st.error("Errore: Scaler LSTM non validi."); predictions_sim_lstm = None
                  else: st.error("Dati tabella non pronti.")

         elif sim_method == 'Usa Ultime Ore da CSV Caricato':
             st.markdown(f"Utilizza ultimi **{input_steps}** steps ({input_steps*0.5:.1f} ore) da CSV caricato.")
             if not data_ready_csv: st.error("Dati CSV non caricati."); st.stop()
             if len(df_current_csv) < input_steps: st.error(f"Dati CSV insuff. ({len(df_current_csv)}/{input_steps})."); st.stop()
             sim_data_input_lstm_csv = None; sim_start_time_lstm_csv = None
             try:
                 latest_csv_data_df = df_current_csv.iloc[-input_steps:]
                 missing_cols_csv = [col for col in feature_columns_current_model if col not in latest_csv_data_df.columns]
                 if missing_cols_csv: st.error(f"Colonne LSTM mancanti nel CSV: `{', '.join(missing_cols_csv)}`")
                 else:
                     sim_data_input_lstm_csv_ordered = latest_csv_data_df[feature_columns_current_model]; sim_data_input_lstm_csv = sim_data_input_lstm_csv_ordered.astype(float).values
                     last_csv_timestamp = df_current_csv[date_col_name_csv].iloc[-1]
                     if pd.notna(last_csv_timestamp) and isinstance(last_csv_timestamp, pd.Timestamp): sim_start_time_lstm_csv = last_csv_timestamp.tz_localize(italy_tz) if last_csv_timestamp.tz is None else last_csv_timestamp.tz_convert(italy_tz)
                     else: sim_start_time_lstm_csv = datetime.now(italy_tz)
                     st.caption("Dati CSV pronti."); # with st.expander("Mostra dati CSV usati"): st.dataframe(latest_csv_data_df[[date_col_name_csv] + feature_columns_current_model].round(3))
                     if np.isnan(sim_data_input_lstm_csv).any(): st.error("NaN negli ultimi dati CSV."); sim_data_input_lstm_csv = None
             except Exception as e_prep_csv: st.error(f"Errore preparazione CSV: {e_prep_csv}"); sim_data_input_lstm_csv = None
             input_ready_lstm_csv = sim_data_input_lstm_csv is not None
             st.divider()
             if st.button('Esegui Simulazione LSTM (CSV)', type="primary", disabled=(not input_ready_lstm_csv), key="sim_run_exec_lstm_csv"):
                 if input_ready_lstm_csv:
                      with st.spinner('Simulazione LSTM (CSV)...'):
                           if isinstance(active_scalers, tuple) and len(active_scalers) == 2:
                               scaler_features_lstm, scaler_targets_lstm = active_scalers
                               predictions_sim_lstm = predict(active_model, sim_data_input_lstm_csv, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                               start_pred_time_lstm = sim_start_time_lstm_csv
                           else: st.error("Errore: Scaler LSTM non validi."); predictions_sim_lstm = None
                 else: st.error("Dati CSV non pronti.")

         # --- Visualizzazione Risultati LSTM ---
         if predictions_sim_lstm is not None:
             output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
             st.subheader(f'Risultato Simulazione LSTM ({sim_method}): {total_hours_output_actual:.1f} ore')
             if start_pred_time_lstm: st.caption(f"A partire da: {start_pred_time_lstm.strftime('%d/%m/%Y %H:%M:%S %Z')}")
             else: st.caption("Previsione calcolata (timestamp non disponibile).")

             results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model)
             q_cols_to_add_lstm = {}; final_display_columns_lstm = ['Ora/Passo Previsto']
             for i, target_col_h in enumerate(target_columns_model):
                 label_h = get_station_label(target_col_h, short=True)
                 unit_match_h = re.search(r'\[(.*?)\]|\((.*?)\)', target_col_h); unit_str_h = f"({(unit_match_h.group(1) or unit_match_h.group(2) or 'm')})"
                 new_name_h = f"{label_h} H {unit_str_h}"; final_display_columns_lstm.append(new_name_h)
                 sensor_info = STATION_COORDS.get(target_col_h)
                 if sensor_info:
                     sensor_code_sim = sensor_info.get('sensor_code')
                     if sensor_code_sim and sensor_code_sim in RATING_CURVES:
                         q_values_sim = calculate_discharge_vectorized(sensor_code_sim, results_df_lstm[target_col_h].values)
                         q_col_name = f"Q {label_h} (m³/s)"; q_cols_to_add_lstm[q_col_name] = q_values_sim; final_display_columns_lstm.append(q_col_name)

             for q_name, q_data in q_cols_to_add_lstm.items(): results_df_lstm[q_name] = q_data
             time_col_name = 'Ora Prevista' if start_pred_time_lstm else 'Passo Futuro'
             if start_pred_time_lstm: pred_times_lstm = [start_pred_time_lstm + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]; results_df_lstm.insert(0, time_col_name, [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
             else: pred_steps_lstm = [f"Step {i+1} (+{(i+1)*0.5}h)" for i in range(output_steps_actual)]; results_df_lstm.insert(0, time_col_name, pred_steps_lstm)

             rename_dict_lstm = {time_col_name: 'Ora/Passo Previsto'}
             for col in target_columns_model: # Rinomina H
                 label_h = get_station_label(col, short=True); unit_match_h = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit_str_h = f"({(unit_match_h.group(1) or unit_match_h.group(2) or 'm')})"
                 rename_dict_lstm[col] = f"{label_h} H {unit_str_h}"
             results_df_lstm_display = results_df_lstm.rename(columns=rename_dict_lstm)

             st.dataframe(results_df_lstm_display[final_display_columns_lstm].round(3), hide_index=True)
             st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_{sim_method.split()[0].lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}.csv"), unsafe_allow_html=True)

             st.subheader(f'Grafici Previsioni Simulate (LSTM {sim_method} - Individuali H/Q)')
             figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, start_pred_time_lstm) # Gestisce H/Q
             num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
             for i, fig_sim in enumerate(figs_sim_lstm):
                 with sim_cols_lstm[i % num_graph_cols_lstm]:
                      target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                      filename_base_lstm_ind = f"grafico_sim_lstm_{sim_method.split()[0].lower()}_{s_name_file}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                      st.plotly_chart(fig_sim, use_container_width=True)
                      st.markdown(get_plotly_download_link(fig_sim, filename_base_lstm_ind), unsafe_allow_html=True)

             st.subheader(f'Grafico Combinato Livelli H Output (LSTM {sim_method})')
             fig_combined_lstm = go.Figure()
             x_axis_comb_lstm, x_title_comb_lstm, x_tick_format_lstm = (pred_times_lstm, "Data e Ora Previste", "%d/%m %H:%M") if start_pred_time_lstm and 'pred_times_lstm' in locals() else ((np.arange(output_steps_actual) + 1) * 0.5, "Ore Future", None)
             for i, sensor in enumerate(target_columns_model): fig_combined_lstm.add_trace(go.Scatter(x=x_axis_comb_lstm, y=predictions_sim_lstm[:, i], mode='lines+markers', name=get_station_label(sensor, short=True)))
             combined_title_lstm = f'Previsioni Combinate Livello H {active_model_type} ({sim_method})<br><span style="font-size:10px;">{ATTRIBUTION_PHRASE}</span>'
             fig_combined_lstm.update_layout(title=combined_title_lstm, xaxis_title=x_title_comb_lstm, yaxis_title="Livello H Previsto (m)", height=500, margin=dict(l=60, r=20, t=70, b=50), hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
             if x_tick_format_lstm: fig_combined_lstm.update_xaxes(tickformat=x_tick_format_lstm)
             fig_combined_lstm.update_yaxes(rangemode='tozero'); st.plotly_chart(fig_combined_lstm, use_container_width=True)
             filename_base_lstm_comb = f"grafico_combinato_H_sim_lstm_{sim_method.split()[0].lower()}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
             st.markdown(get_plotly_download_link(fig_combined_lstm, filename_base_lstm_comb), unsafe_allow_html=True)

         elif any(st.session_state.get(f"sim_run_exec_lstm_{m.split()[0].lower()}", False) for m in sim_method_options if m != 'Importa da Google Sheet (Ultime Ore)'):
             st.error(f"Predizione simulazione LSTM ({sim_method}) fallita.")


# --- PAGINA ANALISI DATI STORICI ---
elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici (da file CSV)')
    if not data_ready_csv: st.warning("Carica dati CSV dalla sidebar.")
    else:
        st.info(f"Dataset CSV caricato: **{len(df_current_csv)}** righe.")
        if date_col_name_csv in df_current_csv:
             try:
                 if not pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]): df_current_csv[date_col_name_csv] = pd.to_datetime(df_current_csv[date_col_name_csv], errors='coerce').dropna(subset=[date_col_name_csv])
                 if not df_current_csv.empty: min_date_str = df_current_csv[date_col_name_csv].min().strftime('%d/%m/%Y %H:%M'); max_date_str = df_current_csv[date_col_name_csv].max().strftime('%d/%m/%Y %H:%M'); st.caption(f"Periodo: **{min_date_str}** - **{max_date_str}**")
                 else: st.warning("Nessuna data valida nel CSV.")
             except Exception as e: st.warning(f"Impossibile determinare periodo CSV: {e}")
        else: st.warning(f"Colonna data '{date_col_name_csv}' non trovata.")

        st.subheader("Esplorazione Dati"); st.dataframe(df_current_csv.head())
        st.markdown(get_table_download_link(df_current_csv, f"dati_storici_completi_{datetime.now().strftime('%Y%m%d')}.csv", link_text="Scarica Dati Completi (CSV)"), unsafe_allow_html=True)
        st.divider()
        st.subheader("Statistiche Descrittive")
        numeric_cols_analysis = df_current_csv.select_dtypes(include=np.number).columns
        if not numeric_cols_analysis.empty: st.dataframe(df_current_csv[numeric_cols_analysis].describe().round(2))
        else: st.info("Nessuna colonna numerica per statistiche.")
        st.divider()
        st.subheader("Visualizzazione Temporale")
        if date_col_name_csv in df_current_csv and not numeric_cols_analysis.empty and pd.api.types.is_datetime64_any_dtype(df_current_csv[date_col_name_csv]):
            cols_to_plot = st.multiselect("Seleziona colonne:", options=numeric_cols_analysis.tolist(), default=numeric_cols_analysis[:min(len(numeric_cols_analysis), 5)].tolist(), key="analysis_plot_select")
            if cols_to_plot:
                fig_analysis = go.Figure()
                for col in cols_to_plot: fig_analysis.add_trace(go.Scatter(x=df_current_csv[date_col_name_csv], y=df_current_csv[col], mode='lines', name=get_station_label(col, short=True)))
                fig_analysis.update_layout(title="Andamento Temporale", xaxis_title="Data", yaxis_title="Valore", height=500, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_analysis, use_container_width=True)
                analysis_filename_base = f"analisi_temporale_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_analysis, analysis_filename_base), unsafe_allow_html=True)
            else: st.info("Seleziona colonne per visualizzare.")
        else: st.info("Colonna data o numeriche mancanti per grafico temporale.")
        st.divider()
        st.subheader("Matrice di Correlazione")
        if len(numeric_cols_analysis) > 1:
            corr_matrix = df_current_csv[numeric_cols_analysis].corr()
            fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='viridis', zmin=-1, zmax=1, colorbar=dict(title='Corr')))
            fig_corr.update_layout(title='Correlazione Variabili Numeriche', xaxis_tickangle=-45, height=600, template="plotly_white"); st.plotly_chart(fig_corr, use_container_width=True)
        else: st.info("Servono >= 2 colonne numeriche per correlazione.")

# --- PAGINA ALLENAMENTO MODELLO ---
elif page == 'Allenamento Modello':
    st.header('Allenamento Nuovo Modello')
    if not data_ready_csv: st.warning("Carica dati CSV dalla sidebar."); st.stop()

    st.success(f"Dati CSV disponibili: {len(df_current_csv)} righe.")
    st.subheader('Configurazione Addestramento')
    train_model_type = st.radio("Tipo Modello:", ["LSTM Standard", "Seq2Seq (Encoder-Decoder)"], key="train_select_type", horizontal=True)
    default_save_name = f"modello_{train_model_type.split()[0].lower()}_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
    save_name_input = st.text_input("Nome base salvataggio:", default_save_name, key="train_save_filename")
    save_name = re.sub(r'[^\w-]', '_', save_name_input).strip('_') or "modello_default"
    if save_name != save_name_input: st.caption(f"Nome file valido: `{save_name}`")
    os.makedirs(MODELS_DIR, exist_ok=True)

    if train_model_type == "LSTM Standard":
        st.markdown("**1. Seleziona Feature e Target (LSTM)**")
        all_features_lstm = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
        default_features_lstm = [f for f in st.session_state.feature_columns if f in all_features_lstm]
        selected_features_train_lstm = st.multiselect("Feature Input LSTM:", options=all_features_lstm, default=default_features_lstm, key="train_lstm_feat")
        level_options_lstm = [f for f in all_features_lstm if 'livello' in f.lower() or '[m]' in f.lower()]
        default_targets_lstm = level_options_lstm[:1]
        selected_targets_train_lstm = st.multiselect("Target Output LSTM (Livelli):", options=level_options_lstm, default=default_targets_lstm, key="train_lstm_target")

        st.markdown("**2. Parametri Modello e Training (LSTM)**")
        with st.expander("Impostazioni Allenamento LSTM", expanded=True):
            c1_lstm, c2_lstm, c3_lstm = st.columns(3)
            with c1_lstm:
                iw_t_lstm_steps = st.number_input("Input Window (steps)", 2, 500, 48, 2, key="t_lstm_in_steps")
                ow_t_lstm_steps = st.number_input("Output Window (steps)", 1, 500, 24, 1, key="t_lstm_out_steps")
                vs_t_lstm = st.slider("Split Validazione (%)", 0, 50, 20, 1, key="t_lstm_vs")
            with c2_lstm:
                hs_t_lstm = st.number_input("Hidden Size", 8, 1024, 128, 8, key="t_lstm_hs")
                nl_t_lstm = st.number_input("Numero Layers", 1, 8, 2, 1, key="t_lstm_nl")
                dr_t_lstm = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_lstm_dr")
            with c3_lstm:
                lr_t_lstm = st.number_input("Learning Rate", 1e-6, 1.0, 0.001, format="%.5f", step=1e-4, key="t_lstm_lr")
                bs_t_lstm = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_lstm_bs")
                ep_t_lstm = st.number_input("Numero Epoche", 1, 1000, 50, 5, key="t_lstm_ep")
            col_dev_lstm, col_save_lstm = st.columns(2)
            with col_dev_lstm: device_option_lstm = st.radio("Device:", ['Auto', 'CPU'], 0, key='train_device_lstm', horizontal=True)
            with col_save_lstm: save_choice_lstm = st.radio("Salva:", ['Migliore', 'Finale'], 0, key='train_save_lstm', horizontal=True)

        st.divider()
        ready_to_train_lstm = bool(save_name and selected_features_train_lstm and selected_targets_train_lstm)
        if st.button("Avvia Addestramento LSTM", type="primary", disabled=not ready_to_train_lstm, key="train_run_lstm"):
             st.info(f"Avvio training LSTM '{save_name}'...")
             with st.spinner("Preparazione dati LSTM..."):
                  # Passa input/output in ORE a prepare_training_data
                  input_hours_lstm = iw_t_lstm_steps / 2; output_hours_lstm = ow_t_lstm_steps / 2
                  X_tr, y_tr, X_v, y_v, sc_f, sc_t = prepare_training_data(df_current_csv.copy(), selected_features_train_lstm, selected_targets_train_lstm, int(input_hours_lstm), int(output_hours_lstm), vs_t_lstm)
             if X_tr is None or sc_f is None or sc_t is None: st.error("Prep dati LSTM fallita."); st.stop()

             # Passa output in STEPS a train_model
             trained_model_lstm, _, _ = train_model( X_tr, y_tr, X_v, y_v, len(selected_features_train_lstm), len(selected_targets_train_lstm), ow_t_lstm_steps,
                   hs_t_lstm, nl_t_lstm, ep_t_lstm, bs_t_lstm, lr_t_lstm, dr_t_lstm, save_strategy=save_choice_lstm.lower(), preferred_device=device_option_lstm.lower())

             if trained_model_lstm:
                  st.success("Training LSTM completato!")
                  try:
                      base_path = os.path.join(MODELS_DIR, save_name)
                      m_path = f"{base_path}.pth"; torch.save(trained_model_lstm.state_dict(), m_path)
                      sf_path = f"{base_path}_features.joblib"; joblib.dump(sc_f, sf_path)
                      st_path = f"{base_path}_targets.joblib"; joblib.dump(sc_t, st_path)
                      c_path = f"{base_path}.json"
                      config_save_lstm = { "model_type": "LSTM", "input_window": iw_t_lstm_steps, "output_window": ow_t_lstm_steps, "hidden_size": hs_t_lstm, "num_layers": nl_t_lstm, "dropout": dr_t_lstm,
                          "feature_columns": selected_features_train_lstm, "target_columns": selected_targets_train_lstm, "training_date": datetime.now(italy_tz).isoformat(), "val_split_percent": vs_t_lstm,
                          "learning_rate": lr_t_lstm, "batch_size": bs_t_lstm, "epochs": ep_t_lstm, "display_name": save_name, }
                      with open(c_path, 'w', encoding='utf-8') as f: json.dump(config_save_lstm, f, indent=4)
                      st.success(f"Modello LSTM '{save_name}' salvato."); st.subheader("Download File LSTM")
                      col_dl_lstm1, col_dl_lstm2 = st.columns(2)
                      with col_dl_lstm1: st.markdown(get_download_link_for_file(m_path), unsafe_allow_html=True); st.markdown(get_download_link_for_file(sf_path, "Scaler Features"), unsafe_allow_html=True)
                      with col_dl_lstm2: st.markdown(get_download_link_for_file(c_path), unsafe_allow_html=True); st.markdown(get_download_link_for_file(st_path, "Scaler Target"), unsafe_allow_html=True)
                      find_available_models.clear()
                  except Exception as e_save: st.error(f"Errore salvataggio LSTM: {e_save}"); st.error(traceback.format_exc())
             else: st.error("Training LSTM fallito.")

    elif train_model_type == "Seq2Seq (Encoder-Decoder)":
         st.markdown("**1. Seleziona Feature (Seq2Seq)**")
         features_csv_s2s = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
         default_past_s2s = [f for f in st.session_state.feature_columns if f in features_csv_s2s]
         selected_past_s2s = st.multiselect("Feature Storiche (Encoder):", options=features_csv_s2s, default=default_past_s2s, key="train_s2s_past_feat")
         options_forecast_s2s = selected_past_s2s
         default_forecast_s2s = [f for f in options_forecast_s2s if 'pioggia' in f.lower() or 'cumulata' in f.lower() or f == HUMIDITY_COL_NAME]
         selected_forecast_s2s = st.multiselect("Feature Forecast (Decoder):", options=options_forecast_s2s, default=default_forecast_s2s, key="train_s2s_forecast_feat")
         level_options_s2s = [f for f in selected_past_s2s if 'livello' in f.lower() or '[m]' in f.lower()]
         default_targets_s2s = level_options_s2s[:1]
         selected_targets_s2s = st.multiselect("Target Output (Livelli):", options=level_options_s2s, default=default_targets_s2s, key="train_s2s_target_feat")

         st.markdown("**2. Parametri Finestre e Training (Seq2Seq)**")
         with st.expander("Impostazioni Allenamento Seq2Seq", expanded=True):
             c1_s2s, c2_s2s, c3_s2s = st.columns(3)
             with c1_s2s:
                 iw_steps_s2s = st.number_input("Input Storico (steps)", 2, 500, 48, 2, key="t_s2s_in")
                 fw_steps_s2s = st.number_input("Input Forecast (steps)", 1, 500, 24, 1, key="t_s2s_fore")
                 ow_steps_s2s = st.number_input("Output Previsione (steps)", 1, 500, 24, 1, key="t_s2s_out")
                 if ow_steps_s2s > fw_steps_s2s: st.caption("Nota: Output > Forecast Steps.")
                 vs_t_s2s = st.slider("Split Validazione (%)", 0, 50, 20, 1, key="t_s2s_vs")
             with c2_s2s:
                 hs_t_s2s = st.number_input("Hidden Size", 8, 1024, 128, 8, key="t_s2s_hs")
                 nl_t_s2s = st.number_input("Numero Layers", 1, 8, 2, 1, key="t_s2s_nl")
                 dr_t_s2s = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_s2s_dr")
             with c3_s2s:
                 lr_t_s2s = st.number_input("Learning Rate", 1e-6, 1.0, 0.001, format="%.5f", step=1e-4, key="t_s2s_lr")
                 bs_t_s2s = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_s2s_bs")
                 ep_t_s2s = st.number_input("Numero Epoche", 1, 1000, 50, 5, key="t_s2s_ep")
             col_dev_s2s, col_save_s2s = st.columns(2)
             with col_dev_s2s: device_option_s2s = st.radio("Device:", ['Auto', 'CPU'], 0, key='train_device_s2s', horizontal=True)
             with col_save_s2s: save_choice_s2s = st.radio("Salva:", ['Migliore', 'Finale'], 0, key='train_save_s2s', horizontal=True)

         st.divider()
         ready_to_train_s2s = bool(save_name and selected_past_s2s and selected_forecast_s2s and selected_targets_s2s)
         if st.button("Avvia Addestramento Seq2Seq", type="primary", disabled=not ready_to_train_s2s, key="train_run_s2s"):
             st.info(f"Avvio training Seq2Seq '{save_name}'...")
             with st.spinner("Preparazione dati Seq2Seq..."):
                  data_tuple = prepare_training_data_seq2seq(df_current_csv.copy(), selected_past_s2s, selected_forecast_s2s, selected_targets_s2s, iw_steps_s2s, fw_steps_s2s, ow_steps_s2s, vs_t_s2s)
             if data_tuple is None or len(data_tuple) != 9 or data_tuple[0] is None: st.error("Prep dati Seq2Seq fallita."); st.stop()
             (X_enc_tr, X_dec_tr, y_tar_tr, X_enc_v, X_dec_v, y_tar_v, sc_past, sc_fore, sc_tar) = data_tuple
             if not all([sc_past, sc_fore, sc_tar]): st.error("Errore: Scaler Seq2Seq non generati."); st.stop()

             try: enc = EncoderLSTM(len(selected_past_s2s), hs_t_s2s, nl_t_s2s, dr_t_s2s); dec = DecoderLSTM(len(selected_forecast_s2s), hs_t_s2s, len(selected_targets_s2s), nl_t_s2s, dr_t_s2s)
             except Exception as e: st.error(f"Errore init modello Seq2Seq: {e}"); st.stop()

             trained_model_s2s, _, _ = train_model_seq2seq( X_enc_tr, X_dec_tr, y_tar_tr, X_enc_v, X_dec_v, y_tar_v, enc, dec, ow_steps_s2s,
                  ep_t_s2s, bs_t_s2s, lr_t_s2s, save_strategy=save_choice_s2s.lower(), preferred_device=device_option_s2s.lower(), teacher_forcing_ratio_schedule=[0.6, 0.1] )

             if trained_model_s2s:
                  st.success("Training Seq2Seq completato!")
                  try:
                      base_path = os.path.join(MODELS_DIR, save_name)
                      m_path = f"{base_path}.pth"; torch.save(trained_model_s2s.state_dict(), m_path)
                      sp_path = f"{base_path}_past_features.joblib"; joblib.dump(sc_past, sp_path)
                      sf_path = f"{base_path}_forecast_features.joblib"; joblib.dump(sc_fore, sf_path)
                      st_path = f"{base_path}_targets.joblib"; joblib.dump(sc_tar, st_path)
                      c_path = f"{base_path}.json"
                      config_save_s2s = { "model_type": "Seq2Seq", "input_window_steps": iw_steps_s2s, "forecast_window_steps": fw_steps_s2s, "output_window_steps": ow_steps_s2s,
                          "hidden_size": hs_t_s2s, "num_layers": nl_t_s2s, "dropout": dr_t_s2s, "all_past_feature_columns": selected_past_s2s, "forecast_input_columns": selected_forecast_s2s, "target_columns": selected_targets_s2s,
                          "training_date": datetime.now(italy_tz).isoformat(), "val_split_percent": vs_t_s2s, "learning_rate": lr_t_s2s, "batch_size": bs_t_s2s, "epochs": ep_t_s2s, "display_name": save_name, }
                      with open(c_path, 'w', encoding='utf-8') as f: json.dump(config_save_s2s, f, indent=4)
                      st.success(f"Modello Seq2Seq '{save_name}' salvato."); st.subheader("Download File Seq2Seq")
                      col_dl_s2s_1, col_dl_s2s_2 = st.columns(2)
                      with col_dl_s2s_1: st.markdown(get_download_link_for_file(m_path), unsafe_allow_html=True); st.markdown(get_download_link_for_file(sp_path,"Scaler Passato"), unsafe_allow_html=True); st.markdown(get_download_link_for_file(st_path,"Scaler Target"), unsafe_allow_html=True)
                      with col_dl_s2s_2: st.markdown(get_download_link_for_file(c_path), unsafe_allow_html=True); st.markdown(get_download_link_for_file(sf_path,"Scaler Forecast"), unsafe_allow_html=True)
                      find_available_models.clear()
                  except Exception as e_save_s2s: st.error(f"Errore salvataggio Seq2Seq: {e_save_s2s}"); st.error(traceback.format_exc())
             else: st.error("Training Seq2Seq fallito.")

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption(f'Modello Predittivo Idrologico © {datetime.now().year}')
