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
from sklearn.model_selection import TimeSeriesSplit # NEW: For Time-Series Cross-Validation

# --- Configurazione Pagina Streamlit ---
st.set_page_config(page_title="Modello Predittivo Idrologico", layout="wide")

# --- Costanti Globali ---
MODELS_DIR = "models"
DEFAULT_DATA_PATH = "dati_idro.csv"
GSHEET_ID = "1pQI6cKrrT-gcVAfl-9ZhUx5b3J-edZRRj6nzDcCBRcA" # ID Foglio per Dashboard e Simulazione Base
GSHEET_DATE_COL = 'Data_Ora'
GSHEET_DATE_FORMAT = '%d/%m/%Y %H:%M'

# --- INIZIO MODIFICA: Costanti per i due fogli ---
GSHEET_MAIN_NAME = 'Dati Idrometeo CAMI' # Nome del foglio principale (o sh.sheet1)
GSHEET_WEATHERLINK_NAME = 'Dati Stazioni Weatherlink' # Nome del nuovo foglio

MAIN_SHEET_RELEVANT_COLS = [
    GSHEET_DATE_COL,
    'Arcevia - Pioggia Ora (mm)', 'Barbara - Pioggia Ora (mm)', 'Corinaldo - Pioggia Ora (mm)',
    'Misa - Pioggia Ora (mm)', 'Umidita\' Sensore 3452 (Montemurello)',
    'Serra dei Conti - Livello Misa (mt)', 'Pianello di Ostra - Livello Misa (m)',
    'Nevola - Livello Nevola (mt)', 'Misa - Livello Misa (mt)',
    'Ponte Garibaldi - Livello Misa 2 (mt)'
]

WEATHERLINK_RELEVANT_COLS = [
    GSHEET_DATE_COL,
    'Montignano - Intensità Pioggia (mm/hr)', 'Montignano - Pioggia Giornaliera (mm)', 'Montignano - Raffica Vento 10 min (km/h)', 'Montignano - Velocità Vento (km/h)',
    'Sant\'Angelo - Intensità Pioggia (mm/hr)', 'Sant\'Angelo - Pioggia Giornaliera (mm)', 'Sant\'Angelo - Raffica Vento 10 min (km/h)', 'Sant\'Angelo - Velocità Vento (km/h)',
    'Scapezzano - Intensità Pioggia (mm/hr)', 'Scapezzano - Pioggia Giornaliera (mm)', 'Scapezzano - Raffica Vento 10 min (km/h)', 'Scapezzano - Velocità Vento (km/h)'
]

# Combina tutte le colonne rilevanti per la dashboard
GSHEET_RELEVANT_COLS = sorted(list(set(MAIN_SHEET_RELEVANT_COLS + WEATHERLINK_RELEVANT_COLS)))
# --- FINE MODIFICA ---

DASHBOARD_REFRESH_INTERVAL_SECONDS = 300
DASHBOARD_HISTORY_ROWS = 48 # 24 ore (48 step da 30 min) - Usato per vista default

DEFAULT_THRESHOLDS = { # Soglie USATE NELLA DASHBOARD
    'Arcevia - Pioggia Ora (mm)': 2.0, 'Barbara - Pioggia Ora (mm)': 2.0, 'Corinaldo - Pioggia Ora (mm)': 2.0,
    'Misa - Pioggia Ora (mm)': 2.0, 'Umidita\' Sensore 3452 (Montemurello)': 95.0,
    'Serra dei Conti - Livello Misa (mt)': 1.7, 'Pianello di Ostra - Livello Misa (m)': 2.0,
    'Nevola - Livello Nevola (mt)': 2.5, 'Misa - Livello Misa (mt)': 2.0,
    'Ponte Garibaldi - Livello Misa 2 (mt)': 2.2,
    # --- INIZIO MODIFICA: Soglie per nuove stazioni Weatherlink (valori di esempio) ---
    'Montignano - Intensità Pioggia (mm/hr)': 25.0,
    'Montignano - Pioggia Giornaliera (mm)': 40.0,
    'Montignano - Raffica Vento 10 min (km/h)': 60.0,
    'Sant\'Angelo - Intensità Pioggia (mm/hr)': 25.0,
    'Sant\'Angelo - Pioggia Giornaliera (mm)': 40.0,
    'Sant\'Angelo - Raffica Vento 10 min (km/h)': 60.0,
    'Scapezzano - Intensità Pioggia (mm/hr)': 25.0,
    'Scapezzano - Pioggia Giornaliera (mm)': 40.0,
    'Scapezzano - Raffica Vento 10 min (km/h)': 60.0,
    # Le velocità del vento e le altre colonne possono non avere una soglia di allerta definita
    # --- FINE MODIFICA ---
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
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)': {'attenzione': 1.2, 'allerta': 1.7},
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)': {'attenzione': 1.5, 'allerta': 2.0},
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)': {'attenzione': 2.0, 'allerta': 2.5},
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)': {'attenzione': 1.5, 'allerta': 2.0},
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'attenzione': 1.5, 'allerta': 2.2}
}

# --- NUOVE COSTANTI PER LA FRASE DI ATTRIBUZIONE ---
ATTRIBUTION_PHRASE = "Previsione progettata ed elaborata da Alberto Bussaglia del Comune di Senigallia. La previsione è di tipo probabilistico e non rappresenta una certezza.<br> Per maggiori informazioni, o per ottenere il consenso all'utilizzo/pubblicazione, contattare l'autore."
ATTRIBUTION_PHRASE_FILENAME_SUFFIX = "_da_Alberto_Bussaglia_Area_PC"

italy_tz = pytz.timezone('Europe/Rome')
HUMIDITY_COL_NAME = "Umidita' Sensore 3452 (Montemurello)"

# --- NUOVA COSTANTE: SCALE DI DEFLUSSO ---
RATING_CURVES = {
    '1112': [
        {'min_H': 0.73, 'max_H': 1.3,  'a': 32.6,   'b': 0.72,  'c': 2.45, 'd': 0.0},
        {'min_H': 1.31, 'max_H': 3.93, 'a': 30.19,  'b': 0.82,  'c': 1.78, 'd': 0.1},
        {'min_H': 3.94, 'max_H': 6.13, 'a': 15.51,  'b': 0.85,  'c': 2.4,  'd': 0.11}
    ],
    '1008': [
        {'min_H': 0.45, 'max_H': 0.73, 'a': 24.52,  'b': 0.45,  'c': 1.86, 'd': 0.0},
        {'min_H': 0.74, 'max_H': 2.9,  'a': 28.77,  'b': 0.55,  'c': 1.48, 'd': 0.0}
    ],
    '1283': [
        {'min_H': 0.85, 'max_H': 1.2,  'a': 0.628,  'b': 0.849, 'c': 0.957, 'd': 0.0},
        {'min_H': 1.21, 'max_H': 1.46, 'a': 22.094, 'b': 1.2,   'c': 1.295, 'd': 0.231},
        {'min_H': 1.47, 'max_H': 2.84, 'a': 48.606, 'b': 1.46,  'c': 1.18,  'd': 4.091},
        {'min_H': 2.85, 'max_H': 5.0,  'a': 94.37,  'b': 2.84,  'c': 1.223, 'd': 75.173}
    ]
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

    # Livelli GSheet Style (con sensor_code)
    'Serra dei Conti - Livello Misa (mt)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti', 'sensor_code': '1008'},
    'Nevola - Livello Nevola (mt)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo', 'sensor_code': '1283'},
    'Misa - Livello Misa (mt)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle', 'sensor_code': '1112'},
    # Livelli GSheet Style (senza scala di deflusso specificata)
    'Pianello di Ostra - Livello Misa (m)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'},
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'},

    # Livelli CSV/Internal Style (con sensor_code) - Assicurarsi che corrispondano a GSheet se usati
     'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti', 'sensor_code': '1008'},
     'Livello Idrometrico Sensore 1112 [m] (Bettolelle)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle', 'sensor_code': '1112'},
     'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo', 'sensor_code': '1283'},
    # Livelli CSV/Internal Style (senza scala di deflusso specificata)
     'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'},
     'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'},

    # --- INIZIO MODIFICA: Aggiunta coordinate e tipi per le nuove stazioni Weatherlink ---
    # NOTA: Le coordinate sono indicative e andrebbero verificate/aggiornate.
    'Montignano - Intensità Pioggia (mm/hr)': {'lat': 43.68, 'lon': 13.30, 'name': 'Montignano (Int. Pioggia)', 'type': 'Pioggia', 'location_id': 'Montignano'},
    'Montignano - Pioggia Giornaliera (mm)': {'lat': 43.68, 'lon': 13.30, 'name': 'Montignano (Pioggia G.)', 'type': 'Pioggia', 'location_id': 'Montignano'},
    'Montignano - Raffica Vento 10 min (km/h)': {'lat': 43.68, 'lon': 13.30, 'name': 'Montignano (Raffica Vento)', 'type': 'Vento', 'location_id': 'Montignano'},
    'Montignano - Velocità Vento (km/h)': {'lat': 43.68, 'lon': 13.30, 'name': 'Montignano (Vel. Vento)', 'type': 'Vento', 'location_id': 'Montignano'},
    'Sant\'Angelo - Intensità Pioggia (mm/hr)': {'lat': 43.70, 'lon': 13.25, 'name': 'Sant\'Angelo (Int. Pioggia)', 'type': 'Pioggia', 'location_id': 'Sant\'Angelo'},
    'Sant\'Angelo - Pioggia Giornaliera (mm)': {'lat': 43.70, 'lon': 13.25, 'name': 'Sant\'Angelo (Pioggia G.)', 'type': 'Pioggia', 'location_id': 'Sant\'Angelo'},
    'Sant\'Angelo - Raffica Vento 10 min (km/h)': {'lat': 43.70, 'lon': 13.25, 'name': 'Sant\'Angelo (Raffica Vento)', 'type': 'Vento', 'location_id': 'Sant\'Angelo'},
    'Sant\'Angelo - Velocità Vento (km/h)': {'lat': 43.70, 'lon': 13.25, 'name': 'Sant\'Angelo (Vel. Vento)', 'type': 'Vento', 'location_id': 'Sant\'Angelo'},
    'Scapezzano - Intensità Pioggia (mm/hr)': {'lat': 43.72, 'lon': 13.30, 'name': 'Scapezzano (Int. Pioggia)', 'type': 'Pioggia', 'location_id': 'Scapezzano'},
    'Scapezzano - Pioggia Giornaliera (mm)': {'lat': 43.72, 'lon': 13.30, 'name': 'Scapezzano (Pioggia G.)', 'type': 'Pioggia', 'location_id': 'Scapezzano'},
    'Scapezzano - Raffica Vento 10 min (km/h)': {'lat': 43.72, 'lon': 13.30, 'name': 'Scapezzano (Raffica Vento)', 'type': 'Vento', 'location_id': 'Scapezzano'},
    'Scapezzano - Velocità Vento (km/h)': {'lat': 43.72, 'lon': 13.30, 'name': 'Scapezzano (Vel. Vento)', 'type': 'Vento', 'location_id': 'Scapezzano'},
    # --- FINE MODIFICA ---
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
        # print(f"Debug: Nessuna scala di deflusso per sensore {sensor_code}")
        return None
    if H is None or not isinstance(H, (int, float, np.number)) or np.isnan(H):
        # print(f"Debug: Livello H non valido ({H}) per sensore {sensor_code}")
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
                    return float(d) if H >= min_H else np.nan # O None
                Q = a * math.pow(base, c) + d
                return float(Q)
        return None

    except (ValueError, TypeError, OverflowError) as e:
        return np.nan # Restituisce NaN in caso di errori matematici

calculate_discharge_vectorized = np.vectorize(calculate_discharge, otypes=[float])
# --- FINE NUOVA FUNZIONE ---


# --- Definizioni Classi Modello (Dataset, LSTM, Encoder, Decoder, Seq2Seq) ---
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
        self.output_window = output_window # Numero di *passi* da 30 min
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * output_window) # Prevede tutti i passi insieme

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Usa output dell'ultimo step
        out = self.fc(out)
        out = out.view(out.size(0), self.output_window, self.output_size) # Reshape
        return out

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x) # Ignora outputs, prendi solo stati finali
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
        self.fc = nn.Linear(hidden_size, output_size) # Prevede 1 step alla volta

    def forward(self, x_forecast_step, hidden, cell):
        output, (hidden, cell) = self.lstm(x_forecast_step, (hidden, cell))
        prediction = self.fc(output.squeeze(1)) # Shape: (batch, output_size)
        return prediction, hidden, cell

class Seq2SeqHydro(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps
        self.device = device

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0): # Default TF a 0 per inferenza
        batch_size = x_past.shape[0]
        forecast_window = x_future_forecast.shape[1]
        target_output_size = self.decoder.output_size

        if forecast_window < self.output_window:
             missing_steps = self.output_window - forecast_window
             last_forecast_step = x_future_forecast[:, -1:, :]
             padding = last_forecast_step.repeat(1, missing_steps, 1)
             x_future_forecast = torch.cat([x_future_forecast, padding], dim=1)
             # print(f"Warning: forecast_window ({forecast_window}) < output_window ({self.output_window}). Padding forecast input.")
             forecast_window = self.output_window # Ora x_future_forecast ha la lunghezza giusta

        outputs = torch.zeros(batch_size, self.output_window, target_output_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :] # Primo input al decoder

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell
            )
            outputs[:, t, :] = decoder_output_step

            # Logica di Teacher Forcing (principalmente per training, qui TF=0 per inferenza usa sempre l'output del decoder come input per il passo successivo se non diversamente specificato)
            # In inferenza pura, x_future_forecast potrebbe contenere solo il primo step o essere una previsione essa stessa.
            # Per il predict() che chiama questo forward(), x_future_forecast sono i dati reali o previsti per l'input del decoder.
            if t < self.output_window - 1:
                if teacher_forcing_ratio > 0 and random.random() < teacher_forcing_ratio : # Teacher forcing (usato in training)
                    # Questo path non dovrebbe essere preso in inferenza se teacher_forcing_ratio=0.0
                    # Prende l'input target reale dal futuro (non implementato qui perché il forward non riceve y_target)
                    # In training, il teacher forcing è gestito esplicitamente nel loop di training
                     pass # Si dovrebbe usare y_target[t+1] se disponibile
                else: # Usa l'input fornito x_future_forecast per il prossimo step
                    decoder_input_step = x_future_forecast[:, t+1:t+2, :]
        return outputs

# --- Funzioni Utilità Modello/Dati ---
def prepare_training_data(df, feature_columns, target_columns, input_window, output_window, # REMOVED val_split
                          lag_config=None, cumulative_config=None):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati LSTM Standard (full scaled data)...")
    # val_split parameter is removed as TimeSeriesSplit will be handled in train_model
    # --- Feature Engineering Start ---
    original_feature_columns = list(feature_columns) # Keep a copy
    engineered_feature_names = []

    # Example configurations (these would eventually come from UI)
    # lag_config = {'Cumulata Sensore 1295 (Arcevia)': [1, 3, 6]} # Lag in hours
    # cumulative_config = {'Cumulata Sensore 1295 (Arcevia)': [3, 6, 12]} # Window in hours

    if lag_config:
        for col, lag_periods_hours in lag_config.items():
            if col in df.columns:
                for lag_hr in lag_periods_hours:
                    lag_steps = lag_hr * 2 # Convert hours to 30-min steps
                    new_col_name = f"{col}_lag{lag_hr}h"
                    df[new_col_name] = df[col].shift(lag_steps)
                    engineered_feature_names.append(new_col_name)
                    print(f"Created lagged feature: {new_col_name}")
            else:
                st.warning(f"Colonna '{col}' specificata in lag_config non trovata nel DataFrame.")

    if cumulative_config:
        for col, window_periods_hours in cumulative_config.items():
            if col in df.columns:
                for win_hr in window_periods_hours:
                    window_steps = win_hr * 2 # Convert hours to 30-min steps
                    new_col_name = f"{col}_cum{win_hr}h"
                    df[new_col_name] = df[col].rolling(window=window_steps, min_periods=1).sum()
                    engineered_feature_names.append(new_col_name)
                    print(f"Created cumulative feature: {new_col_name}")
            else:
                st.warning(f"Colonna '{col}' specificata in cumulative_config non trovata nel DataFrame.")

    current_feature_columns = original_feature_columns + engineered_feature_names

    if engineered_feature_names:
        cols_to_fill_na = current_feature_columns + target_columns # Include targets just in case, though less likely to have new NaNs here
        cols_present_in_df_to_fill = [c for c in cols_to_fill_na if c in df.columns]

        nan_count_before_bfill = df[cols_present_in_df_to_fill].isnull().sum().sum()
        if nan_count_before_bfill > 0:
            df[cols_present_in_df_to_fill] = df[cols_present_in_df_to_fill].fillna(method='bfill')
            nan_count_after_bfill = df[cols_present_in_df_to_fill].isnull().sum().sum()
            st.warning(f"NaNs from feature engineering: {nan_count_before_bfill} before bfill, {nan_count_after_bfill} after bfill. "
                       f"Prime righe potrebbero essere inutilizzabili se i NaN persistono all'inizio.")
            # It's possible some NaNs remain at the very start if bfill can't fill them
            # These will be caught by the isnull().any().any() check below or lead to issues in scaling/training
    # --- Feature Engineering End ---

    try:
        # Use current_feature_columns which includes engineered features
        missing_features = [col for col in current_feature_columns if col not in df.columns]
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_features:
            st.error(f"Errore: Feature columns (originali o ingegnerizzate) mancanti nel DataFrame: {missing_features}")
            return None, None, None, None, None, None
        if missing_targets:
            st.error(f"Errore: Target columns mancanti nel DataFrame: {missing_targets}")
            return None, None, None, None, None, None

        # Check NaNs in the relevant slice of data that will be used for sequences
        # This check is now more critical after potential bfill
        if df[current_feature_columns + target_columns].isnull().any().any():
             st.warning("NaN trovati nelle colonne rilevanti PRIMA della creazione sequenze LSTM. "
                        "Questo potrebbe essere dovuto a NaNs all'inizio del dataset che 'bfill' non ha potuto riempire. "
                        "Le prime sequenze potrebbero essere droppate o causare errori. Controllare pulizia dati.")
             # Option: df.dropna(subset=current_feature_columns + target_columns, inplace=True)
             # However, this would change indices and complicate sequence creation logic.
             # The loop for sequence creation naturally handles this by starting where enough data is available.
    except Exception as e:
        st.error(f"Errore controllo colonne in prepare_training_data: {e}")
        return None, None, None, None, None, None

    X, y = [], []
    total_len = len(df)
    input_steps = input_window * 2 # ore * 2 (mezz'ore)
    output_steps = output_window * 2 # ore * 2 (mezz'ore)
    required_len = input_steps + output_steps

    if total_len < required_len:
         st.error(f"Dati LSTM insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
         return None, None, None, None, None, None

    # Iterate and create sequences. Rows with NaNs (especially at the beginning after bfill)
    # in the window [i : i + required_len] for current_feature_columns or target_columns
    # will result in sequences containing NaNs.
    for i in range(total_len - required_len + 1):
        # Slicing df first, then checking for NaNs in that specific window
        # This is important because NaNs at the very start (even after bfill) might affect the first few potential sequences
        feature_window_data = df.iloc[i : i + input_steps][current_feature_columns]
        target_window_data = df.iloc[i + input_steps : i + required_len][target_columns]

        if feature_window_data.isnull().any().any() or target_window_data.isnull().any().any():
            # Skip this sequence if any NaN is present in the data slice for features or targets
            # This handles NaNs at the beginning of the dataframe that bfill couldn't resolve
            # print(f"Skipping sequence starting at index {i} due to NaNs in window.") # For debugging
            continue

        X.append(feature_window_data.values)
        y.append(target_window_data.values)

    if not X or not y:
        st.error("Errore creazione sequenze X/y LSTM (possibilmente a causa di troppi NaN iniziali o dati insufficienti post-fillna)."); return None, None, None, None, None, None
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)

    if X.size == 0 or y.size == 0:
        st.error("Dati X o y vuoti prima di scaling LSTM."); return None, None, None, None, None, None

    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    num_sequences, seq_len_in, num_features_X = X.shape
    num_sequences_y, seq_len_out, num_targets_y = y.shape

    # Validate against current_feature_columns used for X
    if num_features_X != len(current_feature_columns):
        st.error(f"Errore numero feature in X ({num_features_X}) vs colonne attese ({len(current_feature_columns)}).")
        return None, None, None, None, None, None

    if seq_len_in != input_steps or seq_len_out != output_steps:
        st.error(f"Errore shape sequenze LSTM: In={seq_len_in} (atteso {input_steps}), Out={seq_len_out} (atteso {output_steps})")
        return None, None, None, None, None, None

    X_flat = X.reshape(-1, num_features_X); y_flat = y.reshape(-1, num_targets_y)
    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except ValueError as ve_scale: # Catch specific error for NaNs
        if "Input contains NaN" in str(ve_scale):
             st.error(f"Errore scaling LSTM: Input contiene NaN. Questo può accadere se 'bfill' non ha risolto tutti i NaN o se sono stati introdotti successivamente. Dettagli: {ve_scale}")
        else:
             st.error(f"Errore scaling LSTM: {ve_scale}")
        return None, None, None, None, None, None
    except Exception as e_scale:
        st.error(f"Errore scaling LSTM generico: {e_scale}"); return None, None, None, None, None, None

    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features_X)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets_y)

    # REMOVED: val_split logic is now handled by TimeSeriesSplit in train_model
    # split_idx = int(len(X_scaled) * (1 - val_split / 100))
    # ... (rest of old split logic) ...

    print(f"Dati LSTM pronti: X_scaled shape={X_scaled.shape}, y_scaled shape={y_scaled.shape}. Features utilizzate: {len(current_feature_columns)}")
    # Return full scaled datasets
    return X_scaled, y_scaled, scaler_features, scaler_targets


def prepare_training_data_seq2seq(df, past_feature_cols, forecast_feature_cols, target_cols,
                                 input_window_steps, forecast_window_steps, output_window_steps, # REMOVED val_split
                                 lag_config_past=None, cumulative_config_past=None):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq (full scaled data)...")
    # val_split parameter is removed as TimeSeriesSplit will be handled in train_model_seq2seq
    print(f"Input Steps: {input_window_steps}, Forecast Steps: {forecast_window_steps}, Output Steps: {output_window_steps}")

    # --- Feature Engineering for past_feature_cols Start ---
    original_past_feature_cols = list(past_feature_cols)
    engineered_past_feature_names = []

    # Example configs (would come from UI)
    # lag_config_past = {'Cumulata Sensore 1295 (Arcevia)': [1, 3, 6]} # Lag in hours
    # cumulative_config_past = {'Cumulata Sensore 1295 (Arcevia)': [3, 6, 12]} # Window in hours

    if lag_config_past:
        for col, lag_periods_hours in lag_config_past.items():
            if col in df.columns:
                for lag_hr in lag_periods_hours:
                    lag_steps = lag_hr * 2
                    new_col_name = f"{col}_lag{lag_hr}h"
                    df[new_col_name] = df[col].shift(lag_steps)
                    engineered_past_feature_names.append(new_col_name)
                    print(f"Created lagged past feature: {new_col_name}")
            else:
                st.warning(f"Colonna '{col}' specificata in lag_config_past non trovata.")

    if cumulative_config_past:
        for col, window_periods_hours in cumulative_config_past.items():
            if col in df.columns:
                for win_hr in window_periods_hours:
                    window_steps = win_hr * 2
                    new_col_name = f"{col}_cum{win_hr}h"
                    df[new_col_name] = df[col].rolling(window=window_steps, min_periods=1).sum()
                    engineered_past_feature_names.append(new_col_name)
                    print(f"Created cumulative past feature: {new_col_name}")
            else:
                st.warning(f"Colonna '{col}' specificata in cumulative_config_past non trovata.")

    current_past_feature_cols = original_past_feature_cols + engineered_past_feature_names

    # Note: Feature engineering for forecast_feature_cols is not implemented in this iteration.
    # These are typically future known values, and applying historical transformations might not be appropriate
    # or could lead to data leakage if not handled with extreme care.
    # --- Feature Engineering for past_feature_cols End ---

    all_needed_cols = list(set(current_past_feature_cols + forecast_feature_cols + target_cols))
    try:
        for col in all_needed_cols: # Check based on potentially expanded current_past_feature_cols
            if col not in df.columns: raise ValueError(f"Colonna '{col}' richiesta per Seq2Seq non trovata.")

        # Apply fillna (bfill then ffill) after feature engineering and before sequence creation
        # This aims to handle NaNs from shifting/rolling and any pre-existing NaNs.
        # Using a combined list of columns that will be used.
        cols_for_fillna_seq2seq = list(set(current_past_feature_cols + forecast_feature_cols + target_cols))
        cols_present_for_fillna_s2s = [c for c in cols_for_fillna_seq2seq if c in df.columns]

        nan_sum_before_fill = df[cols_present_for_fillna_s2s].isnull().sum().sum()
        if nan_sum_before_fill > 0:
            st.warning(f"NaN trovati ({nan_sum_before_fill}) prima della creazione sequenze Seq2Seq (post-FE). Applico bfill/ffill.")
            df[cols_present_for_fillna_s2s] = df[cols_present_for_fillna_s2s].fillna(method='bfill').fillna(method='ffill')
            nan_sum_after_fill = df[cols_present_for_fillna_s2s].isnull().sum().sum()
            if nan_sum_after_fill > 0:
                 st.error(f"NaN RESIDUI ({nan_sum_after_fill}) dopo bfill/ffill. Controlla colonne completamente vuote o NaNs all'estremo inizio/fine.")
                 # For Seq2Seq, we might not be able to proceed if critical NaNs remain, as sequences are built on these assumptions.
                 # Unlike LSTM, where we can more easily skip initial bad sequences, here it's more complex.
                 # Consider returning None if critical NaNs persist.
                 return None, None, None, None, None, None, None, None, None
        df_to_use = df # df is now modified in place or reassigned if using .copy() earlier

    except ValueError as e:
        st.error(f"Errore colonne in prepare_seq2seq: {e}")
        return None, None, None, None, None, None, None, None, None

    X_encoder, X_decoder, y_target = [], [], []
    total_len = len(df_to_use)
    required_len = input_window_steps + max(forecast_window_steps, output_window_steps)

    if total_len < required_len:
        st.error(f"Dati Seq2Seq insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
        return None, None, None, None, None, None, None, None, None

    print(f"Creazione sequenze Seq2Seq: {total_len - required_len + 1} possibili...")
    for i in range(total_len - required_len + 1):
        enc_end = i + input_window_steps
        # Check for NaNs in the specific window for this sequence
        past_feature_window_data = df_to_use.iloc[i : enc_end][current_past_feature_cols]

        dec_start = enc_end
        dec_end_forecast = dec_start + forecast_window_steps
        forecast_feature_window_data = df_to_use.iloc[dec_start : dec_end_forecast][forecast_feature_cols]

        target_end = dec_start + output_window_steps
        target_window_data = df_to_use.iloc[dec_start : target_end][target_cols]

        if past_feature_window_data.isnull().any().any() or \
           forecast_feature_window_data.isnull().any().any() or \
           target_window_data.isnull().any().any():
            # print(f"Skipping Seq2Seq sequence at index {i} due to NaNs in window.") # Debug
            continue

        X_encoder.append(past_feature_window_data.values)
        X_decoder.append(forecast_feature_window_data.values)
        y_target.append(target_window_data.values)


    if not X_encoder or not X_decoder or not y_target:
        st.error("Errore creazione sequenze X_enc/X_dec/Y_target Seq2Seq (possibilmente a causa di troppi NaN o dati insufficienti post-fillna).")
        return None, None, None, None, None, None, None, None, None

    X_encoder = np.array(X_encoder, dtype=np.float32)
    X_decoder = np.array(X_decoder, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    print(f"Shapes NumPy: Encoder={X_encoder.shape}, Decoder={X_decoder.shape}, Target={y_target.shape}")

    if X_encoder.shape[1] != input_window_steps or X_decoder.shape[1] != forecast_window_steps or y_target.shape[1] != output_window_steps:
         st.error("Errore shape sequenze Seq2Seq dopo creazione.")
         return None, None, None, None, None, None, None, None, None
    # Validate number of features against current_past_feature_cols
    if X_encoder.shape[2] != len(current_past_feature_cols) or X_decoder.shape[2] != len(forecast_feature_cols) or y_target.shape[2] != len(target_cols):
         st.error(f"Errore numero feature/target nelle sequenze Seq2Seq. Encoder features: {X_encoder.shape[2]} (atteso {len(current_past_feature_cols)}), Decoder features: {X_decoder.shape[2]} (atteso {len(forecast_feature_cols)}), Target features: {y_target.shape[2]} (atteso {len(target_cols)})")
         return None, None, None, None, None, None, None, None, None

    scaler_past_features = MinMaxScaler()
    scaler_forecast_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    num_sequences = X_encoder.shape[0]
    X_enc_flat = X_encoder.reshape(-1, len(current_past_feature_cols)) # Use current_past_feature_cols
    X_dec_flat = X_decoder.reshape(-1, len(forecast_feature_cols))
    y_tar_flat = y_target.reshape(-1, len(target_cols))

    try:
        X_enc_scaled_flat = scaler_past_features.fit_transform(X_enc_flat)
        X_dec_scaled_flat = scaler_forecast_features.fit_transform(X_dec_flat)
        y_tar_scaled_flat = scaler_targets.fit_transform(y_tar_flat)
    except ValueError as ve_scale_s2s: # Catch specific error for NaNs
        if "Input contains NaN" in str(ve_scale_s2s):
             st.error(f"Errore scaling Seq2Seq: Input contiene NaN. Questo può accadere se bfill/ffill non ha risolto tutti i NaN. Dettagli: {ve_scale_s2s}")
        else:
             st.error(f"Errore scaling Seq2Seq: {ve_scale_s2s}")
        return None, None, None, None, None, None, None, None, None
    except Exception as e_scale:
        st.error(f"Errore scaling Seq2Seq generico: {e_scale}")
        return None, None, None, None, None, None, None, None, None

    X_enc_scaled = X_enc_scaled_flat.reshape(num_sequences, input_window_steps, len(current_past_feature_cols)) # Use current_past_feature_cols
    X_dec_scaled = X_dec_scaled_flat.reshape(num_sequences, forecast_window_steps, len(forecast_feature_cols))
    y_tar_scaled = y_tar_scaled_flat.reshape(num_sequences, output_window_steps, len(target_cols))

    # REMOVED: val_split logic is now handled by TimeSeriesSplit in train_model_seq2seq
    # split_idx = int(num_sequences * (1 - val_split / 100))
    # ... (rest of old split logic) ...

    print(f"Dati Seq2Seq pronti: X_enc_scaled={X_enc_scaled.shape}, X_dec_scaled={X_dec_scaled.shape}, y_tar_scaled={y_tar_scaled.shape}. Past features utilizzate: {len(current_past_feature_cols)}")
    # Return full scaled datasets
    return (X_enc_scaled, X_dec_scaled, y_tar_scaled,
            scaler_past_features, scaler_forecast_features, scaler_targets)


# --- Funzioni Caricamento Modello/Scaler (Aggiornate) ---
@st.cache_data(show_spinner="Ricerca modelli disponibili...")
def find_available_models(models_dir=MODELS_DIR):
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
                required_keys = ["input_window_steps", "output_window_steps", "forecast_window_steps",
                                 "hidden_size", "num_layers", "dropout",
                                 "all_past_feature_columns", "forecast_input_columns", "target_columns"]
                s_past_p = os.path.join(models_dir, f"{base}_past_features.joblib")
                s_fore_p = os.path.join(models_dir, f"{base}_forecast_features.joblib")
                s_targ_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(s_past_p) and os.path.exists(s_fore_p) and os.path.exists(s_targ_p)):
                    model_info.update({
                        "scaler_past_features_path": s_past_p,
                        "scaler_forecast_features_path": s_fore_p,
                        "scaler_targets_path": s_targ_p,
                        "model_type": "Seq2Seq"
                    })
                    valid_model = True
            else: # LSTM Standard
                required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns", "feature_columns"]
                scf_p = os.path.join(models_dir, f"{base}_features.joblib")
                sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(scf_p) and os.path.exists(sct_p)):
                     model_info.update({
                         "scaler_features_path": scf_p,
                         "scaler_targets_path": sct_p,
                         "model_type": "LSTM"
                     })
                     valid_model = True
        except Exception as e_cfg:
            st.warning(f"Modello '{base}' ignorato: errore lettura/config JSON ({cfg_path}) non valida: {e_cfg}")
            valid_model = False

        if valid_model:
            available[name] = model_info
        else:
            print(f"Modello '{base}' ignorato: file associati mancanti o config JSON incompleta per tipo '{model_type}'.")
    return available

@st.cache_data
def load_model_config(_config_path):
    try:
        with open(_config_path, 'r', encoding='utf-8') as f: config = json.load(f)
        return config
    except Exception as e:
        st.error(f"Errore caricamento config '{_config_path}': {e}"); return None

@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    if not config: st.error("Config non valida per caricamento modello."); return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get("model_type", "LSTM")

    try:
        model = None
        if model_type == "Seq2Seq":
            enc_input_size = len(config["all_past_feature_columns"])
            dec_input_size = len(config["forecast_input_columns"])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]; layers = config["num_layers"]; drop = config["dropout"]
            out_win = config["output_window_steps"]
            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            decoder = DecoderLSTM(dec_input_size, hidden, dec_output_size, layers, drop).to(device)
            model = Seq2SeqHydro(encoder, decoder, out_win, device).to(device)
        else: # LSTM Standard
            f_cols_lstm = config.get("feature_columns")
            if not f_cols_lstm:
                 # Questo blocco potrebbe non essere necessario se la config viene sempre salvata con feature_columns
                 # st.warning(f"Config LSTM '{config.get('display_name', 'N/A')}' non specifica feature_columns. Uso le globali.")
                 # f_cols_lstm = st.session_state.get("feature_columns", []) # Rimuovere dipendenza da session_state qui
                 # if not f_cols_lstm: raise ValueError("Feature globali non definite per modello LSTM caricato.")
                 raise ValueError(f"Chiave 'feature_columns' mancante nella config per il modello LSTM '{config.get('display_name', 'N/A')}'.")

            input_size_lstm = len(f_cols_lstm)
            target_size_lstm = len(config["target_columns"])
            out_win_lstm = config["output_window"]
            model = HydroLSTM(input_size_lstm, config["hidden_size"], target_size_lstm,
                              out_win_lstm, config["num_layers"], config["dropout"]).to(device)

        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'): # Supporto per file in memoria (es. BytesIO da upload)
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")
        model.eval()
        print(f"Modello '{config.get('display_name', 'N/A')}' (Tipo: {model_type}) caricato su {device}.")
        return model, device
    except Exception as e:
        st.error(f"Errore caricamento modello '{config.get('display_name', 'N/A')}' (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc()); return None, None

@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(config, model_info):
    if not config or not model_info: st.error("Config o info modello mancanti per caricare scaler."); return None
    model_type = config.get("model_type", "LSTM")

    def _load_joblib(path):
         if isinstance(path, str):
              if not os.path.exists(path): raise FileNotFoundError(f"File scaler '{path}' non trovato.")
              return joblib.load(path)
         elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path) # Supporto BytesIO
         else: raise TypeError(f"Percorso scaler non valido: {type(path)}")

    try:
        if model_type == "Seq2Seq":
            scaler_past = _load_joblib(model_info["scaler_past_features_path"])
            scaler_forecast = _load_joblib(model_info["scaler_forecast_features_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            print(f"Scaler Seq2Seq caricati.")
            return {"past": scaler_past, "forecast": scaler_forecast, "targets": scaler_targets}
        else: # LSTM Standard
            scaler_features = _load_joblib(model_info["scaler_features_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            print(f"Scaler LSTM caricati.")
            return scaler_features, scaler_targets
    except Exception as e:
        st.error(f"Errore caricamento scaler (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc())
        if model_type == "Seq2Seq": return None
        else: return None, None

# --- Funzioni Predict (Standard e Seq2Seq) ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict LSTM: Modello, scaler o config mancanti."); return None

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM": st.error(f"Funzione predict chiamata su modello non LSTM (tipo: {model_type})"); return None

    input_steps = config["input_window"] # Numero di righe da 30min
    output_steps = config["output_window"] # Numero di righe da 30min
    target_cols = config["target_columns"]
    f_cols_cfg = config.get("feature_columns", []) # Deve essere presente per LSTM

    if input_data.shape[0] != input_steps:
        st.error(f"Predict LSTM: Input righe {input_data.shape[0]} != Steps richiesti {input_steps}."); return None
    expected_features = getattr(scaler_features, 'n_features_in_', len(f_cols_cfg) if f_cols_cfg else None)
    if expected_features is not None and input_data.shape[1] != expected_features:
        st.error(f"Predict LSTM: Input colonne {input_data.shape[1]} != Features attese {expected_features}."); return None

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device) # (1, input_steps, num_features)
        with torch.no_grad(): output = model(inp_tens) # Output: (1, output_steps, num_targets)

        if output.shape != (1, output_steps, len(target_cols)):
             st.error(f"Predict LSTM: Shape output modello {output.shape} inattesa. Attesa: (1, {output_steps}, {len(target_cols)})"); return None
        out_np = output.cpu().numpy().squeeze(0) # (output_steps, num_targets)

        # scaler_targets si aspetta (num_samples, num_features)
        # out_np ha già la forma corretta (output_steps come num_samples, num_targets come num_features)
        expected_targets_scaler = getattr(scaler_targets, 'n_features_in_', len(target_cols))

        if out_np.shape[1] != expected_targets_scaler:
            st.error(f"Predict LSTM: Numero colonne output modello ({out_np.shape[1]}) non corrisponde a target scaler/config ({expected_targets_scaler}).")
            return None
        preds = scaler_targets.inverse_transform(out_np) # (output_steps, num_targets)
        return preds
    except Exception as e:
        st.error(f"Errore durante predict LSTM: {e}")
        st.error(traceback.format_exc()); return None

def predict_seq2seq(model, past_data, future_forecast_data, scalers, config, device):
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]):
         st.error("Predict Seq2Seq: Input mancanti."); return None

    model_type = config.get("model_type")
    if model_type != "Seq2Seq": st.error(f"Funzione predict_seq2seq chiamata su modello non Seq2Seq"); return None

    input_steps = config["input_window_steps"]
    forecast_steps_input_decoder = config["forecast_window_steps"] # Quanti input forniamo al decoder
    output_steps_model = config["output_window_steps"] # Quanti output il modello è addestrato a produrre
    past_cols = config["all_past_feature_columns"]
    forecast_cols = config["forecast_input_columns"]
    target_cols = config["target_columns"]

    scaler_past = scalers.get("past"); scaler_forecast = scalers.get("forecast"); scaler_targets = scalers.get("targets")
    if not all([scaler_past, scaler_forecast, scaler_targets]):
        st.error("Predict Seq2Seq: Scaler mancanti nel dizionario fornito."); return None

    if past_data.shape != (input_steps, len(past_cols)):
        st.error(f"Predict Seq2Seq: Shape dati passati {past_data.shape} errata (attesa ({input_steps}, {len(past_cols)}))."); return None

    # future_forecast_data è l'input per il decoder. La sua lunghezza deve essere almeno output_steps_model
    # Se è più corta, il modello la padderà internamente nel forward.
    # Se è più lunga, il modello userà solo i primi output_steps_model.
    if future_forecast_data.shape[0] < output_steps_model:
         st.warning(f"Predict Seq2Seq: Finestra input forecast ({future_forecast_data.shape[0]}) < finestra output modello ({output_steps_model}). Il modello userà padding.")
    if future_forecast_data.shape[1] != len(forecast_cols):
         st.error(f"Predict Seq2Seq: Numero colonne dati forecast ({future_forecast_data.shape[1]}) errato (atteso {len(forecast_cols)})."); return None


    model.eval()
    try:
        past_norm = scaler_past.transform(past_data)
        future_norm = scaler_forecast.transform(future_forecast_data)
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device)
        future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)

        with torch.no_grad(): output = model(past_tens, future_tens, teacher_forcing_ratio=0.0) # Output: (1, output_steps_model, num_targets)

        if output.shape != (1, output_steps_model, len(target_cols)):
             st.error(f"Predict Seq2Seq: Shape output modello {output.shape} inattesa (attesa (1, {output_steps_model}, {len(target_cols)}))."); return None
        out_np = output.cpu().numpy().squeeze(0) # (output_steps_model, num_targets)

        expected_targets_scaler = getattr(scaler_targets, 'n_features_in_', len(target_cols))
        if out_np.shape[1] != expected_targets_scaler:
             st.error(f"Predict Seq2Seq: Numero colonne output modello ({out_np.shape[1]}) non corrisponde a target scaler/config ({expected_targets_scaler})."); return None
        preds = scaler_targets.inverse_transform(out_np)
        return preds
    except Exception as e:
        st.error(f"Errore durante predict Seq2Seq: {e}")
        st.error(traceback.format_exc()); return None

# --- Funzione Grafici Previsioni (MODIFICATA per includere dati reali opzionali) ---
def plot_predictions(predictions, config, start_time=None, actual_data=None, actual_data_label="Dati Reali CSV"):
    """
    Genera grafici Plotly INDIVIDUALI per le previsioni.
    Se `actual_data` è fornito, lo plotta accanto alle previsioni per confronto.
    Aggiunge linee orizzontali per le soglie H e un secondo asse Y per la portata Q (solo prevista).
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
            time_steps_datetime = [start_time + timedelta(minutes=30 * (step + 1)) for step in range(output_steps)] # start_time è il primo punto di previsione
            x_axis, x_title = time_steps_datetime, "Data e Ora"
            x_tick_format = "%d/%m %H:%M"
        else:
            time_steps_relative = np.arange(1, output_steps + 1) * 0.5
            x_axis, x_title = time_steps_relative, "Ore Future (passi da 30 min)"
            x_tick_format = None
        x_axis_np = np.array(x_axis)

        station_name_graph = get_station_label(sensor, short=False)
        if actual_data is not None:
            plot_title_base = f'Test: Previsto vs Reale - {station_name_graph}'
        else:
            plot_title_base = f'Previsione {model_type} - {station_name_graph}'
        plot_title = f'{plot_title_base}<br><span style="font-size:10px;">{attribution_text}</span>'


        unit_match = re.search(r'\((.*?)\)|\[(.*?)\]', sensor)
        y_axis_unit = "m"; y_axis_title_h = "Livello H (m)"
        if unit_match:
            unit_content = unit_match.group(1) or unit_match.group(2)
            if unit_content:
                y_axis_unit = unit_content.strip()
                y_axis_title_h = f"Livello H ({y_axis_unit})"

        # Traccia Previsioni H
        fig.add_trace(go.Scatter(
            x=x_axis_np, y=predictions[:, i], mode='lines+markers', name=f'Previsto H ({y_axis_unit})', yaxis='y1'
        ))

        # Traccia Dati Reali H (se forniti)
        if actual_data is not None:
            if actual_data.ndim == 2 and actual_data.shape[0] == output_steps and actual_data.shape[1] == len(target_cols):
                fig.add_trace(go.Scatter(
                    x=x_axis_np, y=actual_data[:, i], mode='lines', name=f'{actual_data_label} H ({y_axis_unit})',
                    line=dict(color='green', dash='dashdot'), yaxis='y1'
                ))
            else:
                st.warning(f"Shape 'actual_data' ({actual_data.shape}) non compatibile per {sensor}. Attesa: ({output_steps}, {len(target_cols)})")

        # Soglie H
        threshold_info = SIMULATION_THRESHOLDS.get(sensor, {})
        soglia_attenzione = threshold_info.get('attenzione')
        soglia_allerta = threshold_info.get('allerta')
        if soglia_attenzione is not None:
            fig.add_hline(y=soglia_attenzione, line_dash="dash", line_color="orange", annotation_text=f"Att.H({soglia_attenzione:.2f})", annotation_position="bottom right", layer="below")
        if soglia_allerta is not None:
            fig.add_hline(y=soglia_allerta, line_dash="dash", line_color="red", annotation_text=f"All.H({soglia_allerta:.2f})", annotation_position="top right", layer="below")

        # Dati Portata (Q) PREVISTA e Asse Y2 (Condizionale)
        sensor_info = STATION_COORDS.get(sensor)
        sensor_code_plot = sensor_info.get('sensor_code') if sensor_info else None
        has_discharge_data = False
        if sensor_code_plot and sensor_code_plot in RATING_CURVES:
            predicted_H_values = predictions[:, i]
            predicted_Q_values = calculate_discharge_vectorized(sensor_code_plot, predicted_H_values)
            valid_Q_mask = pd.notna(predicted_Q_values)
            if np.any(valid_Q_mask):
                has_discharge_data = True
                fig.add_trace(go.Scatter(
                    x=x_axis_np[valid_Q_mask], y=predicted_Q_values[valid_Q_mask], mode='lines', name='Portata Prevista Q (m³/s)',
                    line=dict(color='firebrick', dash='dot'), yaxis='y2'
                ))

        # Aggiornamento Layout
        try:
            fig.update_layout(title=plot_title, height=400, margin=dict(l=60, r=60, t=70, b=50), hovermode="x unified", template="plotly_white")
            fig.update_xaxes(title_text=x_title)
            if x_tick_format: fig.update_xaxes(tickformat=x_tick_format)

            fig.update_yaxes(title=dict(text=y_axis_title_h, font=dict(color="#1f77b4")), tickfont=dict(color="#1f77b4"), side="left", rangemode='tozero')

            if has_discharge_data: # Se c'è asse Y2 per Q prevista
                fig.update_layout(
                     yaxis2=dict(title=dict(text="Portata Q (m³/s)", font=dict(color="firebrick")),
                                 tickfont=dict(color="firebrick"), overlaying="y", side="right", rangemode='tozero', showgrid=False),
                     legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) # Legenda sotto
                )
            elif actual_data is not None and actual_data.ndim == 2 : # Se ci sono dati reali (ma non Q prevista sull'asse Y2)
                 fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)) # Legenda sotto ma più alta
            else: # Solo H previsto
                 fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1)) # Legenda default a destra

        except Exception as e_layout:
            st.error(f"Errore layout Plotly: {e_layout}"); print(traceback.format_exc())
            # Non rilanciare l'errore per permettere agli altri grafici di essere generati
        figs.append(fig)
    return figs

# --- INIZIO MODIFICA: Funzione di Fetch Dati Google Sheet (Dashboard e Simulazione) ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dai fogli Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id,
                                fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS):
    """
    Recupera, processa e unisce i dati dai fogli 'Dati Idrometeo CAMI' (o sheet1) e 'Dati Stazioni Weatherlink'.
    """
    mode = "tutti i dati" if fetch_all else f"ultime {num_rows_default} righe"
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data ({mode})")
    actual_fetch_time = datetime.now(italy_tz)
    error_messages = []

    def _process_sheet_data(worksheet, relevant_cols, sheet_name_for_log):
        """Funzione helper per processare i dati da un singolo foglio."""
        try:
            all_values = worksheet.get_all_values()
            if not all_values or len(all_values) < 2:
                error_messages.append(f"Attenzione: Foglio '{sheet_name_for_log}' vuoto o con solo intestazione.")
                return pd.DataFrame(), []

            headers = all_values[0]
            if fetch_all:
                data_rows = all_values[1:]
            else:
                start_index = max(1, len(all_values) - num_rows_default)
                data_rows = all_values[start_index:]

            missing_cols = [col for col in relevant_cols if col not in headers and col != GSHEET_DATE_COL]
            if missing_cols:
                error_messages.append(f"Attenzione: Colonne mancanti in '{sheet_name_for_log}': {', '.join(missing_cols)}")

            df = pd.DataFrame(data_rows, columns=headers)
            cols_to_select = [c for c in relevant_cols if c in df.columns]
            df = df[cols_to_select]

            parsing_errors = []
            for col in df.columns:
                if col == GSHEET_DATE_COL:
                    try:
                        df[col] = pd.to_datetime(df[col], format=GSHEET_DATE_FORMAT, errors='coerce', infer_datetime_format=True)
                        if df[col].isnull().any():
                            parsing_errors.append(f"Formato data non valido per '{col}' in '{sheet_name_for_log}'")
                        if not df[col].empty and df[col].notna().any():
                            if df[col].dt.tz is None:
                                df[col] = df[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                            else:
                                df[col] = df[col].dt.tz_convert(italy_tz)
                    except Exception as e_date:
                        parsing_errors.append(f"Errore data '{col}' in '{sheet_name_for_log}': {e_date}")
                        df[col] = pd.NaT
                else:
                    try:
                        df_col_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                        df[col] = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e_num:
                        parsing_errors.append(f"Errore numerico '{col}' in '{sheet_name_for_log}': {e_num}")
                        df[col] = np.nan
            
            return df, parsing_errors
        except Exception as e:
            error_messages.append(f"Errore recupero/processamento foglio '{sheet_name_for_log}': {e}")
            return pd.DataFrame(), []

    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets:
            return None, "Errore: Credenziali Google mancanti.", actual_fetch_time

        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id)

        # 1. Processa foglio principale (Idrometeo CAMI)
        try:
            worksheet_main = sh.sheet1 # Accede al primo foglio per default
            df_main, main_errors = _process_sheet_data(worksheet_main, MAIN_SHEET_RELEVANT_COLS, GSHEET_MAIN_NAME)
            if main_errors:
                error_messages.extend(main_errors)
        except gspread.exceptions.WorksheetNotFound:
            df_main = pd.DataFrame()
            error_messages.append(f"Attenzione: Foglio principale (primo foglio) non trovato.")
        except Exception as e_main:
            df_main = pd.DataFrame()
            error_messages.append(f"Errore imprevisto foglio principale: {e_main}")

        # 2. Processa foglio secondario (Weatherlink)
        try:
            worksheet_weatherlink = sh.worksheet(GSHEET_WEATHERLINK_NAME)
            df_weatherlink, weatherlink_errors = _process_sheet_data(worksheet_weatherlink, WEATHERLINK_RELEVANT_COLS, GSHEET_WEATHERLINK_NAME)
            if weatherlink_errors:
                error_messages.extend(weatherlink_errors)
        except gspread.exceptions.WorksheetNotFound:
            df_weatherlink = pd.DataFrame()
            error_messages.append(f"Attenzione: Foglio '{GSHEET_WEATHERLINK_NAME}' non trovato.")
        except Exception as e_weatherlink:
            df_weatherlink = pd.DataFrame()
            error_messages.append(f"Errore imprevisto foglio '{GSHEET_WEATHERLINK_NAME}': {e_weatherlink}")

        # 3. Unisci i due DataFrame
        if not df_main.empty and not df_weatherlink.empty:
            # Assicura che la colonna di merge esista in entrambi
            if GSHEET_DATE_COL in df_main.columns and GSHEET_DATE_COL in df_weatherlink.columns:
                merged_df = pd.merge(df_main, df_weatherlink, on=GSHEET_DATE_COL, how='outer')
            else:
                error_messages.append(f"Colonna data '{GSHEET_DATE_COL}' mancante in uno dei fogli. Impossibile unire.")
                merged_df = df_main if not df_main.empty else df_weatherlink
        elif not df_main.empty:
            merged_df = df_main
        elif not df_weatherlink.empty:
            merged_df = df_weatherlink
        else:
            return None, "Errore: Nessun dato valido recuperato da entrambi i fogli.", actual_fetch_time

        # 4. Finalizza il DataFrame unito
        if not merged_df.empty and GSHEET_DATE_COL in merged_df.columns:
            # Rimuovi eventuali righe senza data e ordina
            merged_df.dropna(subset=[GSHEET_DATE_COL], inplace=True)
            merged_df = merged_df.sort_values(by=GSHEET_DATE_COL, na_position='first').reset_index(drop=True)
            # Rimuovi righe duplicate basate sulla data, mantenendo l'ultima
            merged_df = merged_df.drop_duplicates(subset=[GSHEET_DATE_COL], keep='last')
        else:
            st.warning("Dati finali vuoti o colonna data mancante dopo il merge.")

        final_error_message = "Attenzione conversione dati GSheet: " + " | ".join(error_messages) if error_messages else None
        return merged_df, final_error_message, actual_fetch_time

    except gspread.exceptions.APIError as api_e:
        try:
            error_details = api_e.response.json()
            error_msg = error_details.get('error', {}).get('message', str(api_e))
            status = error_details.get('error', {}).get('code', 'N/A')
            error_msg = f"Codice {status}: {error_msg}"
        except:
            error_msg = str(api_e)
        return None, f"Errore API Google Sheets: {error_msg}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound:
        return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}').", actual_fetch_time
    except Exception as e:
        return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time
# --- FINE MODIFICA ---

@st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet per simulazione...")
def fetch_sim_gsheet_data(sheet_id_fetch, n_rows_steps, date_col_gs, date_format_gs, col_mapping, required_model_cols_fetch, impute_dict):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_sim_gsheet_data (Steps: {n_rows_steps})")
    actual_fetch_time = datetime.now(italy_tz); last_valid_timestamp = None
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", None
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials); sh = gc.open_by_key(sheet_id_fetch); worksheet = sh.sheet1
        all_data_gs = worksheet.get_all_values()
        if not all_data_gs or len(all_data_gs) < 2 : return None, f"Errore: Foglio GSheet vuoto o senza dati.", None

        headers_gs = all_data_gs[0]
        start_index_gs = max(1, len(all_data_gs) - n_rows_steps)
        data_rows_gs = all_data_gs[start_index_gs:]
        df_gsheet_raw = pd.DataFrame(data_rows_gs, columns=headers_gs)

        required_gsheet_cols_from_mapping = list(set(list(col_mapping.keys()) + ([date_col_gs] if date_col_gs not in col_mapping.keys() else [])))
        missing_gsheet_cols_in_sheet = [c for c in required_gsheet_cols_from_mapping if c not in df_gsheet_raw.columns]
        if missing_gsheet_cols_in_sheet: return None, f"Errore: Colonne GSheet ({', '.join(missing_gsheet_cols_in_sheet)}) mancanti nel foglio.", None

        cols_to_select_gsheet = [c for c in required_gsheet_cols_from_mapping if c in df_gsheet_raw.columns]
        df_subset = df_gsheet_raw[cols_to_select_gsheet].copy()
        df_mapped = df_subset.rename(columns=col_mapping)

        for model_col, impute_val in impute_dict.items():
             if model_col not in df_mapped.columns: df_mapped[model_col] = impute_val

        final_missing_model_cols = [c for c in required_model_cols_fetch if c not in df_mapped.columns]
        if final_missing_model_cols: return None, f"Errore: Colonne modello ({', '.join(final_missing_model_cols)}) mancanti dopo mappatura/imputazione.", None

        date_col_model_name = col_mapping.get(date_col_gs, date_col_gs)
        if date_col_model_name not in df_mapped.columns: date_col_model_name = None

        numeric_model_cols = [c for c in required_model_cols_fetch if c != date_col_model_name]
        for col in numeric_model_cols:
            if col not in df_mapped.columns: continue
            try:
                if pd.api.types.is_object_dtype(df_mapped[col]) or pd.api.types.is_string_dtype(df_mapped[col]):
                    col_str = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    df_mapped[col] = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
            except Exception as e_clean_num: st.warning(f"Problema pulizia GSheet colonna '{col}' per simulazione: {e_clean_num}. Trattata come NaN."); df_mapped[col] = np.nan

        if date_col_model_name:
             try:
                 df_mapped[date_col_model_name] = pd.to_datetime(df_mapped[date_col_model_name], format=date_format_gs, errors='coerce', infer_datetime_format=True)
                 if df_mapped[date_col_model_name].isnull().any(): st.warning(f"Date non valide trovate in GSheet simulazione ('{date_col_model_name}').")
                 if not df_mapped[date_col_model_name].empty and df_mapped[date_col_model_name].notna().any():
                     if df_mapped[date_col_model_name].dt.tz is None: df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                     else: df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_convert(italy_tz)
                 df_mapped = df_mapped.sort_values(by=date_col_model_name, na_position='first')
                 valid_dates = df_mapped[date_col_model_name].dropna()
                 if not valid_dates.empty: last_valid_timestamp = valid_dates.iloc[-1]
             except Exception as e_date_clean: st.warning(f"Errore conversione/pulizia data GSheet simulazione '{date_col_model_name}': {e_date_clean}. Impossibile ordinare."); date_col_model_name = None

        try:
            cols_present_final = [c for c in required_model_cols_fetch if c in df_mapped.columns]
            df_final = df_mapped[cols_present_final].copy()
        except KeyError as e_key: return None, f"Errore selezione/ordine colonne finali simulazione: '{e_key}' mancante.", None

        numeric_cols_to_fill = df_final.select_dtypes(include=np.number).columns
        nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
        if nan_count_before > 0:
             st.warning(f"Trovati {nan_count_before} valori NaN nei dati GSheet per simulazione. Applico forward-fill e backward-fill.")
             df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(method='ffill').fillna(method='bfill')
             if df_final[numeric_cols_to_fill].isnull().sum().sum() > 0:
                 st.error(f"NaN residui ({df_final[numeric_cols_to_fill].isnull().sum().sum()}) dopo fillna. Tento fill con 0.")
                 df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(0)
        if len(df_final) != n_rows_steps: st.warning(f"Attenzione: Numero righe finali ({len(df_final)}) diverso da richiesto ({n_rows_steps}) dopo recupero GSheet.")
        try: df_final = df_final[required_model_cols_fetch]
        except KeyError as e_final_order: missing_final = [c for c in required_model_cols_fetch if c not in df_final.columns]; return None, f"Errore: Colonne modello ({missing_final}) mancanti prima del return finale.", None
        return df_final, None, last_valid_timestamp
    except gspread.exceptions.APIError as api_e_sim:
        try: error_details = api_e_sim.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e_sim)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}";
        except: error_msg = str(api_e_sim)
        return None, f"Errore API Google Sheets: {error_msg}", None
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google simulazione non trovato (ID: '{sheet_id_fetch}').", None
    except Exception as e_sim_fetch: st.error(traceback.format_exc()); return None, f"Errore imprevisto importazione GSheet per simulazione: {type(e_sim_fetch).__name__} - {e_sim_fetch}", None

# --- Funzioni Allenamento (Standard e Seq2Seq) --- MODIFICATE ---
def train_model(X_scaled_full, y_scaled_full, # CHANGED: from X_train, y_train, X_val, y_val
                input_size, output_size, output_window_steps,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2,
                save_strategy='migliore', preferred_device='auto',
                n_splits_cv=3, loss_function_name="MSELoss",
                _model_to_continue_train=None): # NEW: _model_to_continue_train
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training LSTM Standard con TimeSeriesSplit (n_splits={n_splits_cv}, loss={loss_function_name})...")

    if input_size <= 0 or output_size <= 0 or output_window_steps <= 0:
        st.error(f"Errore: Parametri modello LSTM non validi: input_size={input_size}, output_size={output_size}, output_window_steps={output_window_steps}")
        return None, ([], []), ([], []) # Ensure consistent return structure

    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"Training LSTM userà: {device}")

    # Loss Function Selection
    if loss_function_name == "HuberLoss":
        criterion = nn.HuberLoss(reduction='none')
        print("Using HuberLoss")
    else: # Default to MSE
        criterion = nn.MSELoss(reduction='none')
        print("Using MSELoss")

    if _model_to_continue_train is not None:
        model = _model_to_continue_train.to(device)
        print("Continuo training da modello LSTM esistente.")
        # Verifica parametri modello esistente
        if model.lstm.input_size != input_size:
            st.warning(f"Mismatch input_size: Modello esistente ({model.lstm.input_size}) vs Richiesto ({input_size}). Training interrotto.")
            return None, ([], []), ([], [])
        if model.output_size != output_size:
            st.warning(f"Mismatch output_size: Modello esistente ({model.output_size}) vs Richiesto ({output_size}). Training interrotto.")
            return None, ([], []), ([], [])
        if model.output_window != output_window_steps:
            st.warning(f"Mismatch output_window: Modello esistente ({model.output_window}) vs Richiesto ({output_window_steps}). Training interrotto.")
            return None, ([], []), ([], [])
    else:
        model = HydroLSTM(input_size, hidden_size, output_size, output_window_steps, num_layers, dropout).to(device)
        print("Creato nuovo modello LSTM per training.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # TimeSeriesSplit for final model training pass and per-epoch validation
    tscv_final_pass = TimeSeriesSplit(n_splits=n_splits_cv)
    all_splits_indices = list(tscv_final_pass.split(X_scaled_full))

    if not all_splits_indices:
        st.error("TimeSeriesSplit non ha prodotto alcun split. Controllare la dimensione dei dati e n_splits_cv.")
        return None, ([], []), ([], [])

    # Use the last split for training the model that will be returned and for per-epoch validation
    train_indices_final, val_indices_final = all_splits_indices[-1]

    X_train_final, y_train_final = X_scaled_full[train_indices_final], y_scaled_full[train_indices_final]
    X_val_final, y_val_final = X_scaled_full[val_indices_final], y_scaled_full[val_indices_final]

    if X_train_final.size == 0 or y_train_final.size == 0:
        st.error("Set di training finale (dall'ultimo split CV) è vuoto.")
        return None, ([], []), ([], [])

    train_dataset_final = TimeSeriesDataset(X_train_final, y_train_final)
    train_loader_final = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    val_loader_final = None
    if X_val_final.size > 0 and y_val_final.size > 0:
        val_dataset_final = TimeSeriesDataset(X_val_final, y_val_final)
        val_loader_final = DataLoader(val_dataset_final, batch_size=batch_size, num_workers=0, pin_memory=True if device.type == 'cuda' else False)
        print(f"Training finale su {len(X_train_final)} campioni, validazione finale su {len(X_val_final)} campioni (dall'ultimo split CV).")
    else:
        st.warning("Set di validazione finale (dall'ultimo split CV) è vuoto. La validazione per epoca e il salvataggio del modello migliore saranno limitati.")
        print(f"Training finale su {len(X_train_final)} campioni. Nessun set di validazione finale dall'ultimo split CV.")

    train_losses_scalar_history, val_losses_scalar_history = [], [] # For the final validation set
    train_losses_per_step_history, val_losses_per_step_history = [], [] # For the final validation set
    best_val_loss_scalar = float('inf') # Based on the final validation set
    best_model_state_dict = None
    progress_bar = st.progress(0.0, text="Training LSTM: Inizio..."); status_text = st.empty(); loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss_scalar, v_loss_scalar, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss_scalar, mode='lines', name='Train Loss (Scalar Avg)'))
        valid_v_loss_scalar = [v for v in v_loss_scalar if v is not None] if v_loss_scalar else []
        if valid_v_loss_scalar: fig.add_trace(go.Scatter(y=valid_v_loss_scalar, mode='lines', name='Validation Loss (Scalar Avg)'))
        fig.update_layout(title='Andamento Loss (LSTM - Media Scalare)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training LSTM per {epochs} epoche su **{device}**..."); start_training_time = pytime.time()
    for epoch in range(epochs):
        epoch_start_time = pytime.time()
        model.train()
        epoch_train_loss_scalar_sum = 0.0
        epoch_train_loss_per_step_sum = torch.zeros(output_window_steps, device=device)

        # Training loop uses X_train_final, y_train_final from the last CV split
        for i, batch_data in enumerate(train_loader_final):
            X_batch, y_batch = batch_data[0].to(device, non_blocking=True), batch_data[1].to(device, non_blocking=True)
            outputs = model(X_batch)

            loss_per_element = criterion(outputs, y_batch)
            scalar_loss_for_backward = loss_per_element.mean()

            optimizer.zero_grad()
            scalar_loss_for_backward.backward()
            optimizer.step()

            epoch_train_loss_scalar_sum += scalar_loss_for_backward.item() * X_batch.size(0)
            epoch_train_loss_per_step_sum += loss_per_element.mean(dim=2).sum(dim=0).detach()

        avg_epoch_train_loss_scalar = epoch_train_loss_scalar_sum / len(train_loader_final.dataset)
        train_losses_scalar_history.append(avg_epoch_train_loss_scalar)
        avg_epoch_train_loss_per_step = (epoch_train_loss_per_step_sum / len(train_loader_final.dataset)).cpu().numpy()
        train_losses_per_step_history.append(avg_epoch_train_loss_per_step)

        avg_epoch_val_loss_scalar = None # For the "final" validation set
        avg_epoch_val_loss_per_step = np.full(output_window_steps, np.nan) # For the "final" validation set

        if val_loader_final: # Use the val_loader for the final split
            model.eval()
            epoch_val_loss_scalar_sum = 0.0
            epoch_val_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for batch_data_val in val_loader_final:
                    X_batch_val, y_batch_val = batch_data_val[0].to(device, non_blocking=True), batch_data_val[1].to(device, non_blocking=True)
                    outputs_val = model(X_batch_val)
                    loss_per_element_val = criterion(outputs_val, y_batch_val)

                    scalar_loss_val_batch = loss_per_element_val.mean()
                    epoch_val_loss_scalar_sum += scalar_loss_val_batch.item() * X_batch_val.size(0)
                    epoch_val_loss_per_step_sum += loss_per_element_val.mean(dim=2).sum(dim=0).detach()

            if len(val_loader_final.dataset) > 0:
                avg_epoch_val_loss_scalar = epoch_val_loss_scalar_sum / len(val_loader_final.dataset)
                avg_epoch_val_loss_per_step = (epoch_val_loss_per_step_sum / len(val_loader_final.dataset)).cpu().numpy()
            else: # Should not happen if val_loader_final is not None
                avg_epoch_val_loss_scalar = float('inf')

            val_losses_scalar_history.append(avg_epoch_val_loss_scalar)
            val_losses_per_step_history.append(avg_epoch_val_loss_per_step)

            scheduler.step(avg_epoch_val_loss_scalar) # Scheduler uses loss from the final validation split
            if avg_epoch_val_loss_scalar < best_val_loss_scalar:
                best_val_loss_scalar = avg_epoch_val_loss_scalar
                best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: # No final validation set to use for per-epoch history
            val_losses_scalar_history.append(None)
            val_losses_per_step_history.append(np.full(output_window_steps, np.nan))
            # No scheduler.step() or best_model_state_dict update if no val_loader_final

        progress_percentage = (epoch + 1) / epochs
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = pytime.time() - epoch_start_time

        train_loss_per_step_str = " | ".join([f"{l:.4f}" for l in avg_epoch_train_loss_per_step]) # MODIFICA 4
        val_loss_scalar_str = f"{avg_epoch_val_loss_scalar:.6f}" if avg_epoch_val_loss_scalar is not None and avg_epoch_val_loss_scalar != float('inf') else "N/A (Final Split)"
        val_loss_per_step_str = "N/A (Final Split)"
        if val_loader_final and not np.all(np.isnan(avg_epoch_val_loss_per_step)):
             val_loss_per_step_str = " | ".join([f"{l:.4f}" for l in avg_epoch_val_loss_per_step])

        status_text.markdown(
            f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) | LR: {current_lr:.6f} | Loss Func: {loss_function_name}<br>'
            f'&nbsp;&nbsp;Train Loss (Scalar Avg): {avg_epoch_train_loss_scalar:.6f}<br>'
            f'&nbsp;&nbsp;Train Loss (Per Step Avg): [{train_loss_per_step_str}]<br>'
            f'&nbsp;&nbsp;Val Loss (Final Split - Scalar Avg): {val_loss_scalar_str}<br>'
            f'&nbsp;&nbsp;Val Loss (Final Split - Per Step Avg): [{val_loss_per_step_str}]',
            unsafe_allow_html=True
        )
        progress_bar.progress(progress_percentage, text=f"Training LSTM: Epoca {epoch+1}/{epochs}")
        update_loss_chart(train_losses_scalar_history, val_losses_scalar_history, loss_chart_placeholder)

    total_training_time = pytime.time() - start_training_time
    st.write(f"Training LSTM completato in {total_training_time:.1f} secondi.")

    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict: # This is based on the final validation split
            try:
                model.load_state_dict({k: v.to(device) for k, v in best_model_state_dict.items()})
                final_model_to_return = model # This is the model trained on the largest training set, chosen by the last val split
                st.success(f"Strategia 'migliore': Caricato modello LSTM con Val Loss (Scalare) minima ({best_val_loss_scalar:.6f}) su split finale.")
            except Exception as e_load_best:
                st.error(f"Errore caricamento stato LSTM migliore: {e_load_best}. Restituito modello ultima epoca.")
                final_model_to_return = model # Fallback to the model at the end of the last epoch
        elif val_loader_final: st.warning("Strategia 'migliore' LSTM: Nessun miglioramento Val Loss (Scalare) su split finale. Restituito modello ultima epoca.")
        else: st.warning("Strategia 'migliore' LSTM: Nessuna validazione su split finale (set vuoto). Restituito modello ultima epoca.")
    elif save_strategy == 'finale':
        final_model_to_return = model
        st.info("Strategia 'finale' LSTM: Restituito modello ultima epoca.")

    # Post-Training Cross-Validation Reporting
    if n_splits_cv > 1 and X_scaled_full.shape[0] >= n_splits_cv : # Ensure enough samples for CV
        st.markdown("--- \n**Valutazione Cross-Validation Post-Training del Modello Finale:**")
        cv_fold_val_losses_scalar_all = []
        cv_fold_val_losses_per_step_all = []

        # Use the already determined 'final_model_to_return' for evaluation on all folds
        final_model_to_return.eval()

        for i_fold, (train_idx_fold, val_idx_fold) in enumerate(all_splits_indices):
            X_val_fold, y_val_fold = X_scaled_full[val_idx_fold], y_scaled_full[val_idx_fold]
            if X_val_fold.size == 0 or y_val_fold.size == 0:
                st.caption(f"Fold CV {i_fold+1}/{n_splits_cv}: Set di validazione vuoto, saltato.")
                continue

            val_dataset_fold = TimeSeriesDataset(X_val_fold, y_val_fold)
            val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, num_workers=0)

            fold_loss_scalar_sum = 0.0
            fold_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for X_batch_f, y_batch_f in val_loader_fold:
                    X_batch_f, y_batch_f = X_batch_f.to(device), y_batch_f.to(device)
                    outputs_f = final_model_to_return(X_batch_f)
                    loss_elem_f = criterion(outputs_f, y_batch_f)
                    fold_loss_scalar_sum += loss_elem_f.mean().item() * X_batch_f.size(0)
                    fold_loss_per_step_sum += loss_elem_f.mean(dim=2).sum(dim=0).detach()

            avg_fold_loss_scalar = fold_loss_scalar_sum / len(val_dataset_fold)
            avg_fold_loss_per_step = (fold_loss_per_step_sum / len(val_dataset_fold)).cpu().numpy()

            cv_fold_val_losses_scalar_all.append(avg_fold_loss_scalar)
            cv_fold_val_losses_per_step_all.append(avg_fold_loss_per_step)
            st.caption(f"Fold CV {i_fold+1}/{n_splits_cv} - Val Loss (Scalar Avg): {avg_fold_loss_scalar:.6f} | Val Loss (Per Step Avg): [{ ' | '.join([f'{l:.4f}' for l in avg_fold_loss_per_step])}]")

        if cv_fold_val_losses_scalar_all:
            avg_cv_scalar_loss = np.mean(cv_fold_val_losses_scalar_all)
            avg_cv_per_step_loss = np.mean(np.array(cv_fold_val_losses_per_step_all), axis=0)
            st.success(f"**Media Validazione CV ({n_splits_cv} folds) - Loss Scalare: {avg_cv_scalar_loss:.6f}**")
            st.markdown(f"**Media Validazione CV ({n_splits_cv} folds) - Loss Per Step:** `{ {f'Step {s+1}': f'{l:.4f}' for s, l in enumerate(avg_cv_per_step_loss)} }`")
        else:
            st.warning("Nessun fold CV valutato con successo post-training.")
    else:
        st.info("Valutazione Cross-Validation Post-Training non eseguita (n_splits_cv <= 1 o dati insufficienti).")


    return final_model_to_return, \
           (train_losses_scalar_history, train_losses_per_step_history), \
           (val_losses_scalar_history, val_losses_per_step_history)


def train_model_seq2seq(X_enc_scaled_full, X_dec_scaled_full, y_tar_scaled_full, # CHANGED
                        encoder, decoder, output_window_steps, epochs=50, batch_size=32, learning_rate=0.001,
                        save_strategy='migliore', preferred_device='auto', teacher_forcing_ratio_schedule=None,
                        n_splits_cv=3, loss_function_name="MSELoss"): # NEW
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training Seq2Seq con TimeSeriesSplit (n_splits={n_splits_cv}, loss={loss_function_name})...")

    if output_window_steps <= 0:
        st.error("output_window_steps deve essere maggiore di 0 per train_model_seq2seq.")
        return None, ([], []), ([], [])
    if encoder.lstm.input_size <= 0 or decoder.forecast_input_size <= 0 or decoder.output_size <= 0:
        st.error(f"Errore: Parametri modello Seq2Seq non validi: Encoder_input_size={encoder.lstm.input_size}, Decoder_input_size={decoder.forecast_input_size}, Decoder_output_size={decoder.output_size}")
        return None, ([], []), ([], [])

    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"Training Seq2Seq userà: {device}")

    # Loss Function Selection
    if loss_function_name == "HuberLoss":
        criterion = nn.HuberLoss(reduction='none')
        print("Using HuberLoss for Seq2Seq")
    else: # Default to MSE
        criterion = nn.MSELoss(reduction='none')
        print("Using MSELoss for Seq2Seq")

    model = Seq2SeqHydro(encoder, decoder, output_window_steps, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # TimeSeriesSplit for final model training pass and per-epoch validation
    # Assuming X_enc_scaled_full is the primary dataset for splitting
    tscv_final_pass_s2s = TimeSeriesSplit(n_splits=n_splits_cv)
    all_splits_indices_s2s = list(tscv_final_pass_s2s.split(X_enc_scaled_full))

    if not all_splits_indices_s2s:
        st.error("TimeSeriesSplit (Seq2Seq) non ha prodotto alcun split.")
        return None, ([], []), ([], [])

    train_indices_final_s2s, val_indices_final_s2s = all_splits_indices_s2s[-1]

    X_enc_train_final = X_enc_scaled_full[train_indices_final_s2s]
    X_dec_train_final = X_dec_scaled_full[train_indices_final_s2s]
    y_tar_train_final = y_tar_scaled_full[train_indices_final_s2s]

    X_enc_val_final = X_enc_scaled_full[val_indices_final_s2s]
    X_dec_val_final = X_dec_scaled_full[val_indices_final_s2s]
    y_tar_val_final = y_tar_scaled_full[val_indices_final_s2s]

    if X_enc_train_final.size == 0 or y_tar_train_final.size == 0:
        st.error("Set di training finale Seq2Seq (dall'ultimo split CV) è vuoto.")
        return None, ([], []), ([], [])

    train_dataset_final_s2s = TimeSeriesDataset(X_enc_train_final, X_dec_train_final, y_tar_train_final)
    train_loader_final_s2s = DataLoader(train_dataset_final_s2s, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if device.type=='cuda' else False)

    val_loader_final_s2s = None
    if X_enc_val_final.size > 0 and y_tar_val_final.size > 0:
        val_dataset_final_s2s = TimeSeriesDataset(X_enc_val_final, X_dec_val_final, y_tar_val_final)
        val_loader_final_s2s = DataLoader(val_dataset_final_s2s, batch_size=batch_size, num_workers=0, pin_memory=True if device.type=='cuda' else False)
        print(f"Training finale Seq2Seq su {len(X_enc_train_final)} campioni, validazione finale su {len(X_enc_val_final)} campioni.")
    else:
        st.warning("Set di validazione finale Seq2Seq (dall'ultimo split CV) è vuoto.")
        print(f"Training finale Seq2Seq su {len(X_enc_train_final)} campioni. Nessun set di validazione finale.")

    train_losses_scalar_history, val_losses_scalar_history = [], [] # For final validation set
    train_losses_per_step_history, val_losses_per_step_history = [], [] # For final validation set
    best_val_loss_scalar = float('inf') # Based on final validation set
    best_model_state_dict = None
    progress_bar = st.progress(0.0, text="Training Seq2Seq: Inizio..."); status_text = st.empty(); loss_chart_placeholder = st.empty()

    def update_loss_chart_seq2seq(t_loss_scalar, v_loss_scalar, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss_scalar, mode='lines', name='Train Loss (Scalar Avg)'))
        valid_v_loss_scalar = [v for v in v_loss_scalar if v is not None] if v_loss_scalar else []
        if valid_v_loss_scalar: fig.add_trace(go.Scatter(y=valid_v_loss_scalar, mode='lines', name='Validation Loss (Scalar Avg)'))
        fig.update_layout(title='Andamento Loss (Seq2Seq - Media Scalare)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training Seq2Seq per {epochs} epoche su **{device}**..."); start_training_time = pytime.time()
    for epoch in range(epochs):
        epoch_start_time = pytime.time()
        model.train()
        epoch_train_loss_scalar_sum = 0.0
        epoch_train_loss_per_step_sum = torch.zeros(output_window_steps, device=device)

        current_tf_ratio = 0.5 # Default, can be adjusted by schedule
        if teacher_forcing_ratio_schedule and isinstance(teacher_forcing_ratio_schedule, list) and len(teacher_forcing_ratio_schedule) == 2:
            start_tf, end_tf = teacher_forcing_ratio_schedule
            if epochs > 1: current_tf_ratio = max(end_tf, start_tf - (start_tf - end_tf) * epoch / (epochs - 1))
            else: current_tf_ratio = start_tf

        # Training loop uses data from the final CV split
        for i, (x_enc_b, x_dec_b, y_tar_b) in enumerate(train_loader_final_s2s):
            x_enc_b, x_dec_b, y_tar_b = x_enc_b.to(device, non_blocking=True), x_dec_b.to(device, non_blocking=True), y_tar_b.to(device, non_blocking=True)
            optimizer.zero_grad()

            # NB: Il forward di Seq2SeqHydro DEVE essere in grado di gestire il teacher_forcing_ratio
            # Se il forward non lo gestisce internamente, il loop manuale è necessario qui.
            # Assumendo che il forward lo gestisca:
            # outputs_train_epoch = model(x_enc_b, x_dec_b, teacher_forcing_ratio=current_tf_ratio)
            # Se il forward NON gestisce TF, bisogna fare il loop qui:
            batch_s, out_win, target_size_dec = x_enc_b.shape[0], model.output_window, model.decoder.output_size
            outputs_train_epoch = torch.zeros(batch_s, out_win, target_size_dec).to(device)
            encoder_hidden, encoder_cell = model.encoder(x_enc_b)
            decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
            # Primo input al decoder (t=0) è sempre da x_dec_b
            decoder_input_step = x_dec_b[:, 0:1, :]

            for t in range(out_win):
                decoder_output_step, decoder_hidden, decoder_cell = model.decoder(
                    decoder_input_step, decoder_hidden, decoder_cell
                )
                outputs_train_epoch[:, t, :] = decoder_output_step

                # Teacher forcing per il *successivo* input al decoder
                if t < out_win - 1:
                    use_teacher_forcing = random.random() < current_tf_ratio
                    if use_teacher_forcing:
                        # Usa il target reale come prossimo input delle feature che il decoder dovrebbe prevedere
                        # Questo assume che y_tar_b[:, t+1:t+2, :] abbia la stessa forma e semantica di x_dec_b per le feature target
                        # Se x_dec_b contiene feature esogene, allora solo una parte di esso può essere sostituita
                        # L'implementazione corrente del decoder prende le feature da x_dec_b, quindi il TF "tradizionale" (reimmettere output)
                        # non è direttamente applicabile senza cambiare la logica del decoder o di x_dec_b.
                        # In questo contesto, si assume che x_dec_b sia il "ground truth" per gli input del decoder.
                        # Se le feature target sono anche in x_dec_b, allora il teacher forcing potrebbe significare usare y_tar_b
                        # per quelle specifiche feature. Per semplicità, continuiamo ad usare x_dec_b per gli input del decoder.
                        # La loss sarà comunque calcolata rispetto a y_tar_b.
                        # Il teacher forcing si manifesta qui nel fatto che il decoder vede sempre le feature future corrette (da x_dec_b)
                        # piuttosto che dover propagare i propri errori di previsione se le feature future fossero esse stesse predette.
                        decoder_input_step = x_dec_b[:, t+1:t+2, :] # Continua ad usare x_dec_b per le feature di input del decoder
                    else: # Usa l'output del decoder per le feature che il decoder dovrebbe predire, se applicabile.
                          # Ma x_dec_b contiene le feature per il decoder, non gli output.
                          # Quindi, anche senza TF, usiamo x_dec_b per il prossimo step.
                        decoder_input_step = x_dec_b[:, t+1:t+2, :]
            # Fine loop manuale per training

            loss_per_element_train = criterion(outputs_train_epoch, y_tar_b) # MODIFICA 3 (Seq2Seq)
            scalar_loss_for_backward_train = loss_per_element_train.mean()

            scalar_loss_for_backward_train.backward()
            optimizer.step()

            epoch_train_loss_scalar_sum += scalar_loss_for_backward_train.item() * x_enc_b.size(0)
            epoch_train_loss_per_step_sum += loss_per_element_train.mean(dim=2).sum(dim=0).detach()

        avg_epoch_train_loss_scalar = epoch_train_loss_scalar_sum / len(train_loader_final_s2s.dataset)
        train_losses_scalar_history.append(avg_epoch_train_loss_scalar)
        avg_epoch_train_loss_per_step = (epoch_train_loss_per_step_sum / len(train_loader_final_s2s.dataset)).cpu().numpy()
        train_losses_per_step_history.append(avg_epoch_train_loss_per_step)

        avg_epoch_val_loss_scalar = None # For the "final" validation set
        avg_epoch_val_loss_per_step = np.full(output_window_steps, np.nan) # For the "final" validation set

        if val_loader_final_s2s: # Use the val_loader for the final split
            model.eval()
            epoch_val_loss_scalar_sum = 0.0
            epoch_val_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for x_enc_vb, x_dec_vb, y_tar_vb in val_loader_final_s2s:
                    x_enc_vb, x_dec_vb, y_tar_vb = x_enc_vb.to(device, non_blocking=True), x_dec_vb.to(device, non_blocking=True), y_tar_vb.to(device, non_blocking=True)
                    outputs_val_epoch = model(x_enc_vb, x_dec_vb, teacher_forcing_ratio=0.0) # TF=0 for validation
                    loss_per_element_val = criterion(outputs_val_epoch, y_tar_vb)

                    scalar_loss_val_batch = loss_per_element_val.mean()
                    epoch_val_loss_scalar_sum += scalar_loss_val_batch.item() * x_enc_vb.size(0)
                    epoch_val_loss_per_step_sum += loss_per_element_val.mean(dim=2).sum(dim=0).detach()

            if len(val_loader_final_s2s.dataset) > 0:
                avg_epoch_val_loss_scalar = epoch_val_loss_scalar_sum / len(val_loader_final_s2s.dataset)
                avg_epoch_val_loss_per_step = (epoch_val_loss_per_step_sum / len(val_loader_final_s2s.dataset)).cpu().numpy()
            else: # Should not happen
                avg_epoch_val_loss_scalar = float('inf')

            val_losses_scalar_history.append(avg_epoch_val_loss_scalar)
            val_losses_per_step_history.append(avg_epoch_val_loss_per_step)
            scheduler.step(avg_epoch_val_loss_scalar) # Scheduler uses loss from the final validation split
            if avg_epoch_val_loss_scalar < best_val_loss_scalar:
                best_val_loss_scalar = avg_epoch_val_loss_scalar
                best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: # No final validation set
            val_losses_scalar_history.append(None)
            val_losses_per_step_history.append(np.full(output_window_steps, np.nan))
            # No scheduler.step() or best_model_state_dict update if no val_loader_final_s2s

        progress_percentage = (epoch + 1) / epochs
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = pytime.time() - epoch_start_time
        tf_ratio_str = f"{current_tf_ratio:.2f}" if teacher_forcing_ratio_schedule else "N/A"

        train_loss_per_step_str_s2s = " | ".join([f"{l:.4f}" for l in avg_epoch_train_loss_per_step]) # MODIFICA 4 (Seq2Seq)
        val_loss_scalar_str_s2s = f"{avg_epoch_val_loss_scalar:.6f}" if avg_epoch_val_loss_scalar is not None and avg_epoch_val_loss_scalar != float('inf') else "N/A (Final Split)"
        val_loss_per_step_str_s2s = "N/A (Final Split)"
        if val_loader_final_s2s and not np.all(np.isnan(avg_epoch_val_loss_per_step)):
             val_loss_per_step_str_s2s = " | ".join([f"{l:.4f}" for l in avg_epoch_val_loss_per_step])

        status_text.markdown(
            f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) | TF Ratio: {tf_ratio_str} | LR: {current_lr:.6f} | Loss Func: {loss_function_name}<br>'
            f'&nbsp;&nbsp;Train Loss (Scalar Avg): {avg_epoch_train_loss_scalar:.6f}<br>'
            f'&nbsp;&nbsp;Train Loss (Per Step Avg): [{train_loss_per_step_str_s2s}]<br>'
            f'&nbsp;&nbsp;Val Loss (Final Split - Scalar Avg): {val_loss_scalar_str_s2s}<br>'
            f'&nbsp;&nbsp;Val Loss (Final Split - Per Step Avg): [{val_loss_per_step_str_s2s}]',
            unsafe_allow_html=True
        )
        progress_bar.progress(progress_percentage, text=f"Training Seq2Seq: Epoca {epoch+1}/{epochs}")
        update_loss_chart_seq2seq(train_losses_scalar_history, val_losses_scalar_history, loss_chart_placeholder)

    total_training_time = pytime.time() - start_training_time
    st.write(f"Training Seq2Seq completato in {total_training_time:.1f} secondi.")
    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict: # Based on final validation split
            try:
                model.load_state_dict({k: v.to(device) for k, v in best_model_state_dict.items()})
                final_model_to_return = model
                st.success(f"Strategia 'migliore': Caricato modello Seq2Seq con Val Loss (Scalare) minima ({best_val_loss_scalar:.6f}) su split finale.")
            except Exception as e_load_best:
                st.error(f"Errore caricamento stato Seq2Seq migliore: {e_load_best}. Restituito modello ultima epoca.")
                final_model_to_return = model
        elif val_loader_final_s2s: st.warning("Strategia 'migliore' Seq2Seq: Nessun miglioramento Val Loss (Scalare) su split finale. Restituito modello ultima epoca.")
        else: st.warning("Strategia 'migliore' Seq2Seq: Nessuna validazione su split finale. Restituito modello ultima epoca.")
    elif save_strategy == 'finale':
        final_model_to_return = model
        st.info("Strategia 'finale' Seq2Seq: Restituito modello ultima epoca.")

    # Post-Training Cross-Validation Reporting for Seq2Seq
    if n_splits_cv > 1 and X_enc_scaled_full.shape[0] >= n_splits_cv:
        st.markdown("--- \n**Valutazione Cross-Validation Post-Training del Modello Finale (Seq2Seq):**")
        cv_fold_val_losses_scalar_s2s_all = []
        cv_fold_val_losses_per_step_s2s_all = []

        final_model_to_return.eval()

        for i_fold_s2s, (train_idx_f_s2s, val_idx_f_s2s) in enumerate(all_splits_indices_s2s):
            X_enc_val_f = X_enc_scaled_full[val_idx_f_s2s]
            X_dec_val_f = X_dec_scaled_full[val_idx_f_s2s]
            y_tar_val_f = y_tar_scaled_full[val_idx_f_s2s]

            if X_enc_val_f.size == 0 or y_tar_val_f.size == 0:
                st.caption(f"Fold CV Seq2Seq {i_fold_s2s+1}/{n_splits_cv}: Set di validazione vuoto, saltato.")
                continue

            val_dataset_f_s2s = TimeSeriesDataset(X_enc_val_f, X_dec_val_f, y_tar_val_f)
            val_loader_f_s2s = DataLoader(val_dataset_f_s2s, batch_size=batch_size, num_workers=0)

            fold_loss_scalar_sum_s2s = 0.0
            fold_loss_per_step_sum_s2s = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for x_ef, x_df, y_tf in val_loader_f_s2s:
                    x_ef, x_df, y_tf = x_ef.to(device), x_df.to(device), y_tf.to(device)
                    outputs_f_s2s = final_model_to_return(x_ef, x_df, teacher_forcing_ratio=0.0)
                    loss_elem_f_s2s = criterion(outputs_f_s2s, y_tf)
                    fold_loss_scalar_sum_s2s += loss_elem_f_s2s.mean().item() * x_ef.size(0)
                    fold_loss_per_step_sum_s2s += loss_elem_f_s2s.mean(dim=2).sum(dim=0).detach()

            avg_fold_loss_scalar_s2s = fold_loss_scalar_sum_s2s / len(val_dataset_f_s2s)
            avg_fold_loss_per_step_s2s = (fold_loss_per_step_sum_s2s / len(val_dataset_f_s2s)).cpu().numpy()

            cv_fold_val_losses_scalar_s2s_all.append(avg_fold_loss_scalar_s2s)
            cv_fold_val_losses_per_step_s2s_all.append(avg_fold_loss_per_step_s2s)
            st.caption(f"Fold CV Seq2Seq {i_fold_s2s+1}/{n_splits_cv} - Val Loss (Scalar Avg): {avg_fold_loss_scalar_s2s:.6f} | Val Loss (Per Step Avg): [{ ' | '.join([f'{l:.4f}' for l in avg_fold_loss_per_step_s2s])}]")

        if cv_fold_val_losses_scalar_s2s_all:
            avg_cv_scalar_loss_s2s = np.mean(cv_fold_val_losses_scalar_s2s_all)
            avg_cv_per_step_loss_s2s = np.mean(np.array(cv_fold_val_losses_per_step_s2s_all), axis=0)
            st.success(f"**Media Validazione CV Seq2Seq ({n_splits_cv} folds) - Loss Scalare: {avg_cv_scalar_loss_s2s:.6f}**")
            st.markdown(f"**Media Validazione CV Seq2Seq ({n_splits_cv} folds) - Loss Per Step:** `{ {f'Step {s+1}': f'{l:.4f}' for s, l in enumerate(avg_cv_per_step_loss_s2s)} }`")
        else:
            st.warning("Nessun fold CV Seq2Seq valutato con successo post-training.")
    else:
        st.info("Valutazione Cross-Validation Post-Training Seq2Seq non eseguita (n_splits_cv <= 1 o dati insufficienti).")

    return final_model_to_return, \
           (train_losses_scalar_history, train_losses_per_step_history), \
           (val_losses_scalar_history, val_losses_per_step_history)

# --- Funzioni Helper Download ---
def get_table_download_link(df, filename="data.csv", link_text="Scarica CSV"):
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        csv_buffer.seek(0); b64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8-sig')).decode()
        return f'<a href="data:text/csv;charset=utf-8-sig;base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.warning(f"Errore generazione link CSV: {e}"); return "<i>Errore link CSV</i>"

def get_binary_file_download_link(file_object, filename, text):
    try:
        file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode()
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None: mime_type = 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
    except Exception as e: st.warning(f"Errore generazione link binario per {filename}: {e}"); return f"<i>Errore link {filename}</i>"

def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    href_html = ""; href_png = ""
    safe_filename_base = re.sub(r'[^\w-]', '_', filename_base)
    try: # HTML
        html_filename = f"{safe_filename_base}.html"; buf_html = io.StringIO(); fig.write_html(buf_html, include_plotlyjs='cdn')
        buf_html.seek(0); b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="{html_filename}">{text_html}</a>'
    except Exception as e_html: print(f"Errore download HTML {safe_filename_base}: {e_html}"); href_html = "<i>Errore HTML</i>"
    try: # PNG
        import importlib
        if importlib.util.find_spec("kaleido"):
            png_filename = f"{safe_filename_base}.png"; buf_png = io.BytesIO(); fig.write_image(buf_png, format="png", scale=2)
            buf_png.seek(0); b64_png = base64.b64encode(buf_png.getvalue()).decode()
            href_png = f'<a href="data:image/png;base64,{b64_png}" download="{png_filename}">{text_png}</a>'
        else: print("Nota: 'kaleido' non installato, download PNG grafico disabilitato.")
    except Exception as e_png: print(f"Errore download PNG {safe_filename_base}: {e_png}"); href_png = "<i>Errore PNG</i>"
    return f"{href_html} {href_png}".strip()

def get_download_link_for_file(filepath, link_text=None):
    if not os.path.exists(filepath): return f"<i>File non trovato: {os.path.basename(filepath)}</i>"
    filename = os.path.basename(filepath); link_text = link_text or f"Scarica {filename}"
    try:
        with open(filepath, "rb") as f: file_content = f.read()
        b64 = base64.b64encode(file_content).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type is None: mime_type = 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.error(f"Errore generazione link per {filename}: {e}"); return f"<i>Errore link</i>"

# --- Funzione Estrazione ID GSheet ---
def extract_sheet_id(url):
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

# --- Funzione Estrazione Etichetta Stazione ---
def get_station_label(col_name, short=False):
    if col_name in STATION_COORDS:
        info = STATION_COORDS[col_name]; loc_id = info.get('location_id'); s_type = info.get('type', '')
        name = info.get('name', loc_id or col_name)
        if short and loc_id:
            sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == loc_id]
            if len(sensors_at_loc) > 1:
                # Per le nuove stazioni, potremmo voler abbreviare diversamente
                if 'Pioggia G.' in name: type_abbr = 'P(G)'
                elif 'Int. Pioggia' in name: type_abbr = 'P(I)'
                elif 'Raffica' in name: type_abbr = 'V(R)'
                elif 'Vel. Vento' in name: type_abbr = 'V(v)'
                else: type_abbr = {'Pioggia': 'P', 'Livello': 'L', 'Umidità': 'U'}.get(s_type, '?')
                label = f"{loc_id} {type_abbr}"
            else: label = loc_id
            label = re.sub(r'\s*\(.*?\)\s*', '', label).strip()
            return label[:20] + ('...' if len(label) > 20 else '')
        else: return name
    label = col_name
    label = re.sub(r'\s*\[.*?\]|\s*\(.*?\)', '', label).strip()
    label = label.replace('Sensore ', '').replace('Livello Idrometrico ', '').replace(' - Pioggia Ora', '').replace(' - Livello Misa', '').replace(' - Livello Nevola', '')
    parts = label.split(' '); label = ' '.join(parts[:2])
    return label[:20] + ('...' if len(label) > 20 else '') if short else label

# --- Inizializzazione Session State ---
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scalers' not in st.session_state: st.session_state.active_scalers = None
if 'df' not in st.session_state: st.session_state.df = None
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [
         'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
         'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
         HUMIDITY_COL_NAME,
         'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
         'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
         'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
         'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
         'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
     ]
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
    try:
        logo_path = "logo.png" # Assicurati che 'logo.png' sia nella stessa cartella o fornisci il percorso corretto
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.caption("Logo non trovato. Posiziona 'logo.png' nella cartella dell'app.")
    except Exception as e: st.warning(f"Impossibile caricare il logo: {e}")
    st.header('Impostazioni')

    st.subheader('Dati Storici (per Analisi/Training/Test)') # Modificato
    uploaded_data_file = st.file_uploader('Carica CSV Dati Storici', type=['csv'], key="data_uploader", help="File CSV con dati storici, separatore ';', decimale ','")
    df = None; df_load_error = None; data_source_info = ""
    data_path_to_load = None; is_uploaded = False
    if uploaded_data_file is not None: data_path_to_load = uploaded_data_file; is_uploaded = True; data_source_info = f"File caricato: **{uploaded_data_file.name}**"
    elif os.path.exists(DEFAULT_DATA_PATH):
        if 'df' not in st.session_state or st.session_state.df is None or st.session_state.get('uploaded_file_processed') is None or st.session_state.get('uploaded_file_processed') == False:
            data_path_to_load = DEFAULT_DATA_PATH; is_uploaded = False; data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
            st.session_state['uploaded_file_processed'] = False
        else: data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"; df = st.session_state.df
    elif 'df' not in st.session_state or st.session_state.df is None: df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato e nessun file caricato."

    process_file = False
    if is_uploaded and st.session_state.get('uploaded_file_processed') != uploaded_data_file.name: process_file = True
    elif data_path_to_load == DEFAULT_DATA_PATH and not st.session_state.get('uploaded_file_processed'): process_file = True

    if process_file and data_path_to_load:
        with st.spinner("Caricamento e processamento dati CSV..."):
            try:
                read_args = {'sep': ';', 'decimal': ',', 'low_memory': False}; encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']; df_temp = None
                file_obj = data_path_to_load
                for enc in encodings_to_try:
                     try:
                         if hasattr(file_obj, 'seek'): file_obj.seek(0)
                         df_temp = pd.read_csv(file_obj, encoding=enc, **read_args); break
                     except UnicodeDecodeError: continue
                     except Exception as read_e: raise read_e
                if df_temp is None: raise ValueError(f"Impossibile leggere CSV con encodings: {encodings_to_try}")
                date_col_csv = st.session_state.date_col_name_csv
                if date_col_csv not in df_temp.columns: raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante.")
                try: df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
                except ValueError:
                     try: df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], errors='coerce'); st.caption(f"Formato data CSV non standard ('{date_col_csv}'). Tentativo di inferenza.")
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
                     st.caption(f"Trovati {n_nan_before_fill} NaN/non numerici nel CSV. Eseguito ffill/bfill.")
                     df_temp.loc[:, numeric_cols] = df_temp[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                     n_nan_after_fill = df_temp[numeric_cols].isnull().sum().sum()
                     if n_nan_after_fill > 0: st.warning(f"NaN residui ({n_nan_after_fill}) dopo fill CSV. Riempiti con 0."); df_temp.loc[:, numeric_cols] = df_temp[numeric_cols].fillna(0)
                st.session_state.df = df_temp
                if is_uploaded: st.session_state['uploaded_file_processed'] = uploaded_data_file.name
                else: st.session_state['uploaded_file_processed'] = True
                st.success(f"Dati CSV caricati ({len(st.session_state.df)} righe)."); df = st.session_state.df
            except Exception as e: df = None; st.session_state.df = None; st.session_state['uploaded_file_processed'] = None; df_load_error = f'Errore caricamento/processamento CSV: {e}'; st.error(f"Errore CSV: {df_load_error}"); print(traceback.format_exc())

    if df is None and 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        if data_source_info == "":
            if st.session_state.get('uploaded_file_processed') and isinstance(st.session_state.get('uploaded_file_processed'), str): data_source_info = f"File caricato: **{st.session_state['uploaded_file_processed']}** (da sessione)"
            elif st.session_state.get('uploaded_file_processed') == True: data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"
    if data_source_info: st.caption(data_source_info)
    if df_load_error and not data_source_info: st.error(df_load_error)
    data_ready_csv = df is not None
    st.divider()

    st.subheader("Modello Predittivo (per Simulazione/Test)")
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
    selected_model_display_name = st.selectbox("Modello Pre-Addestrato:", selection_options, index=current_index, key="model_selector_predefined", help="Scegli un modello pre-addestrato.")

    with st.expander("Carica Modello LSTM Manualmente", expanded=(current_selection_name == MODEL_CHOICE_UPLOAD)):
         is_upload_active = False; m_f = st.file_uploader('File Modello (.pth)', type=['pth'], key="up_pth"); sf_f = st.file_uploader('File Scaler Features (.joblib)', type=['joblib'], key="up_scf_lstm"); st_f = st.file_uploader('File Scaler Target (.joblib)', type=['joblib'], key="up_sct_lstm")
         if m_f and sf_f and st_f:
            st.caption("Configura parametri del modello LSTM caricato:")
            col1_up, col2_up = st.columns(2)
            with col1_up: iw_up = st.number_input("Input Window (steps)", 1, 500, 48, 1, key="up_in_steps_lstm"); ow_up = st.number_input("Output Window (steps)", 1, 500, 24, 1, key="up_out_steps_lstm"); hs_up = st.number_input("Hidden Size", 8, 1024, 128, 8, key="up_hid_lstm")
            with col2_up: nl_up = st.number_input("Numero Layers", 1, 8, 2, 1, key="up_lay_lstm"); dr_up = st.slider("Dropout", 0.0, 0.9, 0.2, 0.05, key="up_drop_lstm")
            available_cols_for_upload = list(df.columns.drop(st.session_state.date_col_name_csv, errors='ignore')) if data_ready_csv else st.session_state.feature_columns
            default_targets_upload = [c for c in available_cols_for_upload if 'Livello' in c or '[m]' in c][:1]; default_features_upload = [c for c in st.session_state.feature_columns if c in available_cols_for_upload]
            targets_up = st.multiselect("Target Columns (Output)", available_cols_for_upload, default=default_targets_upload, key="up_targets_lstm"); features_up = st.multiselect("Feature Columns (Input)", available_cols_for_upload, default=default_features_upload, key="up_features_lstm")
            if targets_up and features_up:
                if st.button("Carica Modello Manuale", key="load_manual_lstm", type="secondary"): is_upload_active = True; selected_model_display_name = MODEL_CHOICE_UPLOAD
            else: st.caption("Seleziona features e target per il modello caricato.")
         else: st.caption("Carica tutti e tre i file per abilitare.")

    config_to_load = None; model_to_load = None; device_to_load = None; scalers_to_load = None; load_error_sidebar = False; model_type_loaded = None
    if is_upload_active:
        st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
        temp_cfg_up = {"input_window": iw_up, "output_window": ow_up, "hidden_size": hs_up, "num_layers": nl_up, "dropout": dr_up, "feature_columns": features_up, "target_columns": targets_up, "name": "uploaded_lstm_model", "display_name": "Modello LSTM Caricato", "model_type": "LSTM"}
        model_to_load, device_to_load = load_specific_model(m_f, temp_cfg_up)
        temp_model_info_up = {"scaler_features_path": sf_f, "scaler_targets_path": st_f}
        scalers_tuple_up = load_specific_scalers(temp_cfg_up, temp_model_info_up)
        if model_to_load and scalers_tuple_up and isinstance(scalers_tuple_up, tuple) and len(scalers_tuple_up)==2: config_to_load = temp_cfg_up; scalers_to_load = scalers_tuple_up; model_type_loaded = "LSTM"
        else: load_error_sidebar = True; st.error("Errore caricamento modello/scaler LSTM manuale.")
    elif selected_model_display_name != MODEL_CHOICE_NONE:
        if selected_model_display_name != st.session_state.get('active_model_name'):
            st.session_state.active_model_name = selected_model_display_name
            if selected_model_display_name in available_models_dict:
                model_info = available_models_dict[selected_model_display_name]; config_to_load = load_model_config(model_info["config_path"])
                if config_to_load:
                    config_to_load["pth_path"] = model_info["pth_path"]; config_to_load["config_name"] = model_info["config_name"]; config_to_load["display_name"] = selected_model_display_name
                    model_type_loaded = model_info.get("model_type", "LSTM"); config_to_load["model_type"] = model_type_loaded
                    model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
                    scalers_to_load = load_specific_scalers(config_to_load, model_info)
                    if not (model_to_load and scalers_to_load): load_error_sidebar = True; config_to_load = None
                else: load_error_sidebar = True
            else: st.error(f"Modello '{selected_model_display_name}' non trovato."); load_error_sidebar = True
        elif st.session_state.get('active_model') and st.session_state.get('active_config'):
             config_to_load = st.session_state.active_config; model_to_load = st.session_state.active_model; device_to_load = st.session_state.active_device
             scalers_to_load = st.session_state.active_scalers; model_type_loaded = config_to_load.get("model_type", "LSTM")
    elif selected_model_display_name == MODEL_CHOICE_NONE and st.session_state.get('active_model_name') != MODEL_CHOICE_NONE:
         st.session_state.active_model_name = MODEL_CHOICE_NONE; st.session_state.active_config = None; st.session_state.active_model = None; st.session_state.active_device = None; st.session_state.active_scalers = None
         config_to_load = None; model_to_load = None

    if config_to_load and model_to_load and device_to_load and scalers_to_load:
        scalers_valid = (isinstance(scalers_to_load, dict) and all(scalers_to_load.values())) or (isinstance(scalers_to_load, tuple) and all(s is not None for s in scalers_to_load))
        if scalers_valid: st.session_state.active_config = config_to_load; st.session_state.active_model = model_to_load; st.session_state.active_device = device_to_load; st.session_state.active_scalers = scalers_to_load
        else: st.error("Errore caricamento scaler."); load_error_sidebar = True; st.session_state.active_config = None; st.session_state.active_model = None; st.session_state.active_device = None; st.session_state.active_scalers = None

    active_config_sess = st.session_state.get('active_config'); active_model_sess = st.session_state.get('active_model'); active_model_name_sess = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)
    if active_model_sess and active_config_sess:
        cfg = active_config_sess; model_type_sess = cfg.get('model_type', 'LSTM'); display_feedback_name = cfg.get("display_name", active_model_name_sess)
        st.success(f"Modello Attivo: **{display_feedback_name}** ({model_type_sess})")
        if model_type_sess == "Seq2Seq": st.caption(f"Input: {cfg['input_window_steps']}s | Forecast: {cfg['forecast_window_steps']}s | Output: {cfg['output_window_steps']}s")
        else: st.caption(f"Input: {cfg['input_window']}s | Output: {cfg['output_window']}s")
    elif load_error_sidebar and active_model_name_sess not in [MODEL_CHOICE_NONE]: st.error(f"Caricamento modello '{active_model_name_sess}' fallito.")
    elif active_model_name_sess == MODEL_CHOICE_UPLOAD and not active_model_sess: st.info("Completa il caricamento manuale del modello LSTM.")
    elif active_model_name_sess == MODEL_CHOICE_NONE: st.info("Nessun modello selezionato per la simulazione/test.")
    st.divider()

    st.subheader("Configurazione Soglie Dashboard")
    with st.expander("Modifica Soglie di Allerta (per Dashboard)"):
        temp_thresholds = st.session_state.dashboard_thresholds.copy();
        # --- INIZIO MODIFICA: usa la lista combinata per le soglie UI ---
        monitorable_cols_thresh = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL];
        # --- FINE MODIFICA ---
        cols_thresh_ui = st.columns(2); col_idx_thresh = 0
        for col in monitorable_cols_thresh:
            with cols_thresh_ui[col_idx_thresh % 2]:
                label_short = get_station_label(col, short=False); is_level = 'Livello' in col or '(m)' in col or '(mt)' in col; is_humidity = 'Umidit' in col
                step = 0.1 if is_level else (1.0 if is_humidity else 0.5); fmt = "%.2f" if is_level else "%.1f"; min_v = 0.0
                current_threshold = st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, 0.0))
                new_threshold = st.number_input(label=f"{label_short}", value=float(current_threshold), min_value=min_v, step=step, format=fmt, key=f"thresh_{col}", help=f"Soglia di allerta per: {col}")
                if new_threshold != current_threshold: temp_thresholds[col] = new_threshold
            col_idx_thresh += 1
        if st.button("Salva Soglie", key="save_thresholds", type="secondary"): st.session_state.dashboard_thresholds = temp_thresholds.copy(); st.success("Soglie dashboard aggiornate!"); pytime.sleep(0.5); st.rerun()
    st.divider()

    st.header('Menu Navigazione')
    model_ready = active_model_sess is not None and active_config_sess is not None
    radio_options = ['Dashboard', 'Simulazione', 'Test Modello su Storico', 'Analisi Dati Storici', 'Allenamento Modello', 'Post-Training Modello'] # MODIFIED
    radio_captions = []; disabled_options = []; default_page_idx = 0
    for i, opt in enumerate(radio_options):
        caption = ""; disabled = False
        if opt == 'Dashboard': caption = "Monitoraggio GSheet"
        elif opt == 'Simulazione':
            if not model_ready: caption = "Richiede Modello attivo"; disabled = True
            else: caption = f"Esegui previsioni ({active_config_sess.get('model_type', 'LSTM')})"
        elif opt == 'Test Modello su Storico':
            if not model_ready: caption = "Richiede Modello attivo"; disabled = True
            elif not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
            else: caption = "Testa modello su CSV"
        elif opt == 'Analisi Dati Storici':
            if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
            else: caption = "Esplora dati CSV"
        elif opt == 'Allenamento Modello':
            if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
            else: caption = "Allena un nuovo modello"
        elif opt == 'Post-Training Modello': # NEW
            caption = "Affina un modello esistente con nuovi dati"
            if not model_ready: caption = "Richiede Modello attivo"; disabled = True
            elif not data_ready_csv: caption = "Richiede Dati CSV per post-training"; disabled = True
            # Ulteriore logica di disabilitazione specifica per Post-Training (es. tipo modello) potrebbe essere aggiunta qui
        radio_captions.append(caption); disabled_options.append(disabled)
        if opt == 'Dashboard': default_page_idx = i

    current_page = st.session_state.get('current_page', radio_options[default_page_idx])
    try:
        current_page_index = radio_options.index(current_page)
        if disabled_options[current_page_index]:
            st.warning(f"Pagina '{current_page}' non disponibile ({radio_captions[current_page_index]}). Reindirizzato a Dashboard.")
            current_page = radio_options[default_page_idx]; st.session_state['current_page'] = current_page
    except ValueError: current_page = radio_options[default_page_idx]; st.session_state['current_page'] = current_page

    chosen_page = st.radio('Scegli una funzionalità:', options=radio_options, captions=radio_captions, index=radio_options.index(current_page), key='page_selector_radio')
    if chosen_page != current_page:
        chosen_page_index = radio_options.index(chosen_page)
        if not disabled_options[chosen_page_index]: st.session_state['current_page'] = chosen_page; st.rerun()
        else: st.error(f"La pagina '{chosen_page}' non è disponibile. {radio_captions[chosen_page_index]}.")
    page = current_page

# ==============================================================================
# --- Logica Pagine Principali (Contenuto nel Main Frame) ---
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
    if "GOOGLE_CREDENTIALS" not in st.secrets: st.error("Errore Configurazione: Credenziali Google mancanti."); st.stop()
    st.checkbox("Visualizza intervallo di date personalizzato", key="dash_custom_range_check", value=st.session_state.dash_custom_range_check)
    df_dashboard = None; error_msg = None; actual_fetch_time = None; data_source_mode = ""
    if st.session_state.dash_custom_range_check:
        st.subheader("Seleziona Intervallo Temporale")
        col_date1, col_date2 = st.columns(2)
        with col_date1: start_date_select = st.date_input("Data Inizio", key="dash_start_date", value=st.session_state.dash_start_date)
        with col_date2: end_date_select = st.date_input("Data Fine", key="dash_end_date", value=st.session_state.dash_end_date)
        if start_date_select != st.session_state.dash_start_date: st.session_state.dash_start_date = start_date_select
        if end_date_select != st.session_state.dash_end_date: st.session_state.dash_end_date = end_date_select
        if start_date_select > end_date_select: st.error("La Data Fine deve essere uguale o successiva alla Data Inizio."); st.stop()
        start_dt_filter = italy_tz.localize(datetime.combine(start_date_select, time.min)); end_dt_filter = italy_tz.localize(datetime.combine(end_date_select, time.max))
        cache_time_key_full = int(pytime.time() // (DASHBOARD_REFRESH_INTERVAL_SECONDS * 2))
        # --- INIZIO MODIFICA: La chiamata non necessita più delle colonne, gestite internamente ---
        df_dashboard_full, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(cache_time_key_full, GSHEET_ID, fetch_all=True)
        # --- FINE MODIFICA ---
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
                else: st.error("Colonna data GSheet mancante per il filtraggio."); df_dashboard = None
            except Exception as e_filter: st.error(f"Errore durante il filtraggio dei dati per data: {e_filter}"); df_dashboard = None
        else:
            df_dashboard_full = st.session_state.get('full_gsheet_data_cache')
            if df_dashboard_full is not None:
                 st.warning("Recupero dati GSheet fallito, utilizzo dati precedentemente caricati.")
                 try:
                      if GSHEET_DATE_COL in df_dashboard_full.columns:
                         if df_dashboard_full[GSHEET_DATE_COL].dt.tz is None: df_dashboard_full[GSHEET_DATE_COL] = df_dashboard_full[GSHEET_DATE_COL].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                         else: df_dashboard_full[GSHEET_DATE_COL] = df_dashboard_full[GSHEET_DATE_COL].dt.tz_convert(italy_tz)
                         df_dashboard = df_dashboard_full[(df_dashboard_full[GSHEET_DATE_COL] >= start_dt_filter) & (df_dashboard_full[GSHEET_DATE_COL] <= end_dt_filter)].copy()
                         data_source_mode = f"Dati (da cache) da {start_date_select.strftime('%d/%m/%Y')} a {end_date_select.strftime('%d/%m/%Y')}"
                 except Exception: pass
            else: st.error(f"Recupero dati GSheet fallito: {error_msg}")
            st.session_state.last_dashboard_error = error_msg
    else:
        cache_time_key_recent = int(pytime.time() // DASHBOARD_REFRESH_INTERVAL_SECONDS)
        # --- INIZIO MODIFICA: La chiamata non necessita più delle colonne, gestite internamente ---
        df_dashboard, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(
            cache_time_key_recent, GSHEET_ID,
            fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS
        )
        # --- FINE MODIFICA ---
        st.session_state.last_dashboard_data = df_dashboard; st.session_state.last_dashboard_error = error_msg
        if df_dashboard is not None or error_msg is None: st.session_state.last_dashboard_fetch_time = actual_fetch_time
        data_source_mode = f"Ultime {DASHBOARD_HISTORY_ROWS // 2} ore circa"

    col_status, col_refresh_btn = st.columns([4, 1])
    with col_status:
        last_fetch_dt_sess = st.session_state.get('last_dashboard_fetch_time')
        if last_fetch_dt_sess:
             fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess; fetch_secs_ago = int(fetch_time_ago.total_seconds()) # Modificato in secondi
             status_text = f"{data_source_mode}. Aggiornato alle {last_fetch_dt_sess.strftime('%H:%M:%S')} ({fetch_secs_ago}s fa)."
             if not st.session_state.dash_custom_range_check: status_text += f" Refresh auto ogni {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
             st.caption(status_text)
        else: st.caption("Recupero dati GSheet in corso o fallito...")
    with col_refresh_btn:
        if st.button("Aggiorna Ora", key="dash_refresh_button"): fetch_gsheet_dashboard_data.clear(); fetch_sim_gsheet_data.clear(); st.session_state.full_gsheet_data_cache = None; st.success("Cache GSheet pulita. Ricaricamento..."); pytime.sleep(0.5); st.rerun()
    last_error_sess = st.session_state.get('last_dashboard_error')
    if last_error_sess:
        if any(x in last_error_sess for x in ["API", "Foglio", "Credenziali", "Errore:"]): st.error(f"Errore GSheet: {last_error_sess}")
        else: st.warning(f"Attenzione Dati GSheet: {last_error_sess}")

    if df_dashboard is not None and not df_dashboard.empty:
        if GSHEET_DATE_COL in df_dashboard.columns:
            latest_row_data = df_dashboard.iloc[-1]; last_update_time = latest_row_data.get(GSHEET_DATE_COL)
            if pd.notna(last_update_time) and isinstance(last_update_time, pd.Timestamp):
                 time_now_italy = datetime.now(italy_tz)
                 if last_update_time.tzinfo is None: last_update_time = italy_tz.localize(last_update_time)
                 else: last_update_time = last_update_time.tz_convert(italy_tz)
                 time_delta = time_now_italy - last_update_time; minutes_ago = int(time_delta.total_seconds() // 60)
                 time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')
                 if minutes_ago < 2: time_ago_str = "pochi istanti fa"
                 elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
                 else: time_ago_str = f"circa {minutes_ago // 60}h e {minutes_ago % 60}min fa"
                 st.markdown(f"**Ultimo rilevamento nel periodo visualizzato:** {time_str} ({time_ago_str})")
                 if not st.session_state.dash_custom_range_check and minutes_ago > 90: st.warning(f"Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa.")
            else: st.warning("Timestamp ultimo rilevamento (nel periodo) non valido o mancante.")
        else: st.warning("Colonna data GSheet non trovata per visualizzare l'ultimo rilevamento.")
        st.divider()
        st.subheader("Valori Ultimo Rilevamento, Stato Allerta e Portata")
        cols_to_monitor = [col for col in df_dashboard.columns if col != GSHEET_DATE_COL]
        table_rows = []; current_alerts = []
        for col_name in cols_to_monitor:
            current_value_H = latest_row_data.get(col_name) if 'latest_row_data' in locals() else None
            threshold = st.session_state.dashboard_thresholds.get(col_name); alert_active = False; value_numeric = np.nan; value_display = "N/D"; unit = ""
            unit_match = re.search(r'\((.*?)\)|\[(.*?)\]', col_name)
            if unit_match: unit_content = unit_match.group(1) or unit_match.group(2); unit = unit_content.strip() if unit_content else ""
            if pd.notna(current_value_H) and isinstance(current_value_H, (int, float, np.number)):
                 value_numeric = float(current_value_H); fmt_spec = ".1f" if unit in ['mm', '%', 'mm/hr', 'km/h'] else ".2f"
                 try: value_display = f"{value_numeric:{fmt_spec}} {unit}".strip()
                 except ValueError: value_display = f"{value_numeric} {unit}".strip()
                 if threshold is not None and isinstance(threshold, (int, float, np.number)) and value_numeric >= float(threshold): alert_active = True; current_alerts.append((col_name, value_numeric, threshold))
            elif pd.notna(current_value_H): value_display = f"{current_value_H} (?)"
            portata_Q = None; portata_display = "N/A"; sensor_info = STATION_COORDS.get(col_name)
            if sensor_info:
                sensor_code = sensor_info.get('sensor_code')
                if sensor_code and sensor_code in RATING_CURVES:
                    portata_Q = calculate_discharge(sensor_code, value_numeric)
                    if portata_Q is not None and pd.notna(portata_Q): portata_display = f"{portata_Q:.2f}"
                    else: portata_display = "-"
            status = "ALLERTA" if alert_active else ("OK" if pd.notna(value_numeric) else "N/D")
            threshold_display = f"{float(threshold):.1f}" if threshold is not None else "-"
            table_rows.append({"Stazione": get_station_label(col_name, short=True), "Valore": value_display, "Portata Q (m³/s)": portata_display, "Soglia": threshold_display, "Stato": status, "Valore Numerico H": value_numeric, "Soglia Numerica H": float(threshold) if threshold else None})
        df_display = pd.DataFrame(table_rows)
        def highlight_alert(row): style = [''] * len(row); style = ['background-color: rgba(255, 0, 0, 0.1);'] * len(row) if row['Stato'] == 'ALLERTA' else style; return style
        st.dataframe(df_display.style.apply(highlight_alert, axis=1), column_order=["Stazione", "Valore", "Portata Q (m³/s)", "Soglia", "Stato"], hide_index=True, use_container_width=True, column_config={"Valore Numerico H": None, "Soglia Numerica H": None})
        st.session_state.active_alerts = current_alerts
        st.divider()
        if GSHEET_DATE_COL in df_dashboard.columns:
            st.subheader("Grafico Storico Comparativo (Periodo Selezionato)")
            sensor_options_compare = {get_station_label(col, short=True): col for col in cols_to_monitor}
            default_selection_labels = [label for label, col in sensor_options_compare.items() if any(x in col for x in ['Livello', 'Raffica', 'Intensità'])][:5] or list(sensor_options_compare.keys())[:3]
            selected_labels_compare = st.multiselect("Seleziona sensori da confrontare:", options=list(sensor_options_compare.keys()), default=default_selection_labels, key="compare_select_multi")
            selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare] # Added for clarity
            if selected_labels_compare:
                fig_compare = go.Figure(); x_axis_data = df_dashboard[GSHEET_DATE_COL]
                for col in selected_cols_compare: fig_compare.add_trace(go.Scatter(x=x_axis_data, y=df_dashboard[col], mode='lines', name=get_station_label(col, short=True)))
                title_compare = f"Andamento Storico Comparato ({data_source_mode})"; fig_compare.update_layout(title=title_compare, xaxis_title='Data e Ora', yaxis_title='Valore Misurato', height=500, hovermode="x unified", template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_compare, use_container_width=True)
                compare_filename_base = f"compare_{'_'.join(sl.replace(' ','_') for sl in selected_labels_compare)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_compare, compare_filename_base), unsafe_allow_html=True)
            st.divider()
            st.subheader("Grafici Storici Individuali (Periodo Selezionato)")
            num_cols_individual = 3; graph_cols = st.columns(num_cols_individual); col_idx_graph = 0; x_axis_data_indiv = df_dashboard[GSHEET_DATE_COL]
            for col_name in cols_to_monitor:
                with graph_cols[col_idx_graph % num_cols_individual]:
                    threshold_individual = st.session_state.dashboard_thresholds.get(col_name); label_individual = get_station_label(col_name, short=True)
                    unit_match_indiv = re.search(r'\((.*?)\)|\[(.*?)\]', col_name); unit_indiv = ""
                    if unit_match_indiv: unit_content = unit_match_indiv.group(1) or unit_match_indiv.group(2); unit_indiv = f"({unit_content.strip()})" if unit_content else ""
                    yaxis_title_individual = f"Valore {unit_indiv}".strip(); fig_individual = go.Figure()
                    fig_individual.add_trace(go.Scatter(x=x_axis_data_indiv, y=df_dashboard[col_name], mode='lines', name=label_individual))
                    if threshold_individual is not None: fig_individual.add_hline(y=threshold_individual, line_dash="dash", line_color="red", annotation_text=f"Soglia ({float(threshold_individual):.1f})", annotation_position="bottom right")
                    fig_individual.update_layout(title=f"{label_individual}", xaxis_title=None, yaxis_title=yaxis_title_individual, height=300, hovermode="x unified", showlegend=False, margin=dict(t=40, b=30, l=50, r=10), template="plotly_white")
                    fig_individual.update_yaxes(rangemode='tozero'); st.plotly_chart(fig_individual, use_container_width=True)
                    ind_filename_base = f"sensor_{label_individual.replace(' ','_').replace('(','').replace(')','')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    st.markdown(get_plotly_download_link(fig_individual, ind_filename_base, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)
                col_idx_graph += 1
        else: st.info("Grafici storici non disponibili perché la colonna data GSheet non è stata trovata.")
        st.divider()
        st.subheader("Riepilogo Allerte (basato su ultimo valore nel periodo)")
        active_alerts_sess = st.session_state.get('active_alerts', [])
        if active_alerts_sess:
             st.warning("**Allerte Attive (Valori >= Soglia):**"); alert_md = ""
             sorted_alerts = sorted(active_alerts_sess, key=lambda x: get_station_label(x[0], short=False))
             for col, val, thr in sorted_alerts:
                 label_alert = get_station_label(col, short=False); sensor_type_alert = STATION_COORDS.get(col, {}).get('type', ''); type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                 unit_match_alert = re.search(r'\((.*?)\)|\[(.*?)\]', col); unit_alert = ""
                 if unit_match_alert: unit_content = unit_match_alert.group(1) or unit_match_alert.group(2); unit_alert = f" ({unit_content.strip()})" if unit_content else ""
                 val_fmt = f"{val:.1f}" if unit_alert in ['(mm)', '(%)', '(mm/hr)', '(km/h)'] else f"{val:.2f}"; thr_fmt = f"{float(thr):.1f}" if isinstance(thr, (int, float, np.number)) else str(thr)
                 alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit_alert}** >= Soglia **{thr_fmt}{unit_alert}**\n"
             st.markdown(alert_md)
        else: st.success("Nessuna soglia superata nell'ultimo rilevamento del periodo visualizzato.")
    elif df_dashboard is not None and df_dashboard.empty and not st.session_state.dash_custom_range_check: st.warning("Recupero dati GSheet riuscito, ma nessun dato trovato nelle ultime ore.")
    elif not st.session_state.last_dashboard_error: st.info("Recupero dati dashboard in corso...")

    if not st.session_state.dash_custom_range_check:
        try:
            component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"; js_code = f"(function() {{ const intervalIdKey = 'streamlit_auto_refresh_interval_id_{component_key}'; if (window[intervalIdKey]) {{ clearInterval(window[intervalIdKey]); }} window[intervalIdKey] = setInterval(function() {{ if (window.streamlitHook && typeof window.streamlitHook.rerunScript === 'function') {{ console.log('Triggering Streamlit rerun via JS timer ({component_key})'); window.streamlitHook.rerunScript(null); }} else {{ console.warn('Streamlit hook not available for JS auto-refresh.'); }} }}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000}); }})();"
            streamlit_js_eval(js_expressions=js_code, key=component_key, want_output=False)
        except Exception as e_js: st.caption(f"Nota: Impossibile impostare auto-refresh ({e_js})")
    else:
         try:
              component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"; js_clear_code = f"(function() {{ const intervalIdKey = 'streamlit_auto_refresh_interval_id_{component_key}'; if (window[intervalIdKey]) {{ console.log('Clearing JS auto-refresh timer ({component_key})'); clearInterval(window[intervalIdKey]); window[intervalIdKey] = null; }} }})();"
              streamlit_js_eval(js_expressions=js_clear_code, key=f"{component_key}_clear", want_output=False)
         except Exception as e_js_clear: st.caption(f"Nota: Impossibile pulire timer auto-refresh ({e_js_clear})")

# --- IL RESTO DEL CODICE RIMANE INVARIATO ---
# ... (incollare qui il resto del codice originale dalla pagina 'Simulazione' in poi)
# --- PAGINA SIMULAZIONE ---
elif page == 'Simulazione':
    # ... (codice originale)
# --- PAGINA TEST MODELLO SU STORICO ---
elif page == 'Test Modello su Storico':
    # ... (codice originale)
# --- PAGINA ANALISI DATI STORICI ---
elif page == 'Analisi Dati Storici':
    # ... (codice originale)
# --- PAGINA ALLENAMENTO MODELLO ---
elif page == 'Allenamento Modello':
    # ... (codice originale)
# --- PAGINA POST-TRAINING MODELLO ---
elif page == 'Post-Training Modello':
    # ... (codice originale)
# --- Footer ---
st.sidebar.divider()
st.sidebar.caption(f'Modello Predittivo Idrologico © {datetime.now().year}')
