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
from sklearn.model_selection import TimeSeriesSplit

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
    'Misa - Pioggia Ora (mm)', 'Umidita\' Sensore 3452 (Montemurello)',
    'Serra dei Conti - Livello Misa (mt)', 'Pianello di Ostra - Livello Misa (m)',
    'Nevola - Livello Nevola (mt)', 'Misa - Livello Misa (mt)',
    'Ponte Garibaldi - Livello Misa 2 (mt)'
]
DASHBOARD_REFRESH_INTERVAL_SECONDS = 300
DASHBOARD_HISTORY_ROWS = 48
DEFAULT_THRESHOLDS = {
    'Arcevia - Pioggia Ora (mm)': 2.0, 'Barbara - Pioggia Ora (mm)': 2.0, 'Corinaldo - Pioggia Ora (mm)': 2.0,
    'Misa - Pioggia Ora (mm)': 2.0, 'Umidita\' Sensore 3452 (Montemurello)': 95.0,
    'Serra dei Conti - Livello Misa (mt)': 1.7, 'Pianello di Ostra - Livello Misa (m)': 2.0,
    'Nevola - Livello Nevola (mt)': 2.5, 'Misa - Livello Misa (mt)': 2.0,
    'Ponte Garibaldi - Livello Misa 2 (mt)': 2.2
}
SIMULATION_THRESHOLDS = {
    'Serra dei Conti - Livello Misa (mt)': {'attenzione': 1.2, 'allerta': 1.7},
    'Pianello di Ostra - Livello Misa (m)': {'attenzione': 1.5, 'allerta': 2.0},
    'Nevola - Livello Nevola (mt)': {'attenzione': 2.0, 'allerta': 2.5},
    'Misa - Livello Misa (mt)': {'attenzione': 1.5, 'allerta': 2.0},
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'attenzione': 1.5, 'allerta': 2.2},
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)': {'attenzione': 1.2, 'allerta': 1.7},
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)': {'attenzione': 1.5, 'allerta': 2.0},
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)': {'attenzione': 2.0, 'allerta': 2.5},
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)': {'attenzione': 1.5, 'allerta': 2.0},
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'attenzione': 1.5, 'allerta': 2.2}
}
ATTRIBUTION_PHRASE = "Previsione progettata ed elaborata da Alberto Bussaglia del Comune di Senigallia. La previsione è di tipo probabilistico e non rappresenta una certezza.<br> Per maggiori informazioni, o per ottenere il consenso all'utilizzo/pubblicazione, contattare l'autore."
ATTRIBUTION_PHRASE_FILENAME_SUFFIX = "_da_Alberto_Bussaglia_Area_PC"
italy_tz = pytz.timezone('Europe/Rome')
HUMIDITY_COL_NAME = "Umidita' Sensore 3452 (Montemurello)"
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
STATION_COORDS = {
    'Arcevia - Pioggia Ora (mm)': {'lat': 43.5228, 'lon': 12.9388, 'name': 'Arcevia (Pioggia)', 'type': 'Pioggia', 'location_id': 'Arcevia'},
    'Barbara - Pioggia Ora (mm)': {'lat': 43.5808, 'lon': 13.0277, 'name': 'Barbara (Pioggia)', 'type': 'Pioggia', 'location_id': 'Barbara'},
    'Corinaldo - Pioggia Ora (mm)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Pioggia)', 'type': 'Pioggia', 'location_id': 'Corinaldo'},
    'Misa - Pioggia Ora (mm)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Pioggia)', 'type': 'Pioggia', 'location_id': 'Bettolelle'},
    HUMIDITY_COL_NAME: {'lat': 43.6, 'lon': 13.0, 'name': 'Montemurello (Umidità)', 'type': 'Umidità', 'location_id': 'Montemurello'},
    'Serra dei Conti - Livello Misa (mt)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti', 'sensor_code': '1008'},
    'Nevola - Livello Nevola (mt)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo', 'sensor_code': '1283'},
    'Misa - Livello Misa (mt)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle', 'sensor_code': '1112'},
    'Pianello di Ostra - Livello Misa (m)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'},
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'},
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti', 'sensor_code': '1008'},
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle', 'sensor_code': '1112'},
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo', 'sensor_code': '1283'},
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'},
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'}
}

# --- Funzione Calcolo Portata ---
def calculate_discharge(sensor_code, H):
    if sensor_code not in RATING_CURVES: return None
    if H is None or not isinstance(H, (int, float, np.number)) or np.isnan(H): return None
    try:
        H = float(H)
        rules = RATING_CURVES[sensor_code]
        for rule in rules:
            min_H, max_H = rule['min_H'], rule['max_H']
            if min_H <= H <= max_H:
                a, b, c, d = rule['a'], rule['b'], rule['c'], rule['d']
                base = H - b
                if base < 0: return float(d) if H >= min_H else np.nan
                Q = a * math.pow(base, c) + d
                return float(Q)
        return None
    except (ValueError, TypeError, OverflowError): return np.nan
calculate_discharge_vectorized = np.vectorize(calculate_discharge, otypes=[float])

# --- Definizioni Classi Modello ---
class TimeSeriesDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tuple(torch.tensor(t, dtype=torch.float32) for t in tensors)
    def __len__(self): return self.tensors[0].shape[0]
    def __getitem__(self, idx): return tuple(tensor[idx] for tensor in self.tensors)

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
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
        self.lstm = nn.LSTM(forecast_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
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
             forecast_window = self.output_window
        outputs = torch.zeros(batch_size, self.output_window, target_output_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :]
        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell = self.decoder(decoder_input_step, decoder_hidden, decoder_cell)
            outputs[:, t, :] = decoder_output_step
            if t < self.output_window - 1:
                if teacher_forcing_ratio > 0 and random.random() < teacher_forcing_ratio : pass
                else: decoder_input_step = x_future_forecast[:, t+1:t+2, :]
        return outputs

# --- Funzioni Utilità Modello/Dati (MODIFICATA prepare_training_data) ---
def prepare_training_data(df, initial_feature_columns, target_columns, input_window_hours, output_window_hours,
                          lag_config=None, cumulative_config=None,
                          existing_scaler_features=None, existing_scaler_targets=None):
    """
    Prepara i dati per il training LSTM standard.
    MODIFICATO:
    - Accetta `initial_feature_columns` (prima di FE).
    - Accetta `existing_scaler_features` e `existing_scaler_targets` opzionali. Se forniti, li usa per .transform().
    - Restituisce `current_feature_columns_used` (initial_feature_columns + nomi delle feature ingegnerizzate).
    """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati LSTM...")
    df_proc = df.copy() # Lavora su una copia per evitare modifiche in-place al df originale

    # --- Feature Engineering Start ---
    original_initial_feature_columns = list(initial_feature_columns) # Copia per sicurezza
    engineered_feature_names = []

    if lag_config:
        for col, lag_periods_hours in lag_config.items():
            if col in df_proc.columns:
                for lag_hr in lag_periods_hours:
                    lag_steps = lag_hr * 2
                    new_col_name = f"{col}_lag{lag_hr}h"
                    df_proc[new_col_name] = df_proc[col].shift(lag_steps)
                    if new_col_name not in engineered_feature_names : engineered_feature_names.append(new_col_name)
            else: st.warning(f"Colonna '{col}' (lag_config) non trovata in df_proc.")

    if cumulative_config:
        for col, window_periods_hours in cumulative_config.items():
            if col in df_proc.columns:
                for win_hr in window_periods_hours:
                    window_steps = win_hr * 2
                    new_col_name = f"{col}_cum{win_hr}h"
                    df_proc[new_col_name] = df_proc[col].rolling(window=window_steps, min_periods=1).sum()
                    if new_col_name not in engineered_feature_names : engineered_feature_names.append(new_col_name)
            else: st.warning(f"Colonna '{col}' (cumulative_config) non trovata in df_proc.")

    current_feature_columns_used = original_initial_feature_columns + engineered_feature_names
    
    cols_to_check_nan = list(set(current_feature_columns_used + target_columns))
    cols_present_in_df_to_fill = [c for c in cols_to_check_nan if c in df_proc.columns]

    if engineered_feature_names or df_proc[cols_present_in_df_to_fill].isnull().any().any():
        nan_count_before_fill = df_proc[cols_present_in_df_to_fill].isnull().sum().sum()
        if nan_count_before_fill > 0:
            df_proc[cols_present_in_df_to_fill] = df_proc[cols_present_in_df_to_fill].fillna(method='bfill').fillna(method='ffill')
            nan_count_after_fill = df_proc[cols_present_in_df_to_fill].isnull().sum().sum()
            st.caption(f"NaN (FE o preesistenti): {nan_count_before_fill} prima -> {nan_count_after_fill} dopo bfill/ffill.")
            if nan_count_after_fill > 0:
                st.warning(f"NaN residui ({nan_count_after_fill}). Riempiti con 0 per le colonne numeriche.")
                numeric_cols_to_zero_fill = df_proc[cols_present_in_df_to_fill].select_dtypes(include=np.number).columns
                df_proc[numeric_cols_to_zero_fill] = df_proc[numeric_cols_to_zero_fill].fillna(0)


    missing_current_features = [col for col in current_feature_columns_used if col not in df_proc.columns]
    missing_targets = [col for col in target_columns if col not in df_proc.columns]
    if missing_current_features:
        st.error(f"Feature (originali o ingegnerizzate) mancanti in df_proc: {missing_current_features}")
        return None, None, None, None, None
    if missing_targets:
        st.error(f"Target columns mancanti in df_proc: {missing_targets}")
        return None, None, None, None, None
    # --- Feature Engineering End ---

    X, y = [], []
    total_len = len(df_proc)
    input_steps = int(input_window_hours * 2)
    output_steps = int(output_window_hours * 2)
    required_len = input_steps + output_steps

    if total_len < required_len:
         st.error(f"Dati insufficienti ({total_len} righe) per creare sequenze (richieste {required_len}).")
         return None, None, None, None, None

    for i in range(total_len - required_len + 1):
        feature_window_data = df_proc.iloc[i : i + input_steps][current_feature_columns_used]
        target_window_data = df_proc.iloc[i + input_steps : i + required_len][target_columns]
        if feature_window_data.isnull().any().any() or target_window_data.isnull().any().any():
            continue
        X.append(feature_window_data.values)
        y.append(target_window_data.values)

    if not X or not y:
        st.error("Errore creazione sequenze X/y (zero sequenze valide)."); return None, None, None, None, None
    
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)
    if X.size == 0 or y.size == 0:
        st.error("Dati X o y vuoti prima di scaling."); return None, None, None, None, None

    num_sequences, seq_len_in, num_features_X = X.shape
    num_sequences_y, seq_len_out, num_targets_y = y.shape
    
    if num_features_X != len(current_feature_columns_used):
        st.error(f"Errore num feature X ({num_features_X}) vs colonne ({len(current_feature_columns_used)}).")
        return None, None, None, None, None
    if seq_len_in != input_steps or seq_len_out != output_steps:
        st.error(f"Errore shape sequenze: In={seq_len_in} (atteso {input_steps}), Out={seq_len_out} (atteso {output_steps})")
        return None, None, None, None, None

    X_flat = X.reshape(-1, num_features_X); y_flat = y.reshape(-1, num_targets_y)
    
    scaler_features_to_return = None
    scaler_targets_to_return = None

    try:
        if existing_scaler_features:
            X_scaled_flat = existing_scaler_features.transform(X_flat)
            scaler_features_to_return = existing_scaler_features
            print("Usato existing_scaler_features per .transform()")
        else:
            scaler_features_new = MinMaxScaler()
            X_scaled_flat = scaler_features_new.fit_transform(X_flat)
            scaler_features_to_return = scaler_features_new
            print("Creato e fittato nuovo scaler_features")

        if existing_scaler_targets:
            y_scaled_flat = existing_scaler_targets.transform(y_flat)
            scaler_targets_to_return = existing_scaler_targets
            print("Usato existing_scaler_targets per .transform()")
        else:
            scaler_targets_new = MinMaxScaler()
            y_scaled_flat = scaler_targets_new.fit_transform(y_flat)
            scaler_targets_to_return = scaler_targets_new
            print("Creato e fittato nuovo scaler_targets")

    except ValueError as ve_scale:
        st.error(f"Errore scaling: Input contiene NaN o shape errata. Dettagli: {ve_scale}")
        return None, None, None, None, None
    except Exception as e_scale:
        st.error(f"Errore scaling generico: {e_scale}"); return None, None, None, None, None

    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features_X)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets_y)

    print(f"Dati pronti: X_scaled shape={X_scaled.shape}, y_scaled shape={y_scaled.shape}. Features usate: {len(current_feature_columns_used)}")
    return X_scaled, y_scaled, scaler_features_to_return, scaler_targets_to_return, current_feature_columns_used


def prepare_training_data_seq2seq(df, initial_past_feature_cols, forecast_feature_cols, target_cols,
                                 input_window_steps, forecast_window_steps, output_window_steps,
                                 lag_config_past=None, cumulative_config_past=None,
                                 existing_scaler_past=None, existing_scaler_forecast=None, existing_scaler_targets=None):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq...")
    df_proc = df.copy()

    original_initial_past_cols = list(initial_past_feature_cols)
    engineered_past_feature_names = []

    if lag_config_past:
        for col, lag_periods_hours in lag_config_past.items():
            if col in df_proc.columns:
                for lag_hr in lag_periods_hours:
                    lag_steps = lag_hr * 2
                    new_col_name = f"{col}_lag{lag_hr}h"
                    df_proc[new_col_name] = df_proc[col].shift(lag_steps)
                    if new_col_name not in engineered_past_feature_names: engineered_past_feature_names.append(new_col_name)
            else: st.warning(f"Colonna '{col}' (lag_config_past) non trovata in df_proc.")
    
    if cumulative_config_past:
        for col, window_periods_hours in cumulative_config_past.items():
            if col in df_proc.columns:
                for win_hr in window_periods_hours:
                    window_steps = win_hr * 2
                    new_col_name = f"{col}_cum{win_hr}h"
                    df_proc[new_col_name] = df_proc[col].rolling(window=window_steps, min_periods=1).sum()
                    if new_col_name not in engineered_past_feature_names: engineered_past_feature_names.append(new_col_name)
            else: st.warning(f"Colonna '{col}' (cumulative_config_past) non trovata in df_proc.")

    current_past_feature_cols_used = original_initial_past_cols + engineered_past_feature_names
    
    cols_to_check_nan_s2s = list(set(current_past_feature_cols_used + forecast_feature_cols + target_cols))
    cols_present_in_df_to_fill_s2s = [c for c in cols_to_check_nan_s2s if c in df_proc.columns]
    
    if engineered_past_feature_names or df_proc[cols_present_in_df_to_fill_s2s].isnull().any().any():
        nan_sum_before_fill_s2s = df_proc[cols_present_in_df_to_fill_s2s].isnull().sum().sum()
        if nan_sum_before_fill_s2s > 0:
            df_proc[cols_present_in_df_to_fill_s2s] = df_proc[cols_present_in_df_to_fill_s2s].fillna(method='bfill').fillna(method='ffill')
            nan_sum_after_fill_s2s = df_proc[cols_present_in_df_to_fill_s2s].isnull().sum().sum()
            st.caption(f"NaN Seq2Seq (FE o preesistenti): {nan_sum_before_fill_s2s} prima -> {nan_sum_after_fill_s2s} dopo bfill/ffill.")
            if nan_sum_after_fill_s2s > 0:
                st.warning(f"NaN residui Seq2Seq ({nan_sum_after_fill_s2s}). Riempiti con 0 per le numeriche.")
                numeric_cols_to_zero_fill_s2s = df_proc[cols_present_in_df_to_fill_s2s].select_dtypes(include=np.number).columns
                df_proc[numeric_cols_to_zero_fill_s2s] = df_proc[numeric_cols_to_zero_fill_s2s].fillna(0)

    missing_curr_past_s2s = [c for c in current_past_feature_cols_used if c not in df_proc.columns]
    missing_fore_s2s = [c for c in forecast_feature_cols if c not in df_proc.columns]
    missing_targ_s2s = [c for c in target_cols if c not in df_proc.columns]
    if missing_curr_past_s2s: st.error(f"Feature passate (orig+FE) mancanti: {missing_curr_past_s2s}"); return None, None, None, None, None, None, None
    if missing_fore_s2s: st.error(f"Feature forecast mancanti: {missing_fore_s2s}"); return None, None, None, None, None, None, None
    if missing_targ_s2s: st.error(f"Target mancanti: {missing_targ_s2s}"); return None, None, None, None, None, None, None

    X_encoder, X_decoder, y_target = [], [], []
    total_len = len(df_proc)
    required_len = input_window_steps + max(forecast_window_steps, output_window_steps)

    if total_len < required_len:
        st.error(f"Dati Seq2Seq insuff. ({total_len}) per sequenze (req. {required_len}).")
        return None, None, None, None, None, None, None

    for i in range(total_len - required_len + 1):
        enc_end = i + input_window_steps
        past_feature_window_data = df_proc.iloc[i : enc_end][current_past_feature_cols_used]
        dec_start = enc_end
        dec_end_forecast = dec_start + forecast_window_steps
        forecast_feature_window_data = df_proc.iloc[dec_start : dec_end_forecast][forecast_feature_cols]
        target_end = dec_start + output_window_steps
        target_window_data = df_proc.iloc[dec_start : target_end][target_cols]

        if past_feature_window_data.isnull().any().any() or \
           forecast_feature_window_data.isnull().any().any() or \
           target_window_data.isnull().any().any():
            continue
        X_encoder.append(past_feature_window_data.values)
        X_decoder.append(forecast_feature_window_data.values)
        y_target.append(target_window_data.values)

    if not X_encoder or not X_decoder or not y_target:
        st.error("Errore creazione sequenze X_enc/X_dec/Y_target Seq2Seq."); return None, None, None, None, None, None, None

    X_encoder = np.array(X_encoder, dtype=np.float32)
    X_decoder = np.array(X_decoder, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)

    if X_encoder.shape[1] != input_window_steps or X_decoder.shape[1] != forecast_window_steps or y_target.shape[1] != output_window_steps:
         st.error("Errore shape sequenze Seq2Seq."); return None, None, None, None, None, None, None
    if X_encoder.shape[2] != len(current_past_feature_cols_used) or X_decoder.shape[2] != len(forecast_feature_cols) or y_target.shape[2] != len(target_cols):
         st.error(f"Errore num feature/target Seq2Seq. Encoder: {X_encoder.shape[2]} (atteso {len(current_past_feature_cols_used)}), Decoder: {X_decoder.shape[2]} (atteso {len(forecast_feature_cols)}), Target: {y_target.shape[2]} (atteso {len(target_cols)})")
         return None, None, None, None, None, None, None

    num_sequences = X_encoder.shape[0]
    X_enc_flat = X_encoder.reshape(-1, len(current_past_feature_cols_used))
    X_dec_flat = X_decoder.reshape(-1, len(forecast_feature_cols))
    y_tar_flat = y_target.reshape(-1, len(target_cols))

    scaler_past_final, scaler_forecast_final, scaler_targets_final = None, None, None
    try:
        if existing_scaler_past: X_enc_scaled_flat = existing_scaler_past.transform(X_enc_flat); scaler_past_final = existing_scaler_past
        else: sc_p_new = MinMaxScaler(); X_enc_scaled_flat = sc_p_new.fit_transform(X_enc_flat); scaler_past_final = sc_p_new
        
        if existing_scaler_forecast: X_dec_scaled_flat = existing_scaler_forecast.transform(X_dec_flat); scaler_forecast_final = existing_scaler_forecast
        else: sc_f_new = MinMaxScaler(); X_dec_scaled_flat = sc_f_new.fit_transform(X_dec_flat); scaler_forecast_final = sc_f_new

        if existing_scaler_targets: y_tar_scaled_flat = existing_scaler_targets.transform(y_tar_flat); scaler_targets_final = existing_scaler_targets
        else: sc_t_new = MinMaxScaler(); y_tar_scaled_flat = sc_t_new.fit_transform(y_tar_flat); scaler_targets_final = sc_t_new
            
    except ValueError as ve_scale_s2s:
        st.error(f"Errore scaling Seq2Seq (NaN o shape): {ve_scale_s2s}"); return None, None, None, None, None, None, None
    except Exception as e_scale_s2s:
        st.error(f"Errore scaling Seq2Seq generico: {e_scale_s2s}"); return None, None, None, None, None, None, None

    X_enc_scaled = X_enc_scaled_flat.reshape(num_sequences, input_window_steps, len(current_past_feature_cols_used))
    X_dec_scaled = X_dec_scaled_flat.reshape(num_sequences, forecast_window_steps, len(forecast_feature_cols))
    y_tar_scaled = y_tar_scaled_flat.reshape(num_sequences, output_window_steps, len(target_cols))

    print(f"Dati Seq2Seq pronti: X_enc={X_enc_scaled.shape}, X_dec={X_dec_scaled.shape}, y_tar={y_tar_scaled.shape}. Past features usate: {len(current_past_feature_cols_used)}")
    return (X_enc_scaled, X_dec_scaled, y_tar_scaled,
            scaler_past_final, scaler_forecast_final, scaler_targets_final, current_past_feature_cols_used)


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
            model_type = config_data.get("model_type", "LSTM") # Default to LSTM
            model_info["model_type"] = model_type # Store it

            if model_type == "Seq2Seq":
                # New keys: encoder_input_feature_columns, decoder_input_feature_columns
                # Old key: all_past_feature_columns, forecast_input_columns
                encoder_features_key = "encoder_input_feature_columns" if "encoder_input_feature_columns" in config_data else "all_past_feature_columns"
                decoder_features_key = "decoder_input_feature_columns" if "decoder_input_feature_columns" in config_data else "forecast_input_columns"

                required_keys = ["input_window_steps", "output_window_steps", "forecast_window_steps",
                                 "hidden_size", "num_layers", "dropout",
                                 encoder_features_key, decoder_features_key, "target_columns"]
                s_past_p = os.path.join(models_dir, f"{base}_past_features.joblib")
                s_fore_p = os.path.join(models_dir, f"{base}_forecast_features.joblib")
                s_targ_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(s_past_p) and os.path.exists(s_fore_p) and os.path.exists(s_targ_p)):
                    model_info.update({
                        "scaler_past_features_path": s_past_p,
                        "scaler_forecast_features_path": s_fore_p,
                        "scaler_targets_path": s_targ_p,
                    })
                    valid_model = True
            else: # LSTM Standard
                # New key: model_input_feature_columns
                # Old key: feature_columns
                features_key = "model_input_feature_columns" if "model_input_feature_columns" in config_data else "feature_columns"
                
                required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout",
                                 "target_columns", features_key]
                scf_p = os.path.join(models_dir, f"{base}_features.joblib")
                sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(scf_p) and os.path.exists(sct_p)):
                     model_info.update({
                         "scaler_features_path": scf_p,
                         "scaler_targets_path": sct_p,
                     })
                     valid_model = True
        except Exception as e_cfg:
            st.warning(f"Modello '{base}' ignorato: errore config JSON ({cfg_path}): {e_cfg}")
            valid_model = False
        if valid_model: available[name] = model_info
        else: print(f"Modello '{base}' ignorato: file/config mancanti per tipo '{model_type}'.")
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
    model_display_name = config.get('display_name', 'N/A')
    try:
        model = None
        if model_type == "Seq2Seq":
            encoder_features_key = "encoder_input_feature_columns" if "encoder_input_feature_columns" in config else "all_past_feature_columns"
            decoder_features_key = "decoder_input_feature_columns" if "decoder_input_feature_columns" in config else "forecast_input_columns"
            
            if encoder_features_key not in config or decoder_features_key not in config:
                raise ValueError(f"Chiavi feature ({encoder_features_key}, {decoder_features_key}) mancanti in config Seq2Seq '{model_display_name}'.")

            enc_input_size = len(config[encoder_features_key])
            dec_input_size = len(config[decoder_features_key])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]; layers = config["num_layers"]; drop = config["dropout"]
            out_win = config["output_window_steps"]
            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            decoder = DecoderLSTM(dec_input_size, hidden, dec_output_size, layers, drop).to(device)
            model = Seq2SeqHydro(encoder, decoder, out_win, device).to(device)
        else: # LSTM Standard
            features_key_lstm = "model_input_feature_columns" if "model_input_feature_columns" in config else "feature_columns"
            
            if features_key_lstm not in config:
                 raise ValueError(f"Chiave feature ('{features_key_lstm}') mancante in config LSTM '{model_display_name}'.")
            
            f_cols_lstm_effective = config[features_key_lstm]
            input_size_lstm = len(f_cols_lstm_effective)
            target_size_lstm = len(config["target_columns"])
            out_win_lstm_steps = config["output_window"] # This is in steps
            model = HydroLSTM(input_size_lstm, config["hidden_size"], target_size_lstm,
                              out_win_lstm_steps, config["num_layers"], config["dropout"]).to(device)

        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'):
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")
        model.eval()
        print(f"Modello '{model_display_name}' (Tipo: {model_type}) caricato su {device}.")
        return model, device
    except Exception as e:
        st.error(f"Errore caricamento modello '{model_display_name}' (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc()); return None, None

@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(config, model_info): # model_info contains paths
    if not config or not model_info: st.error("Config o info modello mancanti per caricare scaler."); return None
    model_type = config.get("model_type", "LSTM")
    def _load_joblib(path):
         if isinstance(path, str):
              if not os.path.exists(path): raise FileNotFoundError(f"File scaler '{path}' non trovato.")
              return joblib.load(path)
         elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
         else: raise TypeError(f"Percorso scaler non valido: {type(path)}")
    try:
        if model_type == "Seq2Seq":
            scaler_past = _load_joblib(model_info["scaler_past_features_path"])
            scaler_forecast = _load_joblib(model_info["scaler_forecast_features_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            return {"past": scaler_past, "forecast": scaler_forecast, "targets": scaler_targets}
        else: # LSTM Standard
            scaler_features = _load_joblib(model_info["scaler_features_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            return scaler_features, scaler_targets
    except Exception as e:
        st.error(f"Errore caricamento scaler (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc())
        return None if model_type == "Seq2Seq" else (None, None)

# --- Funzioni Predict ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict LSTM: Modello, scaler o config mancanti."); return None
    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM": st.error(f"Funzione predict LSTM errata per modello tipo: {model_type}"); return None
    
    # input_window è in steps
    input_steps_cfg = config["input_window"] 
    output_steps_cfg = config["output_window"] # in steps
    target_cols_cfg = config["target_columns"]
    
    # features_key_lstm = "model_input_feature_columns" if "model_input_feature_columns" in config else "feature_columns"
    # f_cols_effective_cfg = config.get(features_key_lstm, [])

    if input_data.shape[0] != input_steps_cfg: # input_data è (num_steps, num_features)
        st.error(f"Predict LSTM: Input righe {input_data.shape[0]} != Steps richiesti {input_steps_cfg}."); return None
    
    # expected_features = getattr(scaler_features, 'n_features_in_', len(f_cols_effective_cfg) if f_cols_effective_cfg else None)
    # Il numero di feature in input_data deve corrispondere a ciò che lo scaler si aspetta
    expected_features_from_scaler = getattr(scaler_features, 'n_features_in_', None)
    if expected_features_from_scaler is not None and input_data.shape[1] != expected_features_from_scaler:
         st.error(f"Predict LSTM: Input colonne {input_data.shape[1]} != Features attese da scaler_features {expected_features_from_scaler}."); return None
    
    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(inp_tens) # Output: (1, output_steps_cfg, num_targets)

        if output.shape != (1, output_steps_cfg, len(target_cols_cfg)):
             st.error(f"Predict LSTM: Shape output {output.shape} inattesa. Attesa: (1, {output_steps_cfg}, {len(target_cols_cfg)})"); return None
        out_np = output.cpu().numpy().squeeze(0)
        
        expected_targets_scaler = getattr(scaler_targets, 'n_features_in_', len(target_cols_cfg))
        if out_np.shape[1] != expected_targets_scaler:
            st.error(f"Predict LSTM: Output colonne mod. ({out_np.shape[1]}) != target scaler ({expected_targets_scaler}).")
            return None
        preds = scaler_targets.inverse_transform(out_np)
        return preds
    except Exception as e:
        st.error(f"Errore durante predict LSTM: {e}"); st.error(traceback.format_exc()); return None

def predict_seq2seq(model, past_data, future_forecast_data, scalers, config, device):
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]):
         st.error("Predict Seq2Seq: Input mancanti."); return None
    model_type = config.get("model_type")
    if model_type != "Seq2Seq": st.error(f"Funzione predict_seq2seq errata per modello tipo {model_type}"); return None

    input_steps_cfg = config["input_window_steps"]
    # forecast_steps_input_decoder_cfg = config["forecast_window_steps"]
    output_steps_model_cfg = config["output_window_steps"]
    
    encoder_features_key = "encoder_input_feature_columns" if "encoder_input_feature_columns" in config else "all_past_feature_columns"
    decoder_features_key = "decoder_input_feature_columns" if "decoder_input_feature_columns" in config else "forecast_input_columns"
    past_cols_cfg = config[encoder_features_key]
    forecast_cols_cfg = config[decoder_features_key]
    target_cols_cfg = config["target_columns"]

    scaler_past = scalers.get("past"); scaler_forecast = scalers.get("forecast"); scaler_targets = scalers.get("targets")
    if not all([scaler_past, scaler_forecast, scaler_targets]):
        st.error("Predict Seq2Seq: Scaler mancanti."); return None

    if past_data.shape != (input_steps_cfg, len(past_cols_cfg)):
        st.error(f"Predict Seq2Seq: Shape dati passati {past_data.shape} errata (attesa ({input_steps_cfg}, {len(past_cols_cfg)}))."); return None
    if future_forecast_data.shape[0] < output_steps_model_cfg:
         st.warning(f"Predict Seq2Seq: Finestra input forecast ({future_forecast_data.shape[0]}) < finestra output modello ({output_steps_model_cfg}).")
    if future_forecast_data.shape[1] != len(forecast_cols_cfg):
         st.error(f"Predict Seq2Seq: Num colonne dati forecast ({future_forecast_data.shape[1]}) errato (atteso {len(forecast_cols_cfg)})."); return None

    model.eval()
    try:
        past_norm = scaler_past.transform(past_data)
        future_norm = scaler_forecast.transform(future_forecast_data)
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device)
        future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)
        with torch.no_grad(): output = model(past_tens, future_tens, teacher_forcing_ratio=0.0)
        if output.shape != (1, output_steps_model_cfg, len(target_cols_cfg)):
             st.error(f"Predict Seq2Seq: Shape output {output.shape} inattesa (attesa (1, {output_steps_model_cfg}, {len(target_cols_cfg)}))."); return None
        out_np = output.cpu().numpy().squeeze(0)
        
        expected_targets_scaler = getattr(scaler_targets, 'n_features_in_', len(target_cols_cfg)) 
        if out_np.shape[1] != expected_targets_scaler:
             st.error(f"Predict Seq2Seq: Num colonne output mod. ({out_np.shape[1]}) != target scaler ({expected_targets_scaler})."); return None
        preds = scaler_targets.inverse_transform(out_np)
        return preds
    except Exception as e:
        st.error(f"Errore durante predict Seq2Seq: {e}"); st.error(traceback.format_exc()); return None

# --- Funzione Grafici Previsioni ---
def plot_predictions(predictions, config, start_time=None, actual_data=None, actual_data_label="Dati Reali CSV"):
    if config is None or predictions is None: return []
    model_type = config.get("model_type", "LSTM")
    target_cols = config["target_columns"]
    output_steps = predictions.shape[0]
    attribution_text = ATTRIBUTION_PHRASE; figs = []
    for i, sensor in enumerate(target_cols):
        fig = go.Figure();
        if start_time:
            time_steps_datetime = [start_time + timedelta(minutes=30 * (step)) for step in range(output_steps)]
            x_axis, x_title = time_steps_datetime, "Data e Ora"; x_tick_format = "%d/%m %H:%M"
        else:
            time_steps_relative = np.arange(1, output_steps + 1) * 0.5
            x_axis, x_title = time_steps_relative, "Ore Future (passi da 30 min)"; x_tick_format = None
        x_axis_np = np.array(x_axis)
        station_name_graph = get_station_label(sensor, short=False)
        plot_title_base = f'Test: Previsto vs Reale - {station_name_graph}' if actual_data is not None else f'Previsione {model_type} - {station_name_graph}'
        plot_title = f'{plot_title_base}<br><span style="font-size:10px;">{attribution_text}</span>'
        unit_match = re.search(r'\((.*?)\)|\[(.*?)\]', sensor); y_axis_unit = "m"; y_axis_title_h = "Livello H (m)"
        if unit_match: unit_content = unit_match.group(1) or unit_match.group(2); y_axis_unit = unit_content.strip() if unit_content else "m"; y_axis_title_h = f"Livello H ({y_axis_unit})"
        fig.add_trace(go.Scatter(x=x_axis_np, y=predictions[:, i], mode='lines+markers', name=f'Previsto H ({y_axis_unit})', yaxis='y1'))
        if actual_data is not None:
            if actual_data.ndim == 2 and actual_data.shape[0] == output_steps and actual_data.shape[1] == len(target_cols):
                fig.add_trace(go.Scatter(x=x_axis_np, y=actual_data[:, i], mode='lines', name=f'{actual_data_label} H ({y_axis_unit})', line=dict(color='green', dash='dashdot'), yaxis='y1'))
            else: st.warning(f"Shape 'actual_data' ({actual_data.shape}) non compatibile per {sensor}.")
        threshold_info = SIMULATION_THRESHOLDS.get(sensor, {}); soglia_attenzione = threshold_info.get('attenzione'); soglia_allerta = threshold_info.get('allerta')
        if soglia_attenzione is not None: fig.add_hline(y=soglia_attenzione, line_dash="dash", line_color="orange", annotation_text=f"Att.H({soglia_attenzione:.2f})", annotation_position="bottom right", layer="below")
        if soglia_allerta is not None: fig.add_hline(y=soglia_allerta, line_dash="dash", line_color="red", annotation_text=f"All.H({soglia_allerta:.2f})", annotation_position="top right", layer="below")
        sensor_info = STATION_COORDS.get(sensor); sensor_code_plot = sensor_info.get('sensor_code') if sensor_info else None; has_discharge_data = False
        if sensor_code_plot and sensor_code_plot in RATING_CURVES:
            predicted_H_values = predictions[:, i]; predicted_Q_values = calculate_discharge_vectorized(sensor_code_plot, predicted_H_values); valid_Q_mask = pd.notna(predicted_Q_values)
            if np.any(valid_Q_mask):
                has_discharge_data = True
                fig.add_trace(go.Scatter(x=x_axis_np[valid_Q_mask], y=predicted_Q_values[valid_Q_mask], mode='lines', name='Portata Prevista Q (m³/s)', line=dict(color='firebrick', dash='dot'), yaxis='y2'))
        try:
            fig.update_layout(title=plot_title, height=400, margin=dict(l=60, r=60, t=70, b=50), hovermode="x unified", template="plotly_white")
            fig.update_xaxes(title_text=x_title);
            if x_tick_format: fig.update_xaxes(tickformat=x_tick_format)
            fig.update_yaxes(title=dict(text=y_axis_title_h, font=dict(color="#1f77b4")), tickfont=dict(color="#1f77b4"), side="left", rangemode='tozero')
            if has_discharge_data: fig.update_layout(yaxis2=dict(title=dict(text="Portata Q (m³/s)", font=dict(color="firebrick")), tickfont=dict(color="firebrick"), overlaying="y", side="right", rangemode='tozero', showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
            elif actual_data is not None and actual_data.ndim == 2 : fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            else: fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1))
        except Exception as e_layout: st.error(f"Errore layout Plotly: {e_layout}"); print(traceback.format_exc())
        figs.append(fig)
    return figs

# --- Funzioni Fetch Dati Google Sheet ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format, fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS):
    mode = "tutti i dati" if fetch_all else f"ultime {num_rows_default} righe"; print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data ({mode})"); actual_fetch_time = datetime.now(italy_tz)
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", actual_fetch_time
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']); gc = gspread.authorize(credentials); sh = gc.open_by_key(sheet_id); worksheet = sh.sheet1; all_values = worksheet.get_all_values()
        if not all_values or len(all_values) < 2: return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time
        headers = all_values[0]; data_rows = all_values[1:] if fetch_all else all_values[max(1, len(all_values) - num_rows_default):]
        missing_cols = [col for col in relevant_columns if col not in headers];
        if missing_cols: return None, f"Errore: Colonne GSheet richieste mancanti: {', '.join(missing_cols)}", actual_fetch_time
        df = pd.DataFrame(data_rows, columns=headers); cols_to_select = [c for c in relevant_columns if c in df.columns]; df = df[cols_to_select]; error_parsing = []
        for col in df.columns:
            if col == date_col:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format if date_format else None, errors='coerce', infer_datetime_format=True)
                    if df[col].isnull().any(): error_parsing.append(f"Formato data non valido per '{col}'")
                    if not df[col].empty and df[col].notna().any(): df[col] = df[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward') if df[col].dt.tz is None else df[col].dt.tz_convert(italy_tz)
                except Exception as e_date: error_parsing.append(f"Errore data '{col}': {e_date}"); df[col] = pd.NaT
            else:
                try:
                    if col in df.columns: df_col_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip(); df[col] = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False); df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e_num: error_parsing.append(f"Errore numerico '{col}': {e_num}"); df[col] = np.nan
        if date_col in df.columns: df = df.sort_values(by=date_col, na_position='first').reset_index(drop=True)
        else: st.warning("Colonna data GSheet non trovata per ordinamento.")
        error_message = "Attenzione conversione dati GSheet: " + " | ".join(error_parsing) if error_parsing else None
        return df, error_message, actual_fetch_time
    except gspread.exceptions.APIError as api_e: error_msg = str(api_e); try: error_details = api_e.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}"; except: pass; return None, f"Errore API Google Sheets: {error_msg}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}').", actual_fetch_time
    except Exception as e: return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time

@st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet per simulazione...")
def fetch_sim_gsheet_data(sheet_id_fetch, n_rows_steps, date_col_gs, date_format_gs, col_mapping, required_model_cols_fetch, impute_dict):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_sim_gsheet_data (Steps: {n_rows_steps})"); actual_fetch_time = datetime.now(italy_tz); last_valid_timestamp = None
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", None
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']); gc = gspread.authorize(credentials); sh = gc.open_by_key(sheet_id_fetch); worksheet = sh.sheet1; all_data_gs = worksheet.get_all_values()
        if not all_data_gs or len(all_data_gs) < 2 : return None, f"Errore: Foglio GSheet vuoto o senza dati.", None
        headers_gs = all_data_gs[0]; data_rows_gs = all_data_gs[max(1, len(all_data_gs) - n_rows_steps):]; df_gsheet_raw = pd.DataFrame(data_rows_gs, columns=headers_gs)
        required_gsheet_cols_from_mapping = list(set(list(col_mapping.keys()) + ([date_col_gs] if date_col_gs not in col_mapping.keys() else [])))
        missing_gsheet_cols_in_sheet = [c for c in required_gsheet_cols_from_mapping if c not in df_gsheet_raw.columns]
        if missing_gsheet_cols_in_sheet: return None, f"Errore: Colonne GSheet ({', '.join(missing_gsheet_cols_in_sheet)}) mancanti nel foglio.", None
        cols_to_select_gsheet = [c for c in required_gsheet_cols_from_mapping if c in df_gsheet_raw.columns]; df_subset = df_gsheet_raw[cols_to_select_gsheet].copy(); df_mapped = df_subset.rename(columns=col_mapping)
        for model_col, impute_val in impute_dict.items():
             if model_col not in df_mapped.columns: df_mapped[model_col] = impute_val
        final_missing_model_cols = [c for c in required_model_cols_fetch if c not in df_mapped.columns]
        if final_missing_model_cols: return None, f"Errore: Colonne modello ({', '.join(final_missing_model_cols)}) mancanti dopo mappatura/imputazione.", None
        date_col_model_name = col_mapping.get(date_col_gs, date_col_gs);
        if date_col_model_name not in df_mapped.columns: date_col_model_name = None
        numeric_model_cols = [c for c in required_model_cols_fetch if c != date_col_model_name]
        for col in numeric_model_cols:
            if col not in df_mapped.columns: continue
            try:
                if pd.api.types.is_object_dtype(df_mapped[col]) or pd.api.types.is_string_dtype(df_mapped[col]): col_str = df_mapped[col].astype(str).str.replace(',', '.', regex=False).str.strip(); df_mapped[col] = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
            except Exception as e_clean_num: st.warning(f"Problema pulizia GSheet colonna '{col}': {e_clean_num}. Trattata come NaN."); df_mapped[col] = np.nan
        if date_col_model_name:
             try:
                 df_mapped[date_col_model_name] = pd.to_datetime(df_mapped[date_col_model_name], format=date_format_gs, errors='coerce', infer_datetime_format=True)
                 if df_mapped[date_col_model_name].isnull().any(): st.warning(f"Date non valide trovate in GSheet simulazione ('{date_col_model_name}').")
                 if not df_mapped[date_col_model_name].empty and df_mapped[date_col_model_name].notna().any():
                     df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward') if df_mapped[date_col_model_name].dt.tz is None else df_mapped[date_col_model_name].dt.tz_convert(italy_tz)
                 df_mapped = df_mapped.sort_values(by=date_col_model_name, na_position='first'); valid_dates = df_mapped[date_col_model_name].dropna()
                 if not valid_dates.empty: last_valid_timestamp = valid_dates.iloc[-1]
             except Exception as e_date_clean: st.warning(f"Errore data GSheet simulazione '{date_col_model_name}': {e_date_clean}."); date_col_model_name = None
        try: cols_present_final = [c for c in required_model_cols_fetch if c in df_mapped.columns]; df_final = df_mapped[cols_present_final].copy()
        except KeyError as e_key: return None, f"Errore selezione/ordine colonne finali simulazione: '{e_key}' mancante.", None
        numeric_cols_to_fill = df_final.select_dtypes(include=np.number).columns; nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
        if nan_count_before > 0:
             st.warning(f"Trovati {nan_count_before} valori NaN nei dati GSheet per simulazione. Applico ffill/bfill.")
             df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(method='ffill').fillna(method='bfill')
             if df_final[numeric_cols_to_fill].isnull().sum().sum() > 0: st.error(f"NaN residui ({df_final[numeric_cols_to_fill].isnull().sum().sum()}) dopo fillna. Tento fill con 0."); df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(0)
        if len(df_final) != n_rows_steps: st.warning(f"Attenzione: Num righe finali ({len(df_final)}) diverso da richiesto ({n_rows_steps}).")
        try: df_final = df_final[required_model_cols_fetch]
        except KeyError as e_final_order: missing_final = [c for c in required_model_cols_fetch if c not in df_final.columns]; return None, f"Errore: Colonne modello ({missing_final}) mancanti prima del return finale.", None
        return df_final, None, last_valid_timestamp
    except gspread.exceptions.APIError as api_e_sim: error_msg = str(api_e_sim); try: error_details = api_e_sim.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e_sim)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}"; except: pass; return None, f"Errore API Google Sheets: {error_msg}", None
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google simulazione non trovato (ID: '{sheet_id_fetch}').", None
    except Exception as e_sim_fetch: st.error(traceback.format_exc()); return None, f"Errore imprevisto importazione GSheet per simulazione: {type(e_sim_fetch).__name__} - {e_sim_fetch}", None

# --- Funzioni Allenamento (MODIFICATA train_model) ---
def train_model(X_scaled_full, y_scaled_full,
                input_size, output_size, output_window_steps,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2,
                save_strategy='migliore', preferred_device='auto', 
                n_splits_cv=3, loss_function_name="MSELoss",
                existing_model_to_finetune=None): # NUOVO PARAMETRO
    
    action_type = "Fine-tuning LSTM esistente" if existing_model_to_finetune else "Training LSTM Standard"
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio {action_type} con TimeSeriesSplit (n_splits={n_splits_cv}, loss={loss_function_name})...")
    
    if input_size <= 0 or output_size <= 0 or output_window_steps <= 0:
        st.error(f"Errore: Parametri modello LSTM non validi: input_size={input_size}, output_size={output_size}, output_window_steps={output_window_steps}")
        return None, ([], []), ([], [])

    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"{action_type} userà: {device}")

    criterion = nn.HuberLoss(reduction='none') if loss_function_name == "HuberLoss" else nn.MSELoss(reduction='none')
    print(f"Using {loss_function_name}")
    
    if existing_model_to_finetune:
        model = existing_model_to_finetune
        model.to(device) # Assicura che il modello sia sul dispositivo corretto
        # Verifica che i parametri del modello esistente corrispondano, se necessario
        if model.lstm.input_size != input_size:
            st.error(f"Errore Fine-tuning: Input size del modello esistente ({model.lstm.input_size}) non corrisponde a quello dei nuovi dati ({input_size}).")
            return None, ([], []), ([], [])
        if model.output_size != output_size or model.output_window != output_window_steps:
            st.error(f"Errore Fine-tuning: Output size/window del modello esistente ({model.output_size}/{model.output_window}) non corrisponde ai parametri ({output_size}/{output_window_steps}).")
            return None, ([], []), ([], [])
    else:
        model = HydroLSTM(input_size, hidden_size, output_size, output_window_steps, num_layers, dropout).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    tscv_final_pass = TimeSeriesSplit(n_splits=n_splits_cv)
    all_splits_indices = list(tscv_final_pass.split(X_scaled_full))
    if not all_splits_indices:
        st.error("TimeSeriesSplit non ha prodotto alcun split."); return None, ([], []), ([], [])
    
    train_indices_final, val_indices_final = all_splits_indices[-1]
    X_train_final, y_train_final = X_scaled_full[train_indices_final], y_scaled_full[train_indices_final]
    X_val_final, y_val_final = X_scaled_full[val_indices_final], y_scaled_full[val_indices_final]

    if X_train_final.size == 0 or y_train_final.size == 0:
        st.error("Set di training finale (da ultimo split CV) è vuoto."); return None, ([], []), ([], [])

    train_dataset_final = TimeSeriesDataset(X_train_final, y_train_final)
    train_loader_final = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == 'cuda')
    
    val_loader_final = None
    if X_val_final.size > 0 and y_val_final.size > 0:
        val_dataset_final = TimeSeriesDataset(X_val_final, y_val_final)
        val_loader_final = DataLoader(val_dataset_final, batch_size=batch_size, num_workers=0, pin_memory=device.type == 'cuda')
        print(f"{action_type}: training su {len(X_train_final)} campioni, validazione su {len(X_val_final)} (da ultimo split CV).")
    else:
        st.warning(f"{action_type}: Set validazione finale (da ultimo split CV) vuoto. Funzionalità limitate.")
        print(f"{action_type}: training su {len(X_train_final)} campioni. Nessun set validazione finale.")

    train_losses_scalar_history, val_losses_scalar_history = [], []
    train_losses_per_step_history, val_losses_per_step_history = [], []
    best_val_loss_scalar = float('inf'); best_model_state_dict = None
    progress_bar_text = f"{action_type}: Inizio..."
    progress_bar = st.progress(0.0, text=progress_bar_text); status_text = st.empty(); loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss_scalar, v_loss_scalar, placeholder, title_suffix=""):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss_scalar, mode='lines', name='Train Loss (Scalar Avg)'))
        valid_v_loss_scalar = [v for v in v_loss_scalar if v is not None] if v_loss_scalar else []
        if valid_v_loss_scalar: fig.add_trace(go.Scatter(y=valid_v_loss_scalar, mode='lines', name='Validation Loss (Scalar Avg)'))
        fig.update_layout(title=f'Andamento Loss ({title_suffix}Media Scalare)', xaxis_title='Epoca', yaxis_title='Loss', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio {action_type} per {epochs} epoche su **{device}**..."); start_training_time = pytime.time()
    for epoch in range(epochs):
        epoch_start_time = pytime.time(); model.train()
        epoch_train_loss_scalar_sum = 0.0; epoch_train_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
        for i, batch_data in enumerate(train_loader_final): 
            X_batch, y_batch = batch_data[0].to(device, non_blocking=True), batch_data[1].to(device, non_blocking=True)
            outputs = model(X_batch); loss_per_element = criterion(outputs, y_batch); scalar_loss_for_backward = loss_per_element.mean()
            optimizer.zero_grad(); scalar_loss_for_backward.backward(); optimizer.step()
            epoch_train_loss_scalar_sum += scalar_loss_for_backward.item() * X_batch.size(0)
            epoch_train_loss_per_step_sum += loss_per_element.mean(dim=2).sum(dim=0).detach()
        avg_epoch_train_loss_scalar = epoch_train_loss_scalar_sum / len(train_loader_final.dataset)
        train_losses_scalar_history.append(avg_epoch_train_loss_scalar)
        avg_epoch_train_loss_per_step = (epoch_train_loss_per_step_sum / len(train_loader_final.dataset)).cpu().numpy()
        train_losses_per_step_history.append(avg_epoch_train_loss_per_step)
        avg_epoch_val_loss_scalar = None; avg_epoch_val_loss_per_step = np.full(output_window_steps, np.nan)
        if val_loader_final:
            model.eval(); epoch_val_loss_scalar_sum = 0.0; epoch_val_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for batch_data_val in val_loader_final:
                    X_batch_val, y_batch_val = batch_data_val[0].to(device, non_blocking=True), batch_data_val[1].to(device, non_blocking=True)
                    outputs_val = model(X_batch_val); loss_per_element_val = criterion(outputs_val, y_batch_val)
                    scalar_loss_val_batch = loss_per_element_val.mean()
                    epoch_val_loss_scalar_sum += scalar_loss_val_batch.item() * X_batch_val.size(0)
                    epoch_val_loss_per_step_sum += loss_per_element_val.mean(dim=2).sum(dim=0).detach()
            if len(val_loader_final.dataset) > 0:
                avg_epoch_val_loss_scalar = epoch_val_loss_scalar_sum / len(val_loader_final.dataset)
                avg_epoch_val_loss_per_step = (epoch_val_loss_per_step_sum / len(val_loader_final.dataset)).cpu().numpy()
            else: avg_epoch_val_loss_scalar = float('inf')
            val_losses_scalar_history.append(avg_epoch_val_loss_scalar)
            val_losses_per_step_history.append(avg_epoch_val_loss_per_step)
            scheduler.step(avg_epoch_val_loss_scalar)
            if avg_epoch_val_loss_scalar < best_val_loss_scalar: best_val_loss_scalar = avg_epoch_val_loss_scalar; best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: val_losses_scalar_history.append(None); val_losses_per_step_history.append(np.full(output_window_steps, np.nan))
        
        progress_percentage = (epoch + 1) / epochs; current_lr = optimizer.param_groups[0]['lr']; epoch_time = pytime.time() - epoch_start_time
        train_loss_per_step_str = " | ".join([f"{l:.4f}" for l in avg_epoch_train_loss_per_step])
        val_loss_scalar_str = f"{avg_epoch_val_loss_scalar:.6f}" if avg_epoch_val_loss_scalar is not None and avg_epoch_val_loss_scalar != float('inf') else "N/A"
        val_loss_per_step_str = "N/A"
        if val_loader_final and not np.all(np.isnan(avg_epoch_val_loss_per_step)): val_loss_per_step_str = " | ".join([f"{l:.4f}" for l in avg_epoch_val_loss_per_step])
        status_text.markdown(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) | LR: {current_lr:.6f} | Loss: {loss_function_name}<br>'
                             f'&nbsp;&nbsp;Train Loss (Sc): {avg_epoch_train_loss_scalar:.6f} | Train Loss (Step): [{train_loss_per_step_str}]<br>'
                             f'&nbsp;&nbsp;Val Loss (Sc): {val_loss_scalar_str} | Val Loss (Step): [{val_loss_per_step_str}]', unsafe_allow_html=True)
        progress_bar.progress(progress_percentage, text=f"{action_type}: Epoca {epoch+1}/{epochs}")
        update_loss_chart(train_losses_scalar_history, val_losses_scalar_history, loss_chart_placeholder, title_suffix=f"{action_type.split()[0]} - ")

    total_training_time = pytime.time() - start_training_time; st.write(f"{action_type} completato in {total_training_time:.1f} secondi.")
    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict:
            try: model.load_state_dict({k: v.to(device) for k, v in best_model_state_dict.items()}); final_model_to_return = model; st.success(f"Strategia 'migliore': Caricato modello con Val Loss minima ({best_val_loss_scalar:.6f}).")
            except Exception as e_load_best: st.error(f"Errore caricamento stato migliore: {e_load_best}. Restituito modello ultima epoca."); final_model_to_return = model
        elif val_loader_final: st.warning("Strategia 'migliore': Nessun miglioramento Val Loss. Restituito modello ultima epoca.")
        else: st.warning("Strategia 'migliore': Nessuna validazione. Restituito modello ultima epoca.")
    elif save_strategy == 'finale': final_model_to_return = model; st.info("Strategia 'finale': Restituito modello ultima epoca.")
    
    if n_splits_cv > 1 and X_scaled_full.shape[0] >= n_splits_cv :
        st.markdown(f"--- \n**Valutazione CV Post-{action_type} del Modello Finale:**")
        cv_fold_val_losses_scalar_all = []; cv_fold_val_losses_per_step_all = []
        final_model_to_return.eval()
        for i_fold, (train_idx_fold, val_idx_fold) in enumerate(all_splits_indices):
            X_val_fold, y_val_fold = X_scaled_full[val_idx_fold], y_scaled_full[val_idx_fold]
            if X_val_fold.size == 0 or y_val_fold.size == 0: st.caption(f"Fold CV {i_fold+1}/{n_splits_cv}: Set validazione vuoto."); continue
            val_dataset_fold = TimeSeriesDataset(X_val_fold, y_val_fold); val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, num_workers=0)
            fold_loss_scalar_sum = 0.0; fold_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for X_batch_f, y_batch_f in val_loader_fold:
                    X_batch_f, y_batch_f = X_batch_f.to(device), y_batch_f.to(device)
                    outputs_f = final_model_to_return(X_batch_f); loss_elem_f = criterion(outputs_f, y_batch_f)
                    fold_loss_scalar_sum += loss_elem_f.mean().item() * X_batch_f.size(0)
                    fold_loss_per_step_sum += loss_elem_f.mean(dim=2).sum(dim=0).detach()
            avg_fold_loss_scalar = fold_loss_scalar_sum / len(val_dataset_fold); avg_fold_loss_per_step = (fold_loss_per_step_sum / len(val_dataset_fold)).cpu().numpy()
            cv_fold_val_losses_scalar_all.append(avg_fold_loss_scalar); cv_fold_val_losses_per_step_all.append(avg_fold_loss_per_step)
            st.caption(f"Fold CV {i_fold+1}/{n_splits_cv} - Val Loss (Sc): {avg_fold_loss_scalar:.6f} | Val Loss (Step): [{ ' | '.join([f'{l:.4f}' for l in avg_fold_loss_per_step])}]")
        if cv_fold_val_losses_scalar_all:
            avg_cv_scalar_loss = np.mean(cv_fold_val_losses_scalar_all); avg_cv_per_step_loss = np.mean(np.array(cv_fold_val_losses_per_step_all), axis=0)
            st.success(f"**Media Validazione CV ({n_splits_cv} folds) - Loss Scalare: {avg_cv_scalar_loss:.6f}**")
            st.markdown(f"**Media Validazione CV ({n_splits_cv} folds) - Loss Per Step:** `{ {f'Step {s+1}': f'{l:.4f}' for s, l in enumerate(avg_cv_per_step_loss)} }`")
        else: st.warning("Nessun fold CV valutato con successo post-training.")
    else: st.info(f"Valutazione CV Post-{action_type} non eseguita.")
    return final_model_to_return, (train_losses_scalar_history, train_losses_per_step_history), (val_losses_scalar_history, val_losses_per_step_history) 

def train_model_seq2seq(X_enc_scaled_full, X_dec_scaled_full, y_tar_scaled_full,
                        encoder, decoder, output_window_steps, epochs=50, batch_size=32, learning_rate=0.001,
                        save_strategy='migliore', preferred_device='auto', teacher_forcing_ratio_schedule=None,
                        n_splits_cv=3, loss_function_name="MSELoss",
                        existing_model_to_finetune=None): # NUOVO PARAMETRO
    action_type = "Fine-tuning Seq2Seq esistente" if existing_model_to_finetune else "Training Seq2Seq Standard"
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio {action_type} con TimeSeriesSplit (n_splits={n_splits_cv}, loss={loss_function_name})...")
    if output_window_steps <= 0: st.error("output_window_steps deve essere > 0."); return None, ([], []), ([], [])

    enc_input_size_data = X_enc_scaled_full.shape[2]
    dec_input_size_data = X_dec_scaled_full.shape[2]
    dec_output_size_data = y_tar_scaled_full.shape[2]

    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"{action_type} userà: {device}")

    criterion = nn.HuberLoss(reduction='none') if loss_function_name == "HuberLoss" else nn.MSELoss(reduction='none')
    print(f"Using {loss_function_name} for Seq2Seq")

    if existing_model_to_finetune:
        model = existing_model_to_finetune
        model.to(device)
        # Verifica parametri modello esistente
        if model.encoder.lstm.input_size != enc_input_size_data:
            st.error(f"Fine-tuning Seq2Seq: Encoder input size modello ({model.encoder.lstm.input_size}) != dati ({enc_input_size_data}).")
            return None, ([], []), ([], [])
        if model.decoder.forecast_input_size != dec_input_size_data:
            st.error(f"Fine-tuning Seq2Seq: Decoder input size modello ({model.decoder.forecast_input_size}) != dati ({dec_input_size_data}).")
            return None, ([], []), ([], [])
        if model.decoder.output_size != dec_output_size_data or model.output_window != output_window_steps:
             st.error(f"Fine-tuning Seq2Seq: Decoder output/window modello ({model.decoder.output_size}/{model.output_window}) != parametri ({dec_output_size_data}/{output_window_steps}).")
             return None, ([], []), ([], [])
    else:
        # Se encoder e decoder sono passati, usali, altrimenti errore (o crea nuovi qui)
        if encoder is None or decoder is None:
            st.error("Per training Seq2Seq da zero, encoder e decoder devono essere forniti (o creati).")
            return None, ([], []), ([], [])
        # Assicurati che i moduli encoder/decoder passati abbiano le dimensioni corrette per i dati
        if encoder.lstm.input_size != enc_input_size_data:
            st.error(f"Training Seq2Seq: Encoder input size fornito ({encoder.lstm.input_size}) != dati ({enc_input_size_data}).")
            return None, ([], []), ([], [])
        if decoder.forecast_input_size != dec_input_size_data:
            st.error(f"Training Seq2Seq: Decoder input size fornito ({decoder.forecast_input_size}) != dati ({dec_input_size_data}).")
            return None, ([], []), ([], [])
        if decoder.output_size != dec_output_size_data:
             st.error(f"Training Seq2Seq: Decoder output size fornito ({decoder.output_size}) != dati ({dec_output_size_data}).")
             return None, ([], []), ([], [])
        model = Seq2SeqHydro(encoder, decoder, output_window_steps, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    tscv_final_pass_s2s = TimeSeriesSplit(n_splits=n_splits_cv)
    all_splits_indices_s2s = list(tscv_final_pass_s2s.split(X_enc_scaled_full))
    if not all_splits_indices_s2s: st.error("TimeSeriesSplit (Seq2Seq) non ha prodotto alcun split."); return None, ([], []), ([], [])
    train_indices_final_s2s, val_indices_final_s2s = all_splits_indices_s2s[-1]
    X_enc_train_final, X_dec_train_final, y_tar_train_final = X_enc_scaled_full[train_indices_final_s2s], X_dec_scaled_full[train_indices_final_s2s], y_tar_scaled_full[train_indices_final_s2s]
    X_enc_val_final, X_dec_val_final, y_tar_val_final = X_enc_scaled_full[val_indices_final_s2s], X_dec_scaled_full[val_indices_final_s2s], y_tar_scaled_full[val_indices_final_s2s]
    if X_enc_train_final.size == 0 or y_tar_train_final.size == 0: st.error("Set di training finale Seq2Seq (da ultimo split CV) è vuoto."); return None, ([], []), ([], [])
    train_dataset_final_s2s = TimeSeriesDataset(X_enc_train_final, X_dec_train_final, y_tar_train_final)
    train_loader_final_s2s = DataLoader(train_dataset_final_s2s, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type=='cuda')
    val_loader_final_s2s = None
    if X_enc_val_final.size > 0 and y_tar_val_final.size > 0:
        val_dataset_final_s2s = TimeSeriesDataset(X_enc_val_final, X_dec_val_final, y_tar_val_final)
        val_loader_final_s2s = DataLoader(val_dataset_final_s2s, batch_size=batch_size, num_workers=0, pin_memory=device.type=='cuda')
        print(f"{action_type} Seq2Seq: training su {len(X_enc_train_final)} campioni, validazione su {len(X_enc_val_final)}.")
    else: st.warning(f"{action_type} Seq2Seq: Set validazione finale (da ultimo split CV) vuoto."); print(f"{action_type} Seq2Seq: training su {len(X_enc_train_final)} campioni. Nessun set validazione finale.")
    
    train_losses_scalar_history, val_losses_scalar_history = [], []; train_losses_per_step_history, val_losses_per_step_history = [], []
    best_val_loss_scalar = float('inf'); best_model_state_dict = None
    progress_bar_text_s2s = f"{action_type} Seq2Seq: Inizio..."
    progress_bar = st.progress(0.0, text=progress_bar_text_s2s); status_text = st.empty(); loss_chart_placeholder = st.empty()

    def update_loss_chart_seq2seq(t_loss_scalar, v_loss_scalar, placeholder, title_suffix=""):
        fig = go.Figure(); fig.add_trace(go.Scatter(y=t_loss_scalar, mode='lines', name='Train Loss (Scalar Avg)'))
        valid_v_loss_scalar = [v for v in v_loss_scalar if v is not None] if v_loss_scalar else []
        if valid_v_loss_scalar: fig.add_trace(go.Scatter(y=valid_v_loss_scalar, mode='lines', name='Validation Loss (Scalar Avg)'))
        fig.update_layout(title=f'Andamento Loss ({title_suffix}Media Scalare)', xaxis_title='Epoca', yaxis_title='Loss', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio {action_type} Seq2Seq per {epochs} epoche su **{device}**..."); start_training_time = pytime.time()
    for epoch in range(epochs):
        epoch_start_time = pytime.time(); model.train()
        epoch_train_loss_scalar_sum = 0.0; epoch_train_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
        current_tf_ratio = 0.5 
        if teacher_forcing_ratio_schedule and isinstance(teacher_forcing_ratio_schedule, list) and len(teacher_forcing_ratio_schedule) == 2:
            start_tf, end_tf = teacher_forcing_ratio_schedule
            if epochs > 1: current_tf_ratio = max(end_tf, start_tf - (start_tf - end_tf) * epoch / (epochs - 1))
            else: current_tf_ratio = start_tf
        for i, (x_enc_b, x_dec_b, y_tar_b) in enumerate(train_loader_final_s2s):
            x_enc_b, x_dec_b, y_tar_b = x_enc_b.to(device, non_blocking=True), x_dec_b.to(device, non_blocking=True), y_tar_b.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs_train_epoch = model(x_enc_b, x_dec_b, teacher_forcing_ratio=current_tf_ratio) # Modello Seq2SeqHydro gestisce TF nel forward
            loss_per_element_train = criterion(outputs_train_epoch, y_tar_b); scalar_loss_for_backward_train = loss_per_element_train.mean()
            scalar_loss_for_backward_train.backward(); optimizer.step()
            epoch_train_loss_scalar_sum += scalar_loss_for_backward_train.item() * x_enc_b.size(0)
            epoch_train_loss_per_step_sum += loss_per_element_train.mean(dim=2).sum(dim=0).detach()
        avg_epoch_train_loss_scalar = epoch_train_loss_scalar_sum / len(train_loader_final_s2s.dataset)
        train_losses_scalar_history.append(avg_epoch_train_loss_scalar)
        avg_epoch_train_loss_per_step = (epoch_train_loss_per_step_sum / len(train_loader_final_s2s.dataset)).cpu().numpy()
        train_losses_per_step_history.append(avg_epoch_train_loss_per_step)
        avg_epoch_val_loss_scalar = None; avg_epoch_val_loss_per_step = np.full(output_window_steps, np.nan)
        if val_loader_final_s2s:
            model.eval(); epoch_val_loss_scalar_sum = 0.0; epoch_val_loss_per_step_sum = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for x_enc_vb, x_dec_vb, y_tar_vb in val_loader_final_s2s:
                    x_enc_vb, x_dec_vb, y_tar_vb = x_enc_vb.to(device, non_blocking=True), x_dec_vb.to(device, non_blocking=True), y_tar_vb.to(device, non_blocking=True)
                    outputs_val_epoch = model(x_enc_vb, x_dec_vb, teacher_forcing_ratio=0.0)
                    loss_per_element_val = criterion(outputs_val_epoch, y_tar_vb); scalar_loss_val_batch = loss_per_element_val.mean()
                    epoch_val_loss_scalar_sum += scalar_loss_val_batch.item() * x_enc_vb.size(0)
                    epoch_val_loss_per_step_sum += loss_per_element_val.mean(dim=2).sum(dim=0).detach()
            if len(val_loader_final_s2s.dataset) > 0:
                avg_epoch_val_loss_scalar = epoch_val_loss_scalar_sum / len(val_loader_final_s2s.dataset)
                avg_epoch_val_loss_per_step = (epoch_val_loss_per_step_sum / len(val_loader_final_s2s.dataset)).cpu().numpy()
            else: avg_epoch_val_loss_scalar = float('inf')
            val_losses_scalar_history.append(avg_epoch_val_loss_scalar); val_losses_per_step_history.append(avg_epoch_val_loss_per_step)
            scheduler.step(avg_epoch_val_loss_scalar)
            if avg_epoch_val_loss_scalar < best_val_loss_scalar: best_val_loss_scalar = avg_epoch_val_loss_scalar; best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: val_losses_scalar_history.append(None); val_losses_per_step_history.append(np.full(output_window_steps, np.nan))
        progress_percentage = (epoch + 1) / epochs; current_lr = optimizer.param_groups[0]['lr']; epoch_time = pytime.time() - epoch_start_time
        tf_ratio_str = f"{current_tf_ratio:.2f}" if teacher_forcing_ratio_schedule else "N/A"
        train_loss_per_step_str_s2s = " | ".join([f"{l:.4f}" for l in avg_epoch_train_loss_per_step])
        val_loss_scalar_str_s2s = f"{avg_epoch_val_loss_scalar:.6f}" if avg_epoch_val_loss_scalar is not None and avg_epoch_val_loss_scalar != float('inf') else "N/A"
        val_loss_per_step_str_s2s = "N/A"
        if val_loader_final_s2s and not np.all(np.isnan(avg_epoch_val_loss_per_step)): val_loss_per_step_str_s2s = " | ".join([f"{l:.4f}" for l in avg_epoch_val_loss_per_step])
        status_text.markdown(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) | TF: {tf_ratio_str} | LR: {current_lr:.6f} | Loss: {loss_function_name}<br>'
                             f'&nbsp;&nbsp;Train Loss (Sc): {avg_epoch_train_loss_scalar:.6f} | Train Loss (Step): [{train_loss_per_step_str_s2s}]<br>'
                             f'&nbsp;&nbsp;Val Loss (Sc): {val_loss_scalar_str_s2s} | Val Loss (Step): [{val_loss_per_step_str_s2s}]', unsafe_allow_html=True)
        progress_bar.progress(progress_percentage, text=f"{action_type} Seq2Seq: Epoca {epoch+1}/{epochs}")
        update_loss_chart_seq2seq(train_losses_scalar_history, val_losses_scalar_history, loss_chart_placeholder, title_suffix=f"{action_type.split()[0]} Seq2Seq - ")
    
    total_training_time = pytime.time() - start_training_time; st.write(f"{action_type} Seq2Seq completato in {total_training_time:.1f} secondi.")
    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict:
            try: model.load_state_dict({k: v.to(device) for k, v in best_model_state_dict.items()}); final_model_to_return = model; st.success(f"Strategia 'migliore' Seq2Seq: Caricato modello con Val Loss minima ({best_val_loss_scalar:.6f}).")
            except Exception as e_load_best: st.error(f"Errore caricamento stato migliore Seq2Seq: {e_load_best}. Restituito modello ultima epoca."); final_model_to_return = model
        elif val_loader_final_s2s: st.warning("Strategia 'migliore' Seq2Seq: Nessun miglioramento Val Loss. Restituito modello ultima epoca.")
        else: st.warning("Strategia 'migliore' Seq2Seq: Nessuna validazione. Restituito modello ultima epoca.")
    elif save_strategy == 'finale': final_model_to_return = model; st.info("Strategia 'finale' Seq2Seq: Restituito modello ultima epoca.")

    if n_splits_cv > 1 and X_enc_scaled_full.shape[0] >= n_splits_cv:
        st.markdown(f"--- \n**Valutazione CV Post-{action_type} del Modello Finale (Seq2Seq):**")
        cv_fold_val_losses_scalar_s2s_all = []; cv_fold_val_losses_per_step_s2s_all = []
        final_model_to_return.eval()
        for i_fold_s2s, (train_idx_f_s2s, val_idx_f_s2s) in enumerate(all_splits_indices_s2s):
            X_enc_val_f, X_dec_val_f, y_tar_val_f = X_enc_scaled_full[val_idx_f_s2s], X_dec_scaled_full[val_idx_f_s2s], y_tar_scaled_full[val_idx_f_s2s]
            if X_enc_val_f.size == 0 or y_tar_val_f.size == 0: st.caption(f"Fold CV Seq2Seq {i_fold_s2s+1}/{n_splits_cv}: Set validazione vuoto."); continue
            val_dataset_f_s2s = TimeSeriesDataset(X_enc_val_f, X_dec_val_f, y_tar_val_f); val_loader_f_s2s = DataLoader(val_dataset_f_s2s, batch_size=batch_size, num_workers=0)
            fold_loss_scalar_sum_s2s = 0.0; fold_loss_per_step_sum_s2s = torch.zeros(output_window_steps, device=device)
            with torch.no_grad():
                for x_ef, x_df, y_tf in val_loader_f_s2s:
                    x_ef, x_df, y_tf = x_ef.to(device), x_df.to(device), y_tf.to(device)
                    outputs_f_s2s = final_model_to_return(x_ef, x_df, teacher_forcing_ratio=0.0); loss_elem_f_s2s = criterion(outputs_f_s2s, y_tf)
                    fold_loss_scalar_sum_s2s += loss_elem_f_s2s.mean().item() * x_ef.size(0)
                    fold_loss_per_step_sum_s2s += loss_elem_f_s2s.mean(dim=2).sum(dim=0).detach()
            avg_fold_loss_scalar_s2s = fold_loss_scalar_sum_s2s / len(val_dataset_f_s2s); avg_fold_loss_per_step_s2s = (fold_loss_per_step_sum_s2s / len(val_dataset_f_s2s)).cpu().numpy()
            cv_fold_val_losses_scalar_s2s_all.append(avg_fold_loss_scalar_s2s); cv_fold_val_losses_per_step_s2s_all.append(avg_fold_loss_per_step_s2s)
            st.caption(f"Fold CV Seq2Seq {i_fold_s2s+1}/{n_splits_cv} - Val Loss (Sc): {avg_fold_loss_scalar_s2s:.6f} | Val Loss (Step): [{ ' | '.join([f'{l:.4f}' for l in avg_fold_loss_per_step_s2s])}]")
        if cv_fold_val_losses_scalar_s2s_all:
            avg_cv_scalar_loss_s2s = np.mean(cv_fold_val_losses_scalar_s2s_all); avg_cv_per_step_loss_s2s = np.mean(np.array(cv_fold_val_losses_per_step_s2s_all), axis=0)
            st.success(f"**Media Validazione CV Seq2Seq ({n_splits_cv} folds) - Loss Scalare: {avg_cv_scalar_loss_s2s:.6f}**")
            st.markdown(f"**Media Validazione CV Seq2Seq ({n_splits_cv} folds) - Loss Per Step:** `{ {f'Step {s+1}': f'{l:.4f}' for s, l in enumerate(avg_cv_per_step_loss_s2s)} }`")
        else: st.warning("Nessun fold CV Seq2Seq valutato con successo post-training.")
    else: st.info(f"Valutazione CV Post-{action_type} Seq2Seq non eseguita.")
    return final_model_to_return, (train_losses_scalar_history, train_losses_per_step_history), (val_losses_scalar_history, val_losses_per_step_history) 

# --- Funzioni Helper Download ---
def get_table_download_link(df, filename="data.csv", link_text="Scarica CSV"):
    try:
        csv_buffer = io.StringIO(); df.to_csv(csv_buffer, index=False, sep=';', decimal=',', encoding='utf-8-sig'); csv_buffer.seek(0); b64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8-sig')).decode()
        return f'<a href="data:text/csv;charset=utf-8-sig;base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.warning(f"Errore link CSV: {e}"); return "<i>Errore link CSV</i>"
def get_binary_file_download_link(file_object, filename, text):
    try:
        file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode(); mime_type, _ = mimetypes.guess_type(filename); mime_type = mime_type or 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
    except Exception as e: st.warning(f"Errore link binario {filename}: {e}"); return f"<i>Errore link {filename}</i>"
def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    href_html = ""; href_png = ""; safe_filename_base = re.sub(r'[^\w-]', '_', filename_base)
    try: html_filename = f"{safe_filename_base}.html"; buf_html = io.StringIO(); fig.write_html(buf_html, include_plotlyjs='cdn'); buf_html.seek(0); b64_html = base64.b64encode(buf_html.getvalue().encode()).decode(); href_html = f'<a href="data:text/html;base64,{b64_html}" download="{html_filename}">{text_html}</a>'
    except Exception as e_html: print(f"Errore download HTML {safe_filename_base}: {e_html}"); href_html = "<i>Errore HTML</i>"
    try:
        import importlib
        if importlib.util.find_spec("kaleido"): png_filename = f"{safe_filename_base}.png"; buf_png = io.BytesIO(); fig.write_image(buf_png, format="png", scale=2); buf_png.seek(0); b64_png = base64.b64encode(buf_png.getvalue()).decode(); href_png = f'<a href="data:image/png;base64,{b64_png}" download="{png_filename}">{text_png}</a>'
        else: print("Nota: 'kaleido' non installato, download PNG grafico disabilitato.")
    except Exception as e_png: print(f"Errore download PNG {safe_filename_base}: {e_png}"); href_png = "<i>Errore PNG</i>"
    return f"{href_html} {href_png}".strip()
def get_download_link_for_file(filepath, link_text=None):
    if not os.path.exists(filepath): return f"<i>File non trovato: {os.path.basename(filepath)}</i>"
    filename = os.path.basename(filepath); link_text = link_text or f"Scarica {filename}"
    try:
        with open(filepath, "rb") as f: file_content = f.read(); b64 = base64.b64encode(file_content).decode("utf-8"); mime_type, _ = mimetypes.guess_type(filepath); mime_type = mime_type or 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.error(f"Errore generazione link per {filename}: {e}"); return f"<i>Errore link</i>"

# --- Funzione Estrazione ID GSheet & Etichetta Stazione ---
def extract_sheet_id(url): patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/'];
                        for pattern in patterns: match = re.search(pattern, url);
                        if match: return match.group(1); return None
def get_station_label(col_name, short=False):
    if col_name in STATION_COORDS:
        info = STATION_COORDS[col_name]; loc_id = info.get('location_id'); s_type = info.get('type', ''); name = info.get('name', loc_id or col_name)
        if short and loc_id:
            sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == loc_id]
            if len(sensors_at_loc) > 1: type_abbr = {'Pioggia': 'P', 'Livello': 'L', 'Umidità': 'U'}.get(s_type, '?'); label = f"{loc_id} ({type_abbr})"
            else: label = loc_id
            label = re.sub(r'\s*\(.*?\)\s*', '', label).strip(); return label[:20] + ('...' if len(label) > 20 else '')
        else: return name
    label = col_name; label = re.sub(r'\s*\[.*?\]|\s*\(.*?\)', '', label).strip(); label = label.replace('Sensore ', '').replace('Livello Idrometrico ', '').replace(' - Pioggia Ora', '').replace(' - Livello Misa', '').replace(' - Livello Nevola', ''); parts = label.split(' '); label = ' '.join(parts[:2])
    return label[:20] + ('...' if len(label) > 20 else '') if short else label

# --- Inizializzazione Session State ---
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
# ... (altri stati sessione invariati) ...
if 'current_page' not in st.session_state: st.session_state.current_page = 'Dashboard' # Default
# ... (mantenere gli altri stati di sessione come sono) ...
if 'df' not in st.session_state: st.session_state.df = None
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [ # Queste sono le feature *iniziali* selezionabili dall'utente
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
        logo_path = "logo.png"
        if os.path.exists(logo_path): st.image(logo_path, use_container_width=True)
        else: st.caption("Logo non trovato.")
    except Exception as e: st.warning(f"Impossibile caricare il logo: {e}")
    st.header('Impostazioni')
    st.subheader('Dati Storici (per Analisi/Training/Test/Post-Training)')
    uploaded_data_file = st.file_uploader('Carica CSV Dati Storici', type=['csv'], key="data_uploader")
    df = None; df_load_error = None; data_source_info = ""
    data_path_to_load = None; is_uploaded = False
    if uploaded_data_file is not None: data_path_to_load = uploaded_data_file; is_uploaded = True; data_source_info = f"File caricato: **{uploaded_data_file.name}**"
    elif os.path.exists(DEFAULT_DATA_PATH):
        if 'df' not in st.session_state or st.session_state.df is None or st.session_state.get('uploaded_file_processed') is None or st.session_state.get('uploaded_file_processed') == False:
            data_path_to_load = DEFAULT_DATA_PATH; data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
            st.session_state['uploaded_file_processed'] = False # Indica che il default non è stato ancora processato in questa sessione se `df` è None
        else: data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"; df = st.session_state.df # Usa df da sessione
    elif 'df' not in st.session_state or st.session_state.df is None: df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato e nessun file caricato."

    process_file = False
    if is_uploaded and st.session_state.get('uploaded_file_processed') != uploaded_data_file.name: process_file = True # Nuovo file caricato
    elif data_path_to_load == DEFAULT_DATA_PATH and not st.session_state.get('uploaded_file_processed') and ('df' not in st.session_state or st.session_state.df is None) : process_file = True # Default non ancora processato e df non in sessione

    if process_file and data_path_to_load:
        with st.spinner("Caricamento e processamento dati CSV..."):
            # ... (logica di caricamento CSV invariata) ...
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
                else: st.session_state['uploaded_file_processed'] = True # Default file successfully processed
                st.success(f"Dati CSV caricati ({len(st.session_state.df)} righe)."); df = st.session_state.df
            except Exception as e: df = None; st.session_state.df = None; st.session_state['uploaded_file_processed'] = None; df_load_error = f'Errore caricamento/processamento CSV: {e}'; st.error(f"Errore CSV: {df_load_error}"); print(traceback.format_exc())

    if df is None and 'df' in st.session_state and st.session_state.df is not None: # Se df non è stato appena processato ma è in sessione
        df = st.session_state.df
        if data_source_info == "": # Aggiorna info se non già impostata
            if st.session_state.get('uploaded_file_processed') and isinstance(st.session_state.get('uploaded_file_processed'), str): data_source_info = f"File caricato: **{st.session_state['uploaded_file_processed']}** (da sessione)"
            elif st.session_state.get('uploaded_file_processed') == True: data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"
    if data_source_info: st.caption(data_source_info)
    if df_load_error and not data_source_info: st.error(df_load_error) # Mostra errore solo se non c'è altra info
    data_ready_csv = df is not None
    st.divider()

    st.subheader("Modello Predittivo (per Simulazione/Test/Post-Training)")
    available_models_dict = find_available_models(MODELS_DIR)
    model_display_names = sorted(list(available_models_dict.keys()))
    MODEL_CHOICE_UPLOAD = "Carica File Manualmente (Solo LSTM)"; MODEL_CHOICE_NONE = "-- Nessun Modello Selezionato --"
    selection_options = [MODEL_CHOICE_NONE] + model_display_names
    current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)
    if current_selection_name not in selection_options and current_selection_name != MODEL_CHOICE_UPLOAD:
        st.session_state.active_model_name = MODEL_CHOICE_NONE; current_selection_name = MODEL_CHOICE_NONE
        # ... reset altri stati attivi del modello ...
    try: current_index = selection_options.index(current_selection_name)
    except ValueError: current_index = 0
    selected_model_display_name = st.selectbox("Modello Pre-Addestrato:", selection_options, index=current_index, key="model_selector_predefined")

    # ... (logica caricamento modello manuale e da lista invariata) ...
    with st.expander("Carica Modello LSTM Manualmente", expanded=(current_selection_name == MODEL_CHOICE_UPLOAD)):
         is_upload_active = False; m_f = st.file_uploader('File Modello (.pth)', type=['pth'], key="up_pth"); sf_f = st.file_uploader('File Scaler Features (.joblib)', type=['joblib'], key="up_scf_lstm"); st_f = st.file_uploader('File Scaler Target (.joblib)', type=['joblib'], key="up_sct_lstm")
         if m_f and sf_f and st_f:
            st.caption("Configura parametri del modello LSTM caricato:")
            col1_up, col2_up = st.columns(2)
            with col1_up: iw_up = st.number_input("Input Window (steps)", 1, 500, 48, 1, key="up_in_steps_lstm"); ow_up = st.number_input("Output Window (steps)", 1, 500, 24, 1, key="up_out_steps_lstm"); hs_up = st.number_input("Hidden Size", 8, 1024, 128, 8, key="up_hid_lstm")
            with col2_up: nl_up = st.number_input("Numero Layers", 1, 8, 2, 1, key="up_lay_lstm"); dr_up = st.slider("Dropout", 0.0, 0.9, 0.2, 0.05, key="up_drop_lstm")
            
            # Per le feature/target del modello caricato manualmente, usiamo le feature disponibili dal CSV se caricato, altrimenti le default
            available_cols_for_upload = list(df.columns.drop(st.session_state.date_col_name_csv, errors='ignore')) if data_ready_csv else st.session_state.feature_columns
            
            # Qui assumiamo che le feature_columns siano quelle *effettive* che il modello si aspetta.
            # Per un modello caricato manualmente, l'utente DEVE specificarle correttamente.
            # Non c'è modo di sapere quali feature ingegnerizzate sono state usate esternamente.
            # Quindi, per il caricamento manuale, "model_input_feature_columns" sarà uguale a "user_initial_features".
            default_features_upload = [c for c in st.session_state.feature_columns if c in available_cols_for_upload] # Iniziali
            features_up_manual = st.multiselect("Feature Columns (Input Effettivo del Modello)", available_cols_for_upload, default=default_features_upload, key="up_features_manual_lstm")

            default_targets_upload = [c for c in available_cols_for_upload if 'Livello' in c or '[m]' in c][:1]
            targets_up = st.multiselect("Target Columns (Output)", available_cols_for_upload, default=default_targets_upload, key="up_targets_lstm")

            if targets_up and features_up_manual:
                if st.button("Carica Modello Manuale", key="load_manual_lstm", type="secondary"): is_upload_active = True; selected_model_display_name = MODEL_CHOICE_UPLOAD
            else: st.caption("Seleziona features e target per il modello caricato.")
         else: st.caption("Carica tutti e tre i file per abilitare.")

    config_to_load = None; model_to_load = None; device_to_load = None; scalers_to_load = None; load_error_sidebar = False; model_type_loaded = None
    if is_upload_active: # Caricamento manuale
        st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
        # Per i modelli caricati manualmente, assumiamo che le "feature_columns" specificate dall'utente
        # siano le "model_input_feature_columns" effettive. Non c'è feature engineering interna qui.
        temp_cfg_up = {
            "input_window": iw_up, "output_window": ow_up, 
            "hidden_size": hs_up, "num_layers": nl_up, "dropout": dr_up, 
            "model_input_feature_columns": features_up_manual, # Feature effettive
            "user_initial_features": features_up_manual, # Uguali per caricamento manuale
            "target_columns": targets_up, 
            "name": "uploaded_lstm_model", "display_name": "Modello LSTM Caricato", "model_type": "LSTM",
            "lag_config": {}, "cumulative_config": {} # No FE info per caricamento manuale
        }
        model_to_load, device_to_load = load_specific_model(m_f, temp_cfg_up)
        temp_model_info_up = {"scaler_features_path": sf_f, "scaler_targets_path": st_f} # Solo per LSTM
        scalers_tuple_up = load_specific_scalers(temp_cfg_up, temp_model_info_up) # Dovrebbe restituire (scaler_f, scaler_t)
        if model_to_load and scalers_tuple_up and isinstance(scalers_tuple_up, tuple) and len(scalers_tuple_up)==2 and all(s is not None for s in scalers_tuple_up):
            config_to_load = temp_cfg_up; scalers_to_load = scalers_tuple_up; model_type_loaded = "LSTM"
        else: load_error_sidebar = True; st.error("Errore caricamento modello/scaler LSTM manuale.")

    elif selected_model_display_name != MODEL_CHOICE_NONE: # Scelta da lista
        if selected_model_display_name != st.session_state.get('active_model_name'):
            st.session_state.active_model_name = selected_model_display_name
            if selected_model_display_name in available_models_dict:
                model_info = available_models_dict[selected_model_display_name]
                config_to_load = load_model_config(model_info["config_path"])
                if config_to_load:
                    config_to_load["pth_path"] = model_info["pth_path"]
                    config_to_load["config_name"] = model_info["config_name"] # Nome base del file
                    config_to_load["display_name"] = selected_model_display_name # Nome visualizzato
                    model_type_loaded = model_info.get("model_type", "LSTM")
                    config_to_load["model_type"] = model_type_loaded # Assicura che sia nel config

                    model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
                    scalers_to_load = load_specific_scalers(config_to_load, model_info) # model_info ha i percorsi
                    if not (model_to_load and scalers_to_load): load_error_sidebar = True; config_to_load = None
                else: load_error_sidebar = True # Errore caricamento config
            else: st.error(f"Modello '{selected_model_display_name}' non trovato."); load_error_sidebar = True
        elif st.session_state.get('active_model') and st.session_state.get('active_config'): # Già attivo
             config_to_load = st.session_state.active_config; model_to_load = st.session_state.active_model
             device_to_load = st.session_state.active_device; scalers_to_load = st.session_state.active_scalers
             model_type_loaded = config_to_load.get("model_type", "LSTM")

    elif selected_model_display_name == MODEL_CHOICE_NONE and st.session_state.get('active_model_name') != MODEL_CHOICE_NONE: # Deselezione
         st.session_state.active_model_name = MODEL_CHOICE_NONE; st.session_state.active_config = None
         st.session_state.active_model = None; st.session_state.active_device = None; st.session_state.active_scalers = None
         config_to_load = None; model_to_load = None # Pulisce variabili locali

    # Aggiorna stato sessione
    if config_to_load and model_to_load and device_to_load and scalers_to_load:
        # Verifica validità scaler
        scalers_are_valid = False
        if model_type_loaded == "Seq2Seq":
            if isinstance(scalers_to_load, dict) and all(s is not None for s in scalers_to_load.values()):
                scalers_are_valid = True
        else: # LSTM
            if isinstance(scalers_to_load, tuple) and len(scalers_to_load) == 2 and all(s is not None for s in scalers_to_load):
                scalers_are_valid = True
        
        if scalers_are_valid:
            st.session_state.active_config = config_to_load
            st.session_state.active_model = model_to_load
            st.session_state.active_device = device_to_load
            st.session_state.active_scalers = scalers_to_load
        else:
            st.error(f"Scaler per modello '{selected_model_display_name}' non caricati correttamente."); load_error_sidebar = True
            st.session_state.active_config = None; st.session_state.active_model = None # Reset
            st.session_state.active_device = None; st.session_state.active_scalers = None

    active_config_sess = st.session_state.get('active_config')
    active_model_sess = st.session_state.get('active_model')
    active_model_name_sess = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

    if active_model_sess and active_config_sess:
        cfg_sidebar = active_config_sess
        model_type_sidebar = cfg_sidebar.get('model_type', 'LSTM')
        display_feedback_name_sidebar = cfg_sidebar.get("display_name", active_model_name_sess)
        st.success(f"Modello Attivo: **{display_feedback_name_sidebar}** ({model_type_sidebar})")
        if model_type_sidebar == "Seq2Seq": st.caption(f"Input: {cfg_sidebar['input_window_steps']}s | Forecast: {cfg_sidebar['forecast_window_steps']}s | Output: {cfg_sidebar['output_window_steps']}s")
        else: st.caption(f"Input: {cfg_sidebar['input_window']}s | Output: {cfg_sidebar['output_window']}s")
        if cfg_sidebar.get("is_finetuned"):
            st.caption(f" Fine-tuned da: {cfg_sidebar.get('base_model_config_name', 'N/A')} il {pd.to_datetime(cfg_sidebar.get('finetuning_date','')).strftime('%d/%m/%Y') if cfg_sidebar.get('finetuning_date') else 'N/A'}")

    elif load_error_sidebar and active_model_name_sess not in [MODEL_CHOICE_NONE]: st.error(f"Caricamento modello '{active_model_name_sess}' fallito.")
    elif active_model_name_sess == MODEL_CHOICE_UPLOAD and not active_model_sess : st.info("Completa il caricamento manuale del modello LSTM.")
    elif active_model_name_sess == MODEL_CHOICE_NONE: st.info("Nessun modello selezionato.")
    st.divider()

    st.subheader("Configurazione Soglie Dashboard")
    with st.expander("Modifica Soglie di Allerta (per Dashboard)"):
        # ... (logica soglie dashboard invariata) ...
        temp_thresholds = st.session_state.dashboard_thresholds.copy(); monitorable_cols_thresh = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]; cols_thresh_ui = st.columns(2); col_idx_thresh = 0
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
    radio_options = ['Dashboard', 'Simulazione', 'Test Modello su Storico', 'Analisi Dati Storici', 'Allenamento Modello', 'Post-Training Modello LSTM']
    radio_captions = []; disabled_options = []; default_page_idx = 0
    for i, opt in enumerate(radio_options):
        caption = ""; disabled = False
        if opt == 'Dashboard': caption = "Monitoraggio GSheet"
        elif opt == 'Simulazione':
            if not model_ready: caption = "Richiede Modello attivo"; disabled = True
            elif active_config_sess: caption = f"Esegui previsioni ({active_config_sess.get('model_type', 'N/A')})"
            else: caption = "Richiede Modello attivo"; disabled = True # Should not happen if model_ready is true
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
        elif opt == 'Post-Training Modello LSTM':
            # Check model_ready first, then active_config_sess, then model_type
            is_lstm_model_active = model_ready and active_config_sess and active_config_sess.get('model_type') == 'LSTM'
            if not is_lstm_model_active:
                caption = "Richiede Modello LSTM attivo"
                disabled = True
            else:
                caption = "Affina un modello LSTM esistente con nuovi dati"
                disabled = False # Data for fine-tuning is uploaded on the page itself
        radio_captions.append(caption); disabled_options.append(disabled)
        if opt == st.session_state.get('current_page', radio_options[0]): default_page_idx = i # Default alla pagina corrente o prima
    
    if disabled_options[default_page_idx] and radio_options[default_page_idx] == st.session_state.get('current_page'):
        st.warning(f"Pagina '{radio_options[default_page_idx]}' non più disponibile ({radio_captions[default_page_idx]}). Reindirizzato a Dashboard.")
        st.session_state['current_page'] = 'Dashboard'; default_page_idx = radio_options.index('Dashboard')

    chosen_page = st.radio('Scegli una funzionalità:', options=radio_options, captions=radio_captions, index=default_page_idx, key='page_selector_radio')
    if chosen_page != st.session_state['current_page']:
        chosen_page_index = radio_options.index(chosen_page)
        if not disabled_options[chosen_page_index]: st.session_state['current_page'] = chosen_page; st.rerun()
        else: st.error(f"La pagina '{chosen_page}' non è disponibile. {radio_captions[chosen_page_index]}.")
    page = st.session_state['current_page']


# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================
active_config = st.session_state.get('active_config')
active_model = st.session_state.get('active_model')
active_device = st.session_state.get('active_device')
active_scalers = st.session_state.get('active_scalers') # LSTM: (sc_f, sc_t), Seq2Seq: {"past":sc_p, "forecast":sc_f, "targets":sc_t}
active_model_type = active_config.get("model_type", "LSTM") if active_config else "LSTM" # Default to LSTM if no config
model_ready = active_model is not None and active_config is not None and active_scalers is not None
df_current_csv = st.session_state.get('df', None)
data_ready_csv = df_current_csv is not None
date_col_name_csv = st.session_state.date_col_name_csv


# --- PAGINA DASHBOARD ---
if page == 'Dashboard':
    # ... (codice Dashboard invariato - omesso per brevità)
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
        df_dashboard, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(cache_time_key_recent, GSHEET_ID, GSHEET_RELEVANT_COLS, GSHEET_DATE_COL, GSHEET_DATE_FORMAT, fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS)
        st.session_state.last_dashboard_data = df_dashboard; st.session_state.last_dashboard_error = error_msg
        if df_dashboard is not None or error_msg is None: st.session_state.last_dashboard_fetch_time = actual_fetch_time
        data_source_mode = f"Ultime {DASHBOARD_HISTORY_ROWS // 2} ore circa"

    col_status, col_refresh_btn = st.columns([4, 1])
    with col_status:
        last_fetch_dt_sess = st.session_state.get('last_dashboard_fetch_time')
        if last_fetch_dt_sess:
             fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess; fetch_secs_ago = int(fetch_time_ago.total_seconds()) # Modificato per secondi
             status_text = f"{data_source_mode}. Aggiornato alle {last_fetch_dt_sess.strftime('%H:%M:%S')} ({fetch_secs_ago}s fa)." # Modificato
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
                 value_numeric = float(current_value_H); fmt_spec = ".1f" if unit in ['mm', '%'] else ".2f"
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
            table_rows.append({"Stazione": get_station_label(col_name, short=True), "Valore H": value_display, "Portata Q (m³/s)": portata_display, "Soglia H": threshold_display, "Stato": status, "Valore Numerico H": value_numeric, "Soglia Numerica H": float(threshold) if threshold else None})
        df_display = pd.DataFrame(table_rows)
        def highlight_alert(row): style = [''] * len(row); style = ['background-color: rgba(255, 0, 0, 0.1);'] * len(row) if row['Stato'] == 'ALLERTA' else style; return style
        st.dataframe(df_display.style.apply(highlight_alert, axis=1), column_order=["Stazione", "Valore H", "Portata Q (m³/s)", "Soglia H", "Stato"], hide_index=True, use_container_width=True, column_config={"Valore Numerico H": None, "Soglia Numerica H": None})
        st.session_state.active_alerts = current_alerts
        st.divider()
        if GSHEET_DATE_COL in df_dashboard.columns:
            st.subheader("Grafico Storico Comparativo (Periodo Selezionato)")
            sensor_options_compare = {get_station_label(col, short=True): col for col in cols_to_monitor}
            default_selection_labels = [label for label, col in sensor_options_compare.items() if 'Livello' in col][:3] or list(sensor_options_compare.keys())[:2]
            selected_labels_compare = st.multiselect("Seleziona sensori da confrontare:", options=list(sensor_options_compare.keys()), default=default_selection_labels, key="compare_select_multi")
            selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare]
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
                    if threshold_individual is not None and ('Livello' in col_name or '(m)' in col_name or '(mt)' in col_name): fig_individual.add_hline(y=threshold_individual, line_dash="dash", line_color="red", annotation_text=f"Soglia H ({float(threshold_individual):.1f})", annotation_position="bottom right")
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
             st.warning("**Allerte Attive (Valori H >= Soglia H):**"); alert_md = ""
             sorted_alerts = sorted(active_alerts_sess, key=lambda x: get_station_label(x[0], short=False))
             for col, val, thr in sorted_alerts:
                 label_alert = get_station_label(col, short=False); sensor_type_alert = STATION_COORDS.get(col, {}).get('type', ''); type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                 unit_match_alert = re.search(r'\((.*?)\)|\[(.*?)\]', col); unit_alert = ""
                 if unit_match_alert: unit_content = unit_match_alert.group(1) or unit_match_alert.group(2); unit_alert = f"({unit_content.strip()})" if unit_content else ""
                 val_fmt = f"{val:.1f}" if unit_alert in ['(mm)', '(%)'] else f"{val:.2f}"; thr_fmt = f"{float(thr):.1f}" if isinstance(thr, (int, float, np.number)) else str(thr)
                 alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit_alert}** >= Soglia **{thr_fmt}{unit_alert}**\n"
             st.markdown(alert_md)
        else: st.success("Nessuna soglia H superata nell'ultimo rilevamento del periodo visualizzato.")
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

# --- PAGINA SIMULAZIONE ---
elif page == 'Simulazione':
    st.header(f'Simulazione Idrologica ({active_model_type})')
    if not model_ready: st.warning("Seleziona un Modello attivo dalla sidebar per eseguire la Simulazione."); st.stop()
    # ... (codice Simulazione invariato - omesso per brevità)
    st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}** ({active_model_type})")
    target_columns_model = []
    if active_model_type == "Seq2Seq":
        st.caption(f"Input Storico: {active_config['input_window_steps']} steps | Input Forecast: {active_config['forecast_window_steps']} steps | Output: {active_config['output_window_steps']} steps")
        with st.expander("Dettagli Colonne Modello Seq2Seq"):
             encoder_features_key_sim = "encoder_input_feature_columns" if "encoder_input_feature_columns" in active_config else "all_past_feature_columns"
             decoder_features_key_sim = "decoder_input_feature_columns" if "decoder_input_feature_columns" in active_config else "forecast_input_columns"
             st.markdown("**Feature Storiche (Input Encoder):**"); st.caption(f"`{', '.join(active_config[encoder_features_key_sim])}`")
             st.markdown("**Feature Forecast (Decoder Input):**"); st.caption(f"`{', '.join(active_config[decoder_features_key_sim])}`")
             st.markdown("**Target (Output - Livello H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        target_columns_model = active_config['target_columns']
        input_steps_model = active_config['input_window_steps']
        output_steps_model = active_config['output_window_steps']
        past_feature_cols_model = active_config[encoder_features_key_sim]
        forecast_feature_cols_model = active_config[decoder_features_key_sim]
        forecast_steps_model = active_config['forecast_window_steps']
    else: # LSTM Standard
        features_key_lstm_sim = "model_input_feature_columns" if "model_input_feature_columns" in active_config else "feature_columns"
        feature_columns_model = active_config[features_key_lstm_sim]
        st.caption(f"Input: {active_config['input_window']} steps | Output: {active_config['output_window']} steps")
        with st.expander("Dettagli Colonne Modello LSTM"):
             st.markdown("**Feature Input Effettive:**"); st.caption(f"`{', '.join(feature_columns_model)}`")
             if "user_initial_features" in active_config:
                 st.markdown("**Feature Iniziali Utente (prima di FE):**"); st.caption(f"`{', '.join(active_config['user_initial_features'])}`")
             st.markdown("**Target (Output - Livello H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        target_columns_model = active_config['target_columns']
        input_steps_model = active_config['input_window'] # in steps
        output_steps_model = active_config['output_window'] # in steps

    st.divider()
    if active_model_type == "Seq2Seq":
         st.subheader(f"Preparazione Dati Input Simulazione Seq2Seq")
         st.markdown("**Passo 1: Recupero Dati Storici (Input Encoder)**"); st.caption(f"Verranno recuperati gli ultimi {input_steps_model} steps dal Foglio Google (ID: `{GSHEET_ID}`).")
         date_col_model_name = st.session_state.date_col_name_csv
         column_mapping_gsheet_to_past_s2s = {'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)', 'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)', 'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)', 'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)', 'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME, 'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)', 'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)', 'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', 'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)', 'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)', GSHEET_DATE_COL: date_col_model_name}
         required_model_cols_for_mapping = past_feature_cols_model + [date_col_model_name] 
         column_mapping_gsheet_to_past_s2s_filtered = {gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_past_s2s.items() if model_col in required_model_cols_for_mapping}
         missing_past_features_in_map = list(set(past_feature_cols_model) - (set(column_mapping_gsheet_to_past_s2s_filtered.values()) - {date_col_model_name})); imputed_values_sim_past_s2s = {}
         if missing_past_features_in_map:
             st.caption(f"Feature storiche non mappate da GSheet: `{', '.join(missing_past_features_in_map)}`. Saranno imputate.");
             for missing_f in missing_past_features_in_map: imputed_values_sim_past_s2s[missing_f] = df_current_csv[missing_f].median() if data_ready_csv and missing_f in df_current_csv.columns and pd.notna(df_current_csv[missing_f].median()) else 0.0
         fetch_error_gsheet_s2s = None
         if st.button("Carica/Aggiorna Storico da GSheet", key="fetch_gsh_s2s_base"):
              fetch_sim_gsheet_data.clear()
              with st.spinner("Recupero dati storici (passato)..."): imported_df_past, import_err_past, last_ts_past = fetch_sim_gsheet_data(GSHEET_ID, input_steps_model, GSHEET_DATE_COL, GSHEET_DATE_FORMAT, column_mapping_gsheet_to_past_s2s_filtered, past_feature_cols_model + [date_col_model_name], imputed_values_sim_past_s2s)
              if import_err_past: st.error(f"Recupero storico Seq2Seq fallito: {import_err_past}"); st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = import_err_past
              elif imported_df_past is not None:
                  try: final_past_df = imported_df_past[past_feature_cols_model]; st.success(f"Recuperate e processate {len(final_past_df)} righe storiche."); st.session_state.seq2seq_past_data_gsheet = final_past_df; st.session_state.seq2seq_last_ts_gsheet = last_ts_past if last_ts_past else datetime.now(italy_tz); fetch_error_gsheet_s2s = None; st.rerun()
                  except KeyError as e_cols: missing_cols_final = [c for c in past_feature_cols_model if c not in imported_df_past.columns]; st.error(f"Errore selezione colonne storiche dopo fetch: Colonne mancanti {missing_cols_final}"); st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = f"Errore colonne: {e_cols}"
              else: st.error("Recupero storico Seq2Seq non riuscito (risultato vuoto)."); fetch_error_gsheet_s2s = "Errore recupero dati."
         past_data_loaded_s2s = st.session_state.get('seq2seq_past_data_gsheet') is not None
         if past_data_loaded_s2s: st.caption("Dati storici base (input encoder) caricati."); st.expander("Mostra dati storici caricati (Input Encoder)").dataframe(st.session_state.seq2seq_past_data_gsheet.round(3))
         elif fetch_error_gsheet_s2s: st.warning(f"Caricamento dati storici fallito ({fetch_error_gsheet_s2s}). Clicca il bottone per riprovare.")
         else: st.info("Clicca il bottone 'Carica/Aggiorna Storico da GSheet' per recuperare i dati necessari.")

         if past_data_loaded_s2s:
             num_rows_decoder_input = max(forecast_steps_model, output_steps_model)
             st.markdown(f"**Passo 2: Inserisci Input Futuri per Decoder ({num_rows_decoder_input} steps)**"); st.caption(f"Inserisci i valori per le seguenti feature: `{', '.join(forecast_feature_cols_model)}`")
             forecast_df_initial = pd.DataFrame(index=range(num_rows_decoder_input), columns=forecast_feature_cols_model); last_known_past_data = st.session_state.seq2seq_past_data_gsheet.iloc[-1]
             for col in forecast_feature_cols_model:
                  if 'pioggia' in col.lower() or 'cumulata' in col.lower(): forecast_df_initial[col] = 0.0
                  elif col in last_known_past_data.index and col in forecast_feature_cols_model : forecast_df_initial[col] = last_known_past_data.get(col, 0.0) 
                  else: forecast_df_initial[col] = 0.0
             edited_forecast_df = st.data_editor(forecast_df_initial.round(2), key="seq2seq_forecast_editor", num_rows="fixed", use_container_width=True, column_config={col: st.column_config.NumberColumn(label=get_station_label(col, short=True), help=f"Valore previsto per {col}", min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower() or 'livello' in col.lower()) else None, max_value=100.0 if 'umidit' in col.lower() else None, format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f", step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1), required=True) for col in forecast_feature_cols_model})
             forecast_data_valid_s2s = False
             if edited_forecast_df is not None and not edited_forecast_df.isnull().any().any():
                 if edited_forecast_df.shape == (num_rows_decoder_input, len(forecast_feature_cols_model)):
                     try: edited_forecast_df.astype(float); forecast_data_valid_s2s = True
                     except ValueError: st.warning("Valori non numerici rilevati nella tabella previsioni.")
                 else: st.warning("Numero di righe o colonne errato nella tabella previsioni.")
             else: st.warning("Completa o correggi la tabella delle previsioni future. Tutti i valori devono essere numerici.")

             st.divider(); st.markdown("**Passo 3: Esegui Simulazione Seq2Seq**"); can_run_s2s_sim = past_data_loaded_s2s and forecast_data_valid_s2s
             if st.button("Esegui Simulazione Seq2Seq", disabled=not can_run_s2s_sim, type="primary", key="run_s2s_sim_button"):
                  if not can_run_s2s_sim: st.error("Mancano dati storici validi o previsioni future valide.")
                  else:
                       # ... (Simulazione Seq2Seq - omessa per brevità)
                       st.warning("Logica simulazione Seq2Seq non mostrata per brevità.")
    else: # LSTM
         st.subheader("Metodo Input Dati Simulazione LSTM")
         sim_method_options = ['Manuale (Valori Costanti)', 'Importa da Google Sheet (Ultime Ore)', 'Orario Dettagliato (Tabella)']
         if data_ready_csv: sim_method_options.append('Usa Ultime Ore da CSV Caricato')
         sim_method = st.radio("Scegli come fornire i dati di input:", sim_method_options, key="sim_method_radio_select_lstm", horizontal=True, label_visibility="collapsed")
         predictions_sim_lstm = None; start_pred_time_lstm = None # Inizializza qui
         
         if sim_method == 'Manuale (Valori Costanti)':
             # ... (Logica UI input manuale - omessa)
             st.warning("Logica UI input manuale LSTM non mostrata per brevità.")
             if st.button('Esegui Simulazione LSTM (Manuale)', type="primary", key="sim_run_exec_lstm_manual_placeholder"):
                 st.warning("Logica simulazione LSTM Manuale non mostrata.")

         elif sim_method == 'Importa da Google Sheet (Ultime Ore)':
             # ... (Logica UI GSheet - omessa)
             st.warning("Logica UI GSheet LSTM non mostrata per brevità.")
             if st.button('Esegui Simulazione LSTM (GSheet)', type="primary", key="sim_run_exec_lstm_gsheet_placeholder"):
                 st.warning("Logica simulazione LSTM GSheet non mostrata.")
         
         elif sim_method == 'Orario Dettagliato (Tabella)':
             # ... (Logica UI Tabella - omessa)
             st.warning("Logica UI Tabella LSTM non mostrata per brevità.")
             if st.button('Esegui Simulazione LSTM (Tabella)', type="primary", key="sim_run_exec_lstm_editor_placeholder"):
                st.warning("Logica simulazione LSTM Tabella non mostrata.")
         
         elif sim_method == 'Usa Ultime Ore da CSV Caricato':
             # ... (Logica UI CSV - omessa)
             st.warning("Logica UI CSV LSTM non mostrata per brevità.")
             if st.button('Esegui Simulazione LSTM (CSV)', type="primary", key="sim_run_exec_lstm_csv_placeholder"):
                 st.warning("Logica simulazione LSTM CSV non mostrata.")

         if predictions_sim_lstm is not None: # Questo blocco ora è unificato
             # ... (Logica visualizzazione risultati LSTM - omessa per brevità)
             st.warning("Logica visualizzazione risultati LSTM non mostrata.")
         elif any(st.session_state.get(f"sim_run_exec_lstm_{method.split()[0].lower()}", False) for method in sim_method_options if "LSTM" in method) :
            st.error(f"Predizione simulazione LSTM ({sim_method}) fallita o non ancora eseguita.")


# --- PAGINA TEST MODELLO SU STORICO ---
elif page == 'Test Modello su Storico':
    # ... (codice Test Modello invariato - omesso per brevità) ...
    st.header('Test Modello su Dati Storici CSV (Walk-Forward Evaluation)')


# --- PAGINA ANALISI DATI STORICI ---
elif page == 'Analisi Dati Storici':
    # ... (codice Analisi Dati invariato - omesso per brevità) ...
    st.header('Analisi Dati Storici (da file CSV)')


# --- PAGINA ALLENAMENTO MODELLO (MODIFICATA) ---
elif page == 'Allenamento Modello':
    # ... (codice Allenamento Modello invariato - omesso per brevità) ...
    st.header('Allenamento Nuovo Modello')

# --- PAGINA POST-TRAINING MODELLO LSTM ---
elif page == 'Post-Training Modello LSTM':
    st.header("Post-Training di Modello LSTM Esistente")

    if not model_ready or not active_config or active_model_type != 'LSTM':
        st.warning("Seleziona un modello LSTM attivo dalla sidebar per procedere con il post-training/fine-tuning.")
        st.stop()

    # Display active model information
    st.info(f"Modello LSTM Attivo per Fine-tuning: **{active_config.get('display_name', 'N/A')}**")
    with st.expander("Dettagli Modello Attivo (LSTM)", expanded=False):
        config_display_ft = dict(active_config) if active_config else {} # Ensure it's a plain dict
        st.json(config_display_ft)

    # New Data Uploader
    new_data_file_ft = st.file_uploader("Carica CSV con i nuovi dati per il fine-tuning:", type=['csv'], key="ft_data_uploader")
    
    # New Model Name
    default_ft_model_name = f"{active_config.get('config_name', 'modello')}_ft_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
    new_model_name_ft = st.text_input("Nome per il modello post-trainato:", value=default_ft_model_name, key="ft_new_model_name")
    save_name_ft = re.sub(r'[\W-]+', '_', new_model_name_ft).strip('_') or "modello_ft_default"
    
    if save_name_ft != new_model_name_ft:
        st.caption(f"Nome file valido suggerito: `{save_name_ft}`")

    # Fine-Tuning Hyperparameters
    with st.expander("Configurazione Parametri di Fine-Tuning (LSTM)", expanded=True):
        c1_ft, c2_ft, c3_ft = st.columns(3)
        with c1_ft:
            ep_ft = st.number_input("Numero Epoche (Fine-Tuning):", min_value=1, value=max(1, int(active_config.get('epochs', 50) * 0.5)), step=1, key="ft_epochs", help="Default: 50% delle epoche del modello originale, min 1.")
            n_splits_cv_ft = st.number_input("Numero di Fold CV per Fine-Tuning (LSTM):", min_value=1, value=max(1, active_config.get('n_splits_cv', 3) -1), step=1, key="ft_n_splits_cv", help="Se 1, usa l'ultimo blocco per validazione. Default: ridotto di 1 rispetto al training originale per usare più dati recenti per il tuning.")
            
        with c2_ft:
            lr_ft = st.number_input("Learning Rate (Fine-Tuning):", min_value=1e-7, value=active_config.get('learning_rate', 0.001) * 0.1, format="%.6f", step=1e-5, key="ft_lr", help="Default: 10% del learning rate originale.")
            bs_ft = st.select_slider("Batch Size (Fine-Tuning):", options=[8, 16, 32, 64, 128], value=active_config.get('batch_size', 32), key="ft_batch_size")

        with c3_ft:
            original_loss_ft = active_config.get('loss_function', 'MSELoss')
            try:
                loss_default_idx_ft = ["MSELoss", "HuberLoss"].index(original_loss_ft)
            except ValueError:
                loss_default_idx_ft = 0 # Default to MSELoss if original not found
            loss_choice_ft = st.selectbox("Funzione di Loss (Fine-Tuning):", ["MSELoss", "HuberLoss"], 
                                          index=loss_default_idx_ft, 
                                          key="ft_loss_choice")
            device_option_ft = st.radio("Device Fine-Tuning:", ['Auto (GPU se disponibile)', 'Forza CPU'], index=0, key='ft_device_select', horizontal=True)
            save_choice_ft = st.radio("Strategia Salvataggio (Fine-Tuning):", ['Migliore (su Validazione)', 'Modello Finale'], index=0, key='ft_save_strategy', horizontal=True)

    if st.button("Avvia Post-Training LSTM", key="ft_run_button"):
        st.info(f"Avvio Post-Training (Fine-Tuning) per il modello: {active_config.get('display_name', 'N/A')}")
        st.info(f"Nuovi dati: {'Caricati' if new_data_file_ft else 'Non caricati'}")
        st.info(f"Nuovo nome modello (base per file): {save_name_ft}")
        st.json({
            "epochs_fine_tuning": ep_ft, 
            "learning_rate_fine_tuning": lr_ft, 
            "batch_size_fine_tuning": bs_ft,
            "loss_function_fine_tuning": loss_choice_ft, 
            "device_fine_tuning": device_option_ft,
            "save_strategy_fine_tuning": save_choice_ft, 
            "n_splits_cv_fine_tuning": n_splits_cv_ft
        })
        st.warning("Logica di fine-tuning e gestione dati non ancora implementata in questo pulsante.")


# --- Footer ---
st.sidebar.divider()
st.sidebar.caption(f'Modello Predittivo Idrologico © {datetime.now().year}')
