import streamlit as st
import torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Non più usato
# import seaborn as sns # Non più usato
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
# import math # Non più usato
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go
import time
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

# Configurazione della pagina
st.set_page_config(page_title="Modello Predittivo Idrologico Seq2Seq", page_icon="🌊", layout="wide")

# --- Costanti ---
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
] # AGGIUNGI QUI ALTRE COLONNE GSheet SE NECESSARIE AL MODELLO
DASHBOARD_REFRESH_INTERVAL_SECONDS = 300
DASHBOARD_HISTORY_ROWS = 48 # Modificato a 48 (24 ore) per coerenza con input window comune
DEFAULT_THRESHOLDS = {
    'Arcevia - Pioggia Ora (mm)': 10.0, 'Barbara - Pioggia Ora (mm)': 10.0, 'Corinaldo - Pioggia Ora (mm)': 10.0,
    'Misa - Pioggia Ora (mm)': 10.0, 'Umidita\' Sensore 3452 (Montemurello)': 95.0, # Soglia Umidità (esempio)
    'Serra dei Conti - Livello Misa (mt)': 2.5, 'Pianello di Ostra - Livello Misa (m)': 3.0,
    'Nevola - Livello Nevola (mt)': 2.0, 'Misa - Livello Misa (mt)': 2.8,
    'Ponte Garibaldi - Livello Misa 2 (mt)': 4.0
}
italy_tz = pytz.timezone('Europe/Rome')
# Colonna specifica umidità/saturazione confermata
HUMIDITY_COL_NAME = "Umidita' Sensore 3452 (Montemurello)"

STATION_COORDS = {
    # ... (invariato, assicurati che HUMIDITY_COL_NAME sia presente se vuoi la sua etichetta) ...
    'Arcevia - Pioggia Ora (mm)': {'lat': 43.5228, 'lon': 12.9388, 'name': 'Arcevia (Pioggia)', 'type': 'Pioggia', 'location_id': 'Arcevia'},
    'Barbara - Pioggia Ora (mm)': {'lat': 43.5808, 'lon': 13.0277, 'name': 'Barbara (Pioggia)', 'type': 'Pioggia', 'location_id': 'Barbara'},
    'Corinaldo - Pioggia Ora (mm)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Pioggia)', 'type': 'Pioggia', 'location_id': 'Corinaldo'},
    'Nevola - Livello Nevola (mt)': {'lat': 43.6491, 'lon': 13.0476, 'name': 'Corinaldo (Livello Nevola)', 'type': 'Livello', 'location_id': 'Corinaldo'}, # Stessa Loc
    'Misa - Pioggia Ora (mm)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Pioggia)', 'type': 'Pioggia', 'location_id': 'Bettolelle'}, # Assunzione Bettolelle
    'Misa - Livello Misa (mt)': {'lat': 43.690, 'lon': 13.165, 'name': 'Bettolelle (Livello Misa)', 'type': 'Livello', 'location_id': 'Bettolelle'}, # Stessa Loc, Assunzione Bettolelle
    'Serra dei Conti - Livello Misa (mt)': {'lat': 43.5427, 'lon': 13.0389, 'name': 'Serra de\' Conti (Livello)', 'type': 'Livello', 'location_id': 'Serra de Conti'},
    'Pianello di Ostra - Livello Misa (m)': {'lat': 43.660, 'lon': 13.135, 'name': 'Pianello di Ostra (Livello)', 'type': 'Livello', 'location_id': 'Pianello Ostra'}, # Coordinate indicative
    'Ponte Garibaldi - Livello Misa 2 (mt)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'}, # Coordinate indicative Ponte Garibaldi Senigallia
    HUMIDITY_COL_NAME: {'lat': 43.6, 'lon': 13.0, 'name': 'Montemurello (Umidità)', 'type': 'Umidità', 'location_id': 'Montemurello'} # Aggiungi coordinate indicative
}

# --- Definizioni Funzioni Core ML ---

# Dataset modificato per Seq2Seq (gestisce 1 o 3 tensor)
class TimeSeriesDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tuple(torch.tensor(t, dtype=torch.float32) for t in tensors)

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

# Modello LSTM Standard (Invariato)
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
        # Usa output dell'ultimo step della sequenza input per predire il futuro
        out = out[:, -1, :]
        out = self.fc(out)
        # Reshape in (batch, output_window_steps, output_size)
        out = out.view(out.size(0), self.output_window, self.output_size)
        return out

# --- NUOVI MODELLI SEQ2SEQ ---
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        # x shape: (batch, input_window, input_size)
        _, (hidden, cell) = self.lstm(x) # Ignora outputs, prendi solo stati
        # hidden shape: (num_layers, batch, hidden_size)
        # cell shape: (num_layers, batch, hidden_size)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        # LSTM input: features previste per lo step corrente + output precedente? (Opzionale, per ora no)
        self.lstm = nn.LSTM(forecast_input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size) # Prevede 1 step alla volta

    def forward(self, x_forecast_step, hidden, cell):
        # x_forecast_step shape: (batch, 1, forecast_input_size) - Solo lo step corrente
        # hidden, cell from encoder or previous decoder step
        output, (hidden, cell) = self.lstm(x_forecast_step, (hidden, cell))
        # output shape: (batch, 1, hidden_size)
        prediction = self.fc(output.squeeze(1)) # Shape: (batch, output_size)
        return prediction, hidden, cell

class Seq2SeqHydro(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps # Numero di *passi* futuri (30 min) da prevedere
        self.device = device

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0): # Teacher forcing di default è 0 per predizione
        # x_past shape: (batch, input_window, past_input_size)
        # x_future_forecast shape: (batch, forecast_window, forecast_input_size)
        # forecast_window DEVE essere >= output_window per questo forward semplice

        batch_size = x_past.shape[0]
        forecast_window = x_future_forecast.shape[1]

        # Verifica che abbiamo abbastanza dati forecast
        if forecast_window < self.output_window:
             # Potremmo gestire questo con padding o ripetendo l'ultimo forecast,
             # ma per ora lanciamo un errore o un warning.
             # Qui potremmo ripetere l'ultimo forecast per semplicità
             missing_steps = self.output_window - forecast_window
             last_forecast_step = x_future_forecast[:, -1:, :] # Shape (batch, 1, forecast_input_size)
             padding = last_forecast_step.repeat(1, missing_steps, 1)
             x_future_forecast = torch.cat([x_future_forecast, padding], dim=1)
             print(f"Warning: forecast_window ({forecast_window}) < output_window ({self.output_window}). Padding forecast input.")
             # Aggiorna forecast_window per il loop sotto
             forecast_window = self.output_window


        target_output_size = self.decoder.output_size # Dal layer FC del decoder

        # Tensor per salvare gli output del decoder
        outputs = torch.zeros(batch_size, self.output_window, target_output_size).to(self.device)

        # 1. Passaggio Encoder
        encoder_hidden, encoder_cell = self.encoder(x_past)

        # 2. Passaggio Decoder (step-by-step)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Input iniziale per il decoder: primo step del forecast futuro
        decoder_input_step = x_future_forecast[:, 0:1, :] # Shape (batch, 1, forecast_input_size)

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell
            )
            outputs[:, t, :] = decoder_output_step

            # Determina l'input per il prossimo step del decoder
            # Durante la predizione (teacher_forcing=0), usiamo sempre il forecast step successivo
            if t < self.output_window - 1:
                # Prendi il forecast corretto per il passo t+1
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]

        # outputs shape: (batch, output_window, target_output_size)
        return outputs

# --- Funzioni Utilità Modello/Dati ---

# Funzione preparazione dati LSTM standard (INVARIATA, serve per compatibilità)
def prepare_training_data(df, feature_columns, target_columns, input_window, output_window, val_split=20):
    # ... (Codice invariato, ma aggiungiamo return None per scaler se preparazione fallisce)
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati LSTM Standard...")
    try:
        # ... (controlli colonne) ...
        missing_features = [col for col in feature_columns if col not in df.columns]
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_features:
            st.error(f"Errore: Feature columns mancanti nel DataFrame: {missing_features}")
            return None, None, None, None, None, None
        if missing_targets:
            st.error(f"Errore: Target columns mancanti nel DataFrame: {missing_targets}")
            return None, None, None, None, None, None

        if df[feature_columns + target_columns].isnull().any().any():
             st.warning("NaN trovati PRIMA della creazione sequenze LSTM standard. Potrebbero esserci problemi.")
             # Considera fillna qui se appropriato
             # df = df.fillna(method='ffill').fillna(method='bfill') # Meglio farlo dopo lo split o usare dati puliti
    except Exception as e:
        st.error(f"Errore controllo colonne in prepare_training_data: {e}")
        return None, None, None, None, None, None # Aggiunto scaler a None

    X, y = [], []
    total_len = len(df)
    # Calcola passi temporali da ore (assumendo 30 min steps)
    input_steps = input_window * 2
    output_steps = output_window * 2 # output_window nel config LSTM è numero di passi!
    required_len = input_steps + output_steps
    if total_len < required_len:
         st.error(f"Dati LSTM insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe per {input_steps} input + {output_steps} output steps).")
         return None, None, None, None, None, None

    print(f"Creazione sequenze LSTM: {total_len - required_len + 1} possibili. Input steps={input_steps}, Output steps={output_steps}")
    for i in range(total_len - required_len + 1):
        X.append(df.iloc[i : i + input_steps][feature_columns].values)
        y.append(df.iloc[i + input_steps : i + required_len][target_columns].values)

    if not X or not y: st.error("Errore creazione sequenze X/y LSTM."); return None, None, None, None, None, None
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)

    # Scaling
    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    if X.size == 0 or y.size == 0: st.error("Dati X o y vuoti prima di scaling LSTM."); return None, None, None, None, None, None
    num_sequences, seq_len_in, num_features = X.shape
    num_sequences_y, seq_len_out, num_targets = y.shape
    if seq_len_in != input_steps or seq_len_out != output_steps:
        st.error(f"Errore shape sequenze LSTM: In={seq_len_in} (atteso {input_steps}), Out={seq_len_out} (atteso {output_steps})")
        return None, None, None, None, None, None

    X_flat = X.reshape(-1, num_features); y_flat = y.reshape(-1, num_targets)
    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except Exception as e_scale: st.error(f"Errore scaling LSTM: {e_scale}"); return None, None, None, None, None, None
    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets)

    # Split
    split_idx = int(len(X_scaled) * (1 - val_split / 100))
    # ... (Gestione split_idx come prima) ...
    if split_idx <= 0 or split_idx >= len(X_scaled):
         st.warning(f"Split indice LSTM ({split_idx}) non valido. Tentativo fallback.")
         if len(X_scaled) < 2: st.error("Dataset LSTM troppo piccolo per split."); return None, None, None, None, None, None
         split_idx = max(1, min(len(X_scaled) - 1, split_idx))

    X_train = X_scaled[:split_idx]; y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]; y_val = y_scaled[split_idx:]

    if X_train.size == 0 or y_train.size == 0: st.error("Set Training LSTM vuoto."); return None, None, None, None, None, None
    if X_val.size == 0 or y_val.size == 0:
         st.warning("Set Validazione LSTM vuoto (split 0% o pochi dati).")
         X_val = np.empty((0, seq_len_in, num_features), dtype=np.float32)
         y_val = np.empty((0, seq_len_out, num_targets), dtype=np.float32)

    print(f"Dati LSTM pronti: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

# --- NUOVA Funzione Preparazione Dati Seq2Seq ---
def prepare_training_data_seq2seq(df, past_feature_cols, forecast_feature_cols, target_cols,
                                 input_window_steps, forecast_window_steps, output_window_steps, val_split=20):
    """Prepara dati per modello Seq2Seq con 3 scaler separati."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq...")
    print(f"Input Steps: {input_window_steps}, Forecast Steps: {forecast_window_steps}, Output Steps: {output_window_steps}")
    print(f"Past Features ({len(past_feature_cols)}): {past_feature_cols[:3]}...")
    print(f"Forecast Features ({len(forecast_feature_cols)}): {forecast_feature_cols}")
    print(f"Target Features ({len(target_cols)}): {target_cols}")

    all_needed_cols = list(set(past_feature_cols + forecast_feature_cols + target_cols))
    try:
        for col in all_needed_cols:
            if col not in df.columns: raise ValueError(f"Colonna '{col}' richiesta per Seq2Seq non trovata.")
        # Controlla NaN PRIMA di creare sequenze
        if df[all_needed_cols].isnull().any().any():
             st.warning("NaN trovati PRIMA della creazione sequenze Seq2Seq. Applico ffill/bfill sull'intero DataFrame.")
             # Applica fill sull'intera copia per evitare data leakage nello split
             df_filled = df.copy()
             df_filled[all_needed_cols] = df_filled[all_needed_cols].fillna(method='ffill').fillna(method='bfill')
             if df_filled[all_needed_cols].isnull().any().any():
                  st.error("NaN RESIDUI dopo fillna. Controlla colonne completamente vuote.")
                  return None, None, None, None, None, None, None, None, None # 9 return values
             df_to_use = df_filled
        else:
            df_to_use = df # Usa l'originale se non ci sono NaN

    except ValueError as e:
        st.error(f"Errore colonne in prepare_seq2seq: {e}")
        return None, None, None, None, None, None, None, None, None

    X_encoder, X_decoder, y_target = [], [], []
    total_len = len(df_to_use)
    # Lunghezza totale necessaria per un campione
    required_len = input_window_steps + max(forecast_window_steps, output_window_steps)

    if total_len < required_len:
        st.error(f"Dati Seq2Seq insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
        return None, None, None, None, None, None, None, None, None

    print(f"Creazione sequenze Seq2Seq: {total_len - required_len + 1} possibili...")
    for i in range(total_len - required_len + 1):
        # Dati passati per l'encoder
        enc_end = i + input_window_steps
        X_encoder.append(df_to_use.iloc[i : enc_end][past_feature_cols].values)

        # Dati futuri per il decoder (usiamo dati reali come forecast target)
        dec_start = enc_end
        dec_end_forecast = dec_start + forecast_window_steps
        X_decoder.append(df_to_use.iloc[dec_start : dec_end_forecast][forecast_feature_cols].values)

        # Dati target reali da prevedere
        target_end = dec_start + output_window_steps
        y_target.append(df_to_use.iloc[dec_start : target_end][target_cols].values)

    if not X_encoder or not X_decoder or not y_target:
        st.error("Errore creazione sequenze X_enc/X_dec/Y_target Seq2Seq.")
        return None, None, None, None, None, None, None, None, None

    # Converti in NumPy
    X_encoder = np.array(X_encoder, dtype=np.float32)
    X_decoder = np.array(X_decoder, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    print(f"Shapes NumPy: Encoder={X_encoder.shape}, Decoder={X_decoder.shape}, Target={y_target.shape}")

    # Validazione Shapes
    if X_encoder.shape[1] != input_window_steps or X_decoder.shape[1] != forecast_window_steps or y_target.shape[1] != output_window_steps:
         st.error("Errore shape sequenze Seq2Seq dopo creazione.")
         return None, None, None, None, None, None, None, None, None
    if X_encoder.shape[2] != len(past_feature_cols) or X_decoder.shape[2] != len(forecast_feature_cols) or y_target.shape[2] != len(target_cols):
         st.error("Errore numero feature/target nelle sequenze Seq2Seq.")
         return None, None, None, None, None, None, None, None, None


    # Scaling con 3 scaler separati
    scaler_past_features = MinMaxScaler()
    scaler_forecast_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    # Reshape per scaling
    num_sequences = X_encoder.shape[0]
    X_enc_flat = X_encoder.reshape(-1, len(past_feature_cols))
    X_dec_flat = X_decoder.reshape(-1, len(forecast_feature_cols))
    y_tar_flat = y_target.reshape(-1, len(target_cols))

    try:
        X_enc_scaled_flat = scaler_past_features.fit_transform(X_enc_flat)
        X_dec_scaled_flat = scaler_forecast_features.fit_transform(X_dec_flat)
        y_tar_scaled_flat = scaler_targets.fit_transform(y_tar_flat)
    except Exception as e_scale:
        st.error(f"Errore scaling Seq2Seq: {e_scale}")
        return None, None, None, None, None, None, None, None, None

    # Reshape back
    X_enc_scaled = X_enc_scaled_flat.reshape(num_sequences, input_window_steps, len(past_feature_cols))
    X_dec_scaled = X_dec_scaled_flat.reshape(num_sequences, forecast_window_steps, len(forecast_feature_cols))
    y_tar_scaled = y_tar_scaled_flat.reshape(num_sequences, output_window_steps, len(target_cols))

    # Split Train/Validation
    split_idx = int(num_sequences * (1 - val_split / 100))
    # ... (Gestione split_idx come prima) ...
    if split_idx <= 0 or split_idx >= num_sequences:
         st.warning(f"Split indice Seq2Seq ({split_idx}) non valido. Tentativo fallback.")
         if num_sequences < 2: st.error("Dataset Seq2Seq troppo piccolo per split."); return None, None, None, None, None, None, None, None, None
         split_idx = max(1, min(num_sequences - 1, split_idx))


    X_enc_train, X_dec_train, y_tar_train = X_enc_scaled[:split_idx], X_dec_scaled[:split_idx], y_tar_scaled[:split_idx]
    X_enc_val, X_dec_val, y_tar_val = X_enc_scaled[split_idx:], X_dec_scaled[split_idx:], y_tar_scaled[split_idx:]

    # Controlli finali su set vuoti
    if X_enc_train.size == 0 or X_dec_train.size == 0 or y_tar_train.size == 0:
        st.error("Set Training Seq2Seq vuoto dopo split.")
        return None, None, None, None, None, None, None, None, None
    if X_enc_val.size == 0 or X_dec_val.size == 0 or y_tar_val.size == 0:
        st.warning("Set Validazione Seq2Seq vuoto (split 0% o pochi dati).")
        # Crea array vuoti con shape corretta per evitare errori downstream
        X_enc_val = np.empty((0, input_window_steps, len(past_feature_cols)), dtype=np.float32)
        X_dec_val = np.empty((0, forecast_window_steps, len(forecast_feature_cols)), dtype=np.float32)
        y_tar_val = np.empty((0, output_window_steps, len(target_cols)), dtype=np.float32)

    print(f"Dati Seq2Seq pronti: Train={len(X_enc_train)}, Val={len(X_enc_val)}")
    return (X_enc_train, X_dec_train, y_tar_train,
            X_enc_val, X_dec_val, y_tar_val,
            scaler_past_features, scaler_forecast_features, scaler_targets)


# --- Funzioni Caricamento Modello/Scaler (Aggiornate) ---

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

        if not os.path.exists(cfg_path): continue # Manca config, ignora

        model_info = {"config_name": base, "pth_path": pth_path, "config_path": cfg_path}
        model_type = "LSTM" # Default
        valid_model = False

        try:
            with open(cfg_path, 'r') as f: config_data = json.load(f)
            name = config_data.get("display_name", base)
            model_type = config_data.get("model_type", "LSTM") # Legge il tipo dal config

            # Verifica file necessari in base al tipo
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
            else: # Assume LSTM standard
                required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout",
                                 "feature_columns", "target_columns"] # feature_columns può mancare e usare globali
                scf_p = os.path.join(models_dir, f"{base}_features.joblib")
                sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys[:6] + ["target_columns"]) and # feature_columns opzionale
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
            available[name] = model_info # Usa display_name come chiave
        else:
            # Commenta questo warning se troppo verboso per file incompleti
            # st.warning(f"Modello '{base}' ignorato: file associati mancanti o config JSON incompleta per tipo '{model_type}'.")
            pass

    return available

@st.cache_data
def load_model_config(_config_path):
    """Carica config e valida chiavi base."""
    try:
        with open(_config_path, 'r') as f: config = json.load(f)
        # Non validiamo tutte le chiavi qui, dipende dal tipo che verrà controllato dopo
        return config
    except Exception as e: st.error(f"Errore caricamento config '{_config_path}': {e}"); return None

@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(_model_path, config):
    """Carica modello Pytorch (LSTM o Seq2Seq) in base al config."""
    if not config: st.error("Config non valida per caricamento modello."); return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config.get("model_type", "LSTM") # Determina tipo

    try:
        model = None
        if model_type == "Seq2Seq":
            # Istanzia Encoder, Decoder, Seq2Seq
            enc_input_size = len(config["all_past_feature_columns"])
            dec_input_size = len(config["forecast_input_columns"])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]
            layers = config["num_layers"]
            drop = config["dropout"]
            out_win = config["output_window_steps"]

            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            decoder = DecoderLSTM(dec_input_size, hidden, dec_output_size, layers, drop).to(device)
            model = Seq2SeqHydro(encoder, decoder, out_win, device).to(device)
            # Nota: Qui carichiamo lo state_dict dell'intero Seq2SeqHydro
        else: # Assume LSTM standard
            # Gestione feature_columns per LSTM (se mancano, usa globali)
            f_cols_lstm = config.get("feature_columns")
            if not f_cols_lstm:
                 st.warning(f"Config LSTM '{config.get('display_name', 'N/A')}' non specifica feature_columns. Uso le globali.")
                 f_cols_lstm = st.session_state.get("feature_columns", []) # Usa quelle globali
                 if not f_cols_lstm: raise ValueError("Feature globali non definite.")
                 config["feature_columns"] = f_cols_lstm # Aggiorna config in memoria

            input_size_lstm = len(f_cols_lstm)
            target_size_lstm = len(config["target_columns"])
            # output_window in config LSTM è già il numero di steps
            out_win_lstm = config["output_window"]

            model = HydroLSTM(input_size_lstm, config["hidden_size"], target_size_lstm,
                              out_win_lstm, config["num_layers"], config["dropout"]).to(device)

        # Carica state_dict
        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'): # File caricato
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
    """Carica 1, 2 o 3 scaler in base al tipo di modello nel config."""
    if not config or not model_info: st.error("Config o info modello mancanti per caricare scaler."); return None
    model_type = config.get("model_type", "LSTM")

    def _load_joblib(path):
         # Helper interno per caricare da path o file object
         if isinstance(path, str):
              if not os.path.exists(path): raise FileNotFoundError(f"File scaler '{path}' non trovato.")
              return joblib.load(path)
         elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
         else: raise TypeError(f"Percorso scaler non valido: {type(path)}")

    try:
        if model_type == "Seq2Seq":
            sp_path = model_info["scaler_past_features_path"]
            sf_path = model_info["scaler_forecast_features_path"]
            st_path = model_info["scaler_targets_path"]
            scaler_past = _load_joblib(sp_path)
            scaler_forecast = _load_joblib(sf_path)
            scaler_targets = _load_joblib(st_path)
            print(f"Scaler Seq2Seq caricati.")
            # Restituisce una tupla o un dizionario per chiarezza
            return {"past": scaler_past, "forecast": scaler_forecast, "targets": scaler_targets}
        else: # Assume LSTM
            sf_path = model_info["scaler_features_path"]
            st_path = model_info["scaler_targets_path"]
            scaler_features = _load_joblib(sf_path)
            scaler_targets = _load_joblib(st_path)
            print(f"Scaler LSTM caricati.")
            # Restituisce come prima per compatibilità
            return scaler_features, scaler_targets

    except Exception as e:
        st.error(f"Errore caricamento scaler (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc())
        if model_type == "Seq2Seq": return None # Singolo None per fallimento
        else: return None, None # Due None per LSTM

# --- Funzioni Predict (Standard e Seq2Seq) ---

# Predict LSTM Standard (leggermente adattata per robustezza)
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    """Esegue predizione per modello LSTM standard."""
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict LSTM: Modello, scaler o config mancanti."); return None

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM": st.error(f"Funzione predict chiamata su modello non LSTM (tipo: {model_type})"); return None

    # Input window in config LSTM è numero di steps
    input_steps = config["input_window"]
    # Output window in config LSTM è numero di steps
    output_steps = config["output_window"]
    target_cols = config["target_columns"]
    f_cols_cfg = config.get("feature_columns", []) # Può mancare

    # Verifica shape input
    if input_data.shape[0] != input_steps:
        st.error(f"Predict LSTM: Input righe {input_data.shape[0]} != Steps {input_steps}."); return None
    expected_features = getattr(scaler_features, 'n_features_in_', len(f_cols_cfg) if f_cols_cfg else None)
    if expected_features is not None and input_data.shape[1] != expected_features:
        st.error(f"Predict LSTM: Input colonne {input_data.shape[1]} != Features attese {expected_features}."); return None
    elif expected_features is None:
        st.warning("Predict LSTM: Impossibile verificare numero colonne input (scaler non fittato?).")
        # Potrebbe essere un problema, ma proviamo comunque
        # return None # Forse più sicuro restituire None qui

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device) # Aggiunge batch dim
        with torch.no_grad(): output = model(inp_tens) # Shape (1, output_steps, num_targets)

        # Verifica shape output
        if output.shape != (1, output_steps, len(target_cols)):
             st.error(f"Predict LSTM: Shape output modello {output.shape} inattesa (atteso (1, {output_steps}, {len(target_cols)})).")
             return None

        out_np = output.cpu().numpy().squeeze(0) # Rimuove batch dim -> (output_steps, num_targets)

        # Verifica scaler target
        expected_targets = getattr(scaler_targets, 'n_features_in_', None)
        if expected_targets is None: st.error("Predict LSTM: Scaler targets non fittato."); return None
        if expected_targets != len(target_cols): st.error(f"Predict LSTM: Targets modello ({len(target_cols)}) != Scaler Targets ({expected_targets})."); return None

        preds = scaler_targets.inverse_transform(out_np)
        return preds # Shape (output_steps, num_targets)

    except Exception as e:
        st.error(f"Errore durante predict LSTM: {e}")
        st.error(traceback.format_exc()); return None

# --- NUOVA Funzione Predict Seq2Seq ---
def predict_seq2seq(model, past_data, future_forecast_data, scalers, config, device):
    """Esegue predizione per modello Seq2Seq."""
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]):
         st.error("Predict Seq2Seq: Input mancanti (modello, dati, scalers, config, device)."); return None

    model_type = config.get("model_type")
    if model_type != "Seq2Seq": st.error(f"Funzione predict_seq2seq chiamata su modello non Seq2Seq (tipo: {model_type})"); return None

    # Recupera dimensioni e colonne dal config
    input_steps = config["input_window_steps"]
    forecast_steps = config["forecast_window_steps"] # Steps di forecast forniti
    output_steps = config["output_window_steps"] # Steps da prevedere
    past_cols = config["all_past_feature_columns"]
    forecast_cols = config["forecast_input_columns"]
    target_cols = config["target_columns"]

    # Recupera scaler
    scaler_past = scalers.get("past")
    scaler_forecast = scalers.get("forecast")
    scaler_targets = scalers.get("targets")
    if not all([scaler_past, scaler_forecast, scaler_targets]):
        st.error("Predict Seq2Seq: Scaler mancanti nel dizionario fornito."); return None

    # Verifica shape input dati REALI passati
    if past_data.shape != (input_steps, len(past_cols)):
        st.error(f"Predict Seq2Seq: Shape dati passati {past_data.shape} != Atteso ({input_steps}, {len(past_cols)})."); return None
    # Verifica shape input dati FORECAST futuri
    if future_forecast_data.shape != (forecast_steps, len(forecast_cols)):
        st.error(f"Predict Seq2Seq: Shape dati forecast {future_forecast_data.shape} != Atteso ({forecast_steps}, {len(forecast_cols)})."); return None
    # Verifica se forecast è sufficiente per output richiesto (anche se il modello può gestire padding)
    if forecast_steps < output_steps:
         st.warning(f"Predict Seq2Seq: La finestra forecast fornita ({forecast_steps} steps) è minore della finestra output richiesta ({output_steps} steps). Il modello userà l'ultimo forecast per i passi mancanti.")

    model.eval()
    try:
        # Scala input
        past_norm = scaler_past.transform(past_data)
        future_norm = scaler_forecast.transform(future_forecast_data)

        # Converti in Tensor e aggiungi batch dim
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device)
        future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)

        # Esegui predizione
        with torch.no_grad():
            output = model(past_tens, future_tens) # Shape (1, output_steps, num_targets)

        # Verifica shape output
        if output.shape != (1, output_steps, len(target_cols)):
             st.error(f"Predict Seq2Seq: Shape output modello {output.shape} inattesa (atteso (1, {output_steps}, {len(target_cols)})).")
             return None

        out_np = output.cpu().numpy().squeeze(0) # Rimuove batch dim -> (output_steps, num_targets)

        # Verifica scaler target
        expected_targets = getattr(scaler_targets, 'n_features_in_', None)
        if expected_targets is None: st.error("Predict Seq2Seq: Scaler targets non fittato."); return None
        if expected_targets != len(target_cols): st.error(f"Predict Seq2Seq: Targets modello ({len(target_cols)}) != Scaler Targets ({expected_targets})."); return None

        # Denormalizza output
        preds = scaler_targets.inverse_transform(out_np)
        return preds # Shape (output_steps, num_targets)

    except Exception as e:
        st.error(f"Errore durante predict Seq2Seq: {e}")
        st.error(traceback.format_exc()); return None


# --- plot_predictions (Adattata leggermente per chiarezza asse X) ---
def plot_predictions(predictions, config, start_time=None):
    """Genera grafici Plotly per le previsioni (LSTM o Seq2Seq)."""
    if config is None or predictions is None: return []

    model_type = config.get("model_type", "LSTM")
    target_cols = config["target_columns"]
    # Determina il numero di passi previsti dalla shape dell'output
    output_steps = predictions.shape[0]
    # Calcola ore totali previste (ogni step = 30 min)
    total_hours_predicted = output_steps * 0.5

    figs = []
    for i, sensor in enumerate(target_cols):
        fig = go.Figure()
        # Genera timestamp futuri (ogni 30 min)
        if start_time:
            time_steps = [start_time + timedelta(minutes=30 * (step + 1)) for step in range(output_steps)]
            x_axis, x_title = time_steps, "Data e Ora Previste"
        else: # Fallback a passi numerici se start_time non fornito
            time_steps = np.arange(1, output_steps + 1) * 0.5 # Ore relative
            x_axis, x_title = time_steps, "Ore Future (passi da 30 min)"

        station_name_graph = get_station_label(sensor, short=False) # Usa la funzione helper

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsto'))

        # Estrai unità di misura in modo più robusto
        unit_match = re.search(r'\((.*?)\)', sensor)
        y_axis_unit = unit_match.group(1).strip() if unit_match else "Valore"

        fig.update_layout(
            title=f'Previsione {model_type} - {station_name_graph}', # Titolo aggiornato
            xaxis_title=x_title,
            yaxis_title=y_axis_unit,
            height=400,
            hovermode="x unified"
        )
        # Formatta asse X per date/ore se disponibili
        if start_time:
            fig.update_xaxes(tickformat="%d/%m %H:%M") # Formato desiderato

        fig.update_yaxes(rangemode='tozero')
        figs.append(fig)
    return figs

# --- Funzione Fetch GSheet Dashboard (Invariata) ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format, num_rows_to_fetch=DASHBOARD_HISTORY_ROWS):
    # ... (Codice Invariato) ...
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data (Cache Key: {_cache_key_time}, Rows: {num_rows_to_fetch})") # Debug
    actual_fetch_time = datetime.now(italy_tz)
    try:
        # ... (Logica GSheet API e lettura) ...
        if "GOOGLE_CREDENTIALS" not in st.secrets:
            return None, "Errore: Credenziali Google mancanti.", actual_fetch_time
        # ... (Autorizzazione) ...
        credentials = Credentials.from_service_account_info(
            st.secrets["GOOGLE_CREDENTIALS"],
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        all_values = worksheet.get_all_values()

        if not all_values or len(all_values) < 2:
             return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time

        headers = all_values[0]
        # Prendi le ultime num_rows_to_fetch righe di DATI (escludendo l'header)
        start_index = max(1, len(all_values) - num_rows_to_fetch)
        data_rows = all_values[start_index:]
        # ... (Verifica headers mancanti) ...
        headers_set = set(headers)
        missing_cols = [col for col in relevant_columns if col not in headers_set]
        if missing_cols:
            return None, f"Errore: Colonne GSheet richieste mancanti: {', '.join(missing_cols)}", actual_fetch_time

        df = pd.DataFrame(data_rows, columns=headers)
        df = df[relevant_columns] # Seleziona solo colonne utili

        # ... (Pulizia dati e conversione tipi, gestione date con timezone) ...
        error_parsing = []
        for col in relevant_columns: # Ora itera solo sulle colonne rilevanti
            if col == date_col:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    if df[col].isnull().any(): error_parsing.append(f"Formato data non valido per '{col}'")
                    # Localizza o converti a italy_tz
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                    else:
                        df[col] = df[col].dt.tz_convert(italy_tz)
                except Exception as e_date: error_parsing.append(f"Errore data '{col}': {e_date}"); df[col] = pd.NaT
            else: # Colonne numeriche
                try:
                    df_col_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                    df[col] = df_col_str.replace(['N/A', '', '-', ' ', 'None', 'null'], np.nan, regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Non serve warning per NaN qui, gestito dopo
                except Exception as e_num: error_parsing.append(f"Errore numerico '{col}': {e_num}"); df[col] = np.nan

        df = df.sort_values(by=date_col, na_position='first').reset_index(drop=True)
        # Gestione NaN (non fare ffill qui, meglio vederli)
        nan_numeric_count = df.drop(columns=[date_col]).isnull().sum().sum()
        if nan_numeric_count > 0:
             st.info(f"Nota Dashboard: Trovati {nan_numeric_count} valori mancanti/non numerici nelle ultime {num_rows_to_fetch} righe GSheet.")

        error_message = "Attenzione conversione dati GSheet: " + " | ".join(error_parsing) if error_parsing else None
        return df, error_message, actual_fetch_time

    except gspread.exceptions.APIError as api_e: # ... (Gestione errori API come prima) ...
        try: error_details = api_e.response.json(); error_message = error_details.get('error', {}).get('message', str(api_e)); status_code = error_details.get('error', {}).get('code', 'N/A'); error_message = f"Codice {status_code}: {error_message}";
        except: error_message = str(api_e)
        return None, f"Errore API Google Sheets: {error_message}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}').", actual_fetch_time
    except Exception as e: return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time


# --- Funzione Fetch GSheet Simulazione (Adattata leggermente per più robustezza) ---
# Questa funzione è usata da ENTRAMBI i metodi di simulazione che partono da GSheet
@st.cache_data(ttl=120, show_spinner="Importazione dati storici da Google Sheet per simulazione...")
def fetch_sim_gsheet_data(sheet_id_fetch, n_rows_steps, date_col_gs, date_format_gs, col_mapping, required_model_cols_fetch, impute_dict):
    """ Recupera, mappa, pulisce e ordina n_rows_steps di dati da GSheet per simulazione. """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_sim_gsheet_data (Steps: {n_rows_steps})")
    actual_fetch_time = datetime.now(italy_tz) # Tempo per eventuale timestamp
    last_valid_timestamp = None
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets: return None, "Errore: Credenziali Google mancanti.", None
        # ... (Autorizzazione GSpread) ...
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id_fetch)
        worksheet = sh.sheet1
        all_data_gs = worksheet.get_all_values()

        if not all_data_gs or len(all_data_gs) < 2 : # Almeno header e 1 riga dati
            return None, f"Errore: Foglio GSheet vuoto o senza dati.", None

        headers_gs = all_data_gs[0]
        # Prendi le ultime n righe di DATI
        start_index_gs = max(1, len(all_data_gs) - n_rows_steps)
        data_rows_gs = all_data_gs[start_index_gs:]

        if len(data_rows_gs) < n_rows_steps:
            return None, f"Errore: Dati GSheet insufficienti (trovate {len(data_rows_gs)} righe dati, richieste {n_rows_steps}).", None


        df_gsheet_raw = pd.DataFrame(data_rows_gs, columns=headers_gs)

        # Verifica colonne GSheet richieste dal MAPPING
        required_gsheet_cols_from_mapping = list(col_mapping.keys()) + ([date_col_gs] if date_col_gs not in col_mapping.keys() else []) # Aggiungi colonna data se non mappata esplicitamente
        required_gsheet_cols_from_mapping = list(set(required_gsheet_cols_from_mapping)) # Rimuovi duplicati

        missing_gsheet_cols_in_sheet = [c for c in required_gsheet_cols_from_mapping if c not in df_gsheet_raw.columns]
        if missing_gsheet_cols_in_sheet:
            return None, f"Errore: Colonne GSheet ({', '.join(missing_gsheet_cols_in_sheet)}) mancanti nel foglio.", None

        # Seleziona colonne GSheet necessarie (mappate + data) e rinomina a nomi modello
        df_subset = df_gsheet_raw[required_gsheet_cols_from_mapping].copy()
        df_mapped = df_subset.rename(columns=col_mapping) # Rinomina DOPO la selezione

        # Aggiungi colonne mancanti nel modello con valori costanti (imputazione)
        for model_col, impute_val in impute_dict.items():
             if model_col not in df_mapped.columns:
                 df_mapped[model_col] = impute_val

        # Verifica che TUTTE le colonne modello richieste siano presenti ORA
        final_missing_model_cols = [c for c in required_model_cols_fetch if c not in df_mapped.columns]
        if final_missing_model_cols:
            return None, f"Errore: Colonne modello ({', '.join(final_missing_model_cols)}) mancanti dopo mappatura/imputazione.", None

        # --- Pulizia Dati ---
        # Trova il nome della colonna data DOPO la mappatura
        date_col_model_name = col_mapping.get(date_col_gs, date_col_gs) # Usa nome mappato se esiste, altrimenti originale
        if date_col_model_name not in df_mapped.columns:
             st.warning(f"Colonna data '{date_col_model_name}' non presente dopo mappatura, impossibile ordinare o usare timestamp.")
             date_col_model_name = None # Non usare per ordinamento

        for col in required_model_cols_fetch: # Itera su tutte le feature modello richieste
            if col == date_col_model_name: continue # Salta pulizia numerica per data

            if col not in df_mapped.columns: continue # Salta se colonna non esiste (dovrebbe essere stata imputata)

            # Pulizia colonne numeriche
            try:
                if pd.api.types.is_object_dtype(df_mapped[col]) or pd.api.types.is_string_dtype(df_mapped[col]):
                    col_str = df_mapped[col].astype(str)
                    # Pulisci virgole, spazi, N/A etc.
                    col_str = col_str.str.replace(',', '.', regex=False).str.strip()
                    df_mapped[col] = col_str.replace(['N/A', '', '-', ' ', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                # Converti in numerico, forzando errori a NaN
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
            except Exception as e_clean_num:
                st.warning(f"Problema pulizia GSheet colonna '{col}' per simulazione: {e_clean_num}. Verrà trattata come NaN.")
                df_mapped[col] = np.nan

        # Gestione e ordinamento data (se possibile)
        if date_col_model_name:
             try:
                 df_mapped[date_col_model_name] = pd.to_datetime(df_mapped[date_col_model_name], format=date_format_gs, errors='coerce')
                 if df_mapped[date_col_model_name].isnull().any(): st.warning(f"Date non valide trovate in GSheet simulazione ('{date_col_model_name}').")
                 # Localizza/Converti timezone
                 if df_mapped[date_col_model_name].dt.tz is None:
                     df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_localize(italy_tz, ambiguous='infer', nonexistent='shift_forward')
                 else:
                     df_mapped[date_col_model_name] = df_mapped[date_col_model_name].dt.tz_convert(italy_tz)

                 df_mapped = df_mapped.sort_values(by=date_col_model_name, na_position='first')
                 if not df_mapped[date_col_model_name].dropna().empty:
                    last_valid_timestamp = df_mapped[date_col_model_name].dropna().iloc[-1]
             except Exception as e_date_clean:
                 st.warning(f"Errore conversione/pulizia data GSheet simulazione '{date_col_model_name}': {e_date_clean}. Impossibile ordinare.")
                 date_col_model_name = None # Non usare più la colonna data

        # Seleziona e Riordina colonne come richiesto dal modello
        try:
            df_final = df_mapped[required_model_cols_fetch]
        except KeyError as e_key:
            return None, f"Errore selezione/ordine colonne finali simulazione: '{e_key}' mancante.", None

        # Gestione NaN residui (applica ffill/bfill DOPO ordinamento se possibile)
        numeric_cols_to_fill = df_final.select_dtypes(include=np.number).columns
        nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
        if nan_count_before > 0:
             st.warning(f"Trovati {nan_count_before} valori NaN nei dati GSheet per simulazione. Applico forward-fill e backward-fill.")
             df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(method='ffill').fillna(method='bfill')
             if df_final[numeric_cols_to_fill].isnull().sum().sum() > 0:
                 # Se rimangono NaN, prova a riempire con 0 come ultima risorsa
                 st.error(f"NaN residui ({df_final[numeric_cols_to_fill].isnull().sum().sum()}) dopo fillna. Tento fill con 0.")
                 df_final.loc[:, numeric_cols_to_fill] = df_final[numeric_cols_to_fill].fillna(0)
                 # return None, "Errore: NaN residui dopo fillna. Controlla dati GSheet.", None

        # Controllo finale numero righe
        if len(df_final) != n_rows_steps:
            st.warning(f"Attenzione: Numero righe finali ({len(df_final)}) diverso da richiesto ({n_rows_steps}) dopo recupero GSheet. Potrebbe mancare storico.")
            # Non bloccare l'esecuzione, ma segnala
            # return None, f"Errore: Numero righe finali ({len(df_final)}) diverso da richiesto ({n_rows_steps}).", None

        print(f"Dati simulazione da GSheet pronti ({len(df_final)} righe). Ultimo Timestamp: {last_valid_timestamp}")
        return df_final, None, last_valid_timestamp # Successo

    except gspread.exceptions.APIError as api_e_sim: # ... (Gestione errori API come prima) ...
        try: error_details = api_e_sim.response.json(); error_message = error_details.get('error', {}).get('message', str(api_e_sim)); status_code = error_details.get('error', {}).get('code', 'N/A'); error_message = f"Codice {status_code}: {error_message}";
        except: error_message = str(api_e_sim)
        return None, f"Errore API Google Sheets: {error_message}", None
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google simulazione non trovato (ID: '{sheet_id_fetch}').", None
    except Exception as e_sim_fetch:
        st.error(traceback.format_exc())
        return None, f"Errore imprevisto importazione GSheet per simulazione: {type(e_sim_fetch).__name__} - {e_sim_fetch}", None

# --- Funzioni Allenamento (Standard e Seq2Seq) ---

# Allenamento LSTM Standard (Aggiornata per robustezza e device)
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window_steps,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2,
                save_strategy='migliore',
                preferred_device='auto'):
    """Allena modello LSTM standard."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training LSTM Standard...")
    # --- Logica Selezione Device (come prima) ---
    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"Training LSTM userà: {device}")

    # Istanzia Modello, Dataset, DataLoader
    model = HydroLSTM(input_size, hidden_size, output_size, output_window_steps, num_layers, dropout).to(device)
    train_dataset = TimeSeriesDataset(X_train, y_train) # Gestisce 1 o 3 tensor
    val_dataset = TimeSeriesDataset(X_val, y_val) if (X_val is not None and X_val.size > 0) else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) if val_dataset else None
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # --- Loop di Allenamento (come prima, con UI updates) ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf'); best_model_state_dict = None
    progress_bar = st.progress(0.0, text="Training LSTM..."); status_text = st.empty(); loss_chart_placeholder = st.empty()
    def update_loss_chart(t_loss, v_loss, placeholder):
        # ... (funzione plot loss come prima) ...
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        valid_v_loss = [v for v in v_loss if v is not None] if v_loss else []
        if valid_v_loss: fig.add_trace(go.Scatter(y=valid_v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss (LSTM)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0))
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training LSTM per {epochs} epoche su **{device}**...")
    start_training_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time(); model.train(); train_loss = 0
        for i, batch_data in enumerate(train_loader): # Dataset ora ritorna tuple
            X_batch, y_batch = batch_data # Assumendo che ritorni (X, y)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch); loss = criterion(outputs, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader); train_losses.append(train_loss)

        val_loss = None
        if val_loader:
            model.eval(); current_val_loss = 0
            with torch.no_grad():
                for batch_data_val in val_loader:
                    X_batch_val, y_batch_val = batch_data_val # Assumendo ritorni (X_val, y_val)
                    X_batch_val, y_batch_val = X_batch_val.to(device), y_batch_val.to(device)
                    outputs_val = model(X_batch_val); loss_val = criterion(outputs_val, y_batch_val)
                    current_val_loss += loss_val.item()
            if len(val_loader) > 0: val_loss = current_val_loss / len(val_loader)
            else: val_loss = float('inf') # Avoid division by zero if val_loader is empty
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: val_losses.append(None)

        # Aggiorna UI
        progress_percentage = (epoch + 1) / epochs
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None and val_loss != float('inf') else "N/A"
        epoch_time = time.time() - epoch_start_time
        status_text.text(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) - Train Loss: {train_loss:.6f}, Val Loss: {val_loss_str} - LR: {current_lr:.6f}')
        progress_bar.progress(progress_percentage, text=f"Training LSTM: Epoca {epoch+1}/{epochs}")
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

    total_training_time = time.time() - start_training_time
    st.write(f"Training LSTM completato in {total_training_time:.1f} secondi.")

    # --- Logica Restituzione Modello (come prima) ---
    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict:
            try:
                best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
                model.load_state_dict(best_model_state_dict_on_device)
                final_model_to_return = model
                st.success(f"Strategia 'migliore': Caricato modello LSTM con Val Loss minima ({best_val_loss:.6f}).")
            except Exception as e_load_best: st.error(f"Errore caricamento stato LSTM migliore: {e_load_best}.")
        elif val_loader: st.warning("Strategia 'migliore' LSTM: Nessun miglioramento Val Loss. Restituito modello finale.")
        else: st.warning("Strategia 'migliore' LSTM: Nessuna validazione. Restituito modello finale.")
    elif save_strategy == 'finale': st.info("Strategia 'finale' LSTM: Restituito modello ultima epoca.")

    return final_model_to_return, train_losses, val_losses


# --- NUOVA Funzione Allenamento Seq2Seq ---
def train_model_seq2seq(X_enc_train, X_dec_train, y_tar_train,
                        X_enc_val, X_dec_val, y_tar_val,
                        encoder, decoder, output_window_steps, # Passa encoder/decoder istanziati
                        epochs=50, batch_size=32, learning_rate=0.001,
                        save_strategy='migliore',
                        preferred_device='auto',
                        teacher_forcing_ratio_schedule=None): # Opzione per programmare teacher forcing
    """Allena modello Seq2Seq."""
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training Seq2Seq...")
    # --- Logica Selezione Device ---
    if preferred_device == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device('cpu')
    print(f"Training Seq2Seq userà: {device}")

    # Istanzia Modello Combinato, Dataset, DataLoader
    model = Seq2SeqHydro(encoder, decoder, output_window_steps, device).to(device)
    train_dataset = TimeSeriesDataset(X_enc_train, X_dec_train, y_tar_train) # 3 Tensor
    val_dataset = TimeSeriesDataset(X_enc_val, X_dec_val, y_tar_val) if (X_enc_val is not None and X_enc_val.size > 0) else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) if val_dataset else None
    criterion = nn.MSELoss()
    # Ottimizza parametri di encoder E decoder
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # --- Loop di Allenamento ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf'); best_model_state_dict = None
    progress_bar = st.progress(0.0, text="Training Seq2Seq..."); status_text = st.empty(); loss_chart_placeholder = st.empty()
    def update_loss_chart_seq2seq(t_loss, v_loss, placeholder): # Funzione plot separata per titolo
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        valid_v_loss = [v for v in v_loss if v is not None] if v_loss else []
        if valid_v_loss: fig.add_trace(go.Scatter(y=valid_v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss (Seq2Seq)', xaxis_title='Epoca', yaxis_title='Loss (MSE)', height=300, margin=dict(t=30, b=0))
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training Seq2Seq per {epochs} epoche su **{device}**...")
    start_training_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time(); model.train(); train_loss = 0
        # Determina teacher forcing ratio per questa epoca (se programmato)
        current_tf_ratio = 0.5 # Default se non programmato (o usa 0?) -> Usiamo 0.5 come esempio
        if teacher_forcing_ratio_schedule:
            # Esempio: Decadimento lineare da 0.6 a 0.1
            current_tf_ratio = max(0.1, 0.6 - (0.5 * epoch / max(1, epochs-1))) # Avoid division by zero

        for i, (x_enc_b, x_dec_b, y_tar_b) in enumerate(train_loader):
            x_enc_b, x_dec_b, y_tar_b = x_enc_b.to(device), x_dec_b.to(device), y_tar_b.to(device)

            optimizer.zero_grad()

            # --- Forward Pass Seq2Seq con Teacher Forcing ---
            batch_s = x_enc_b.shape[0]
            out_win = model.output_window
            target_size = model.decoder.output_size
            outputs_train = torch.zeros(batch_s, out_win, target_size).to(device)

            encoder_hidden, encoder_cell = model.encoder(x_enc_b)
            decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
            decoder_input_step = x_dec_b[:, 0:1, :] # Primo input forecast

            for t in range(out_win):
                decoder_output_step, decoder_hidden, decoder_cell = model.decoder(
                    decoder_input_step, decoder_hidden, decoder_cell
                )
                outputs_train[:, t, :] = decoder_output_step

                # Teacher Forcing?
                use_teacher_forcing = random.random() < current_tf_ratio
                if use_teacher_forcing and t < out_win - 1:
                     # Per il prossimo input, usa il VERO dato forecast futuro
                     # (che durante l'allenamento è preso da x_dec_b)
                     decoder_input_step = x_dec_b[:, t+1:t+2, :]
                elif t < out_win - 1:
                     # Altrimenti (no TF), usa il forecast futuro fornito
                     # (durante l'allenamento è sempre x_dec_b)
                     decoder_input_step = x_dec_b[:, t+1:t+2, :]
                     # NOTA: durante la PREDIZIONE vera, qui useremmo il forecast REALE.
                     #       Durante l'allenamento, usiamo sempre x_dec_b che rappresenta
                     #       il "forecast perfetto" (i dati reali futuri).
                     #       Non possiamo usare l'output del decoder come input qui
                     #       perché l'input del decoder sono le FEATURES forecast.

            # --- Fine Forward Pass ---

            loss = criterion(outputs_train, y_tar_b) # Confronta con target reali
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader); train_losses.append(train_loss)

        # --- Validation Loop ---
        val_loss = None
        if val_loader:
            model.eval(); current_val_loss = 0
            with torch.no_grad():
                for x_enc_vb, x_dec_vb, y_tar_vb in val_loader:
                    x_enc_vb, x_dec_vb, y_tar_vb = x_enc_vb.to(device), x_dec_vb.to(device), y_tar_vb.to(device)
                    # Per validazione, MAI teacher forcing
                    outputs_val = model(x_enc_vb, x_dec_vb) # Chiama forward senza TF
                    loss_val = criterion(outputs_val, y_tar_vb)
                    current_val_loss += loss_val.item()
            if len(val_loader) > 0: val_loss = current_val_loss / len(val_loader)
            else: val_loss = float('inf') # Avoid division by zero
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Salva stato dell'intero modello Seq2Seq
                best_model_state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
        else: val_losses.append(None)

        # Aggiorna UI
        progress_percentage = (epoch + 1) / epochs
        current_lr = optimizer.param_groups[0]['lr']
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None and val_loss != float('inf') else "N/A"
        epoch_time = time.time() - epoch_start_time
        status_text.text(f'Epoca {epoch+1}/{epochs} ({epoch_time:.1f}s) - Train Loss: {train_loss:.6f}, Val Loss: {val_loss_str} - LR: {current_lr:.6f}')
        progress_bar.progress(progress_percentage, text=f"Training Seq2Seq: Epoca {epoch+1}/{epochs}")
        update_loss_chart_seq2seq(train_losses, val_losses, loss_chart_placeholder)


    total_training_time = time.time() - start_training_time
    st.write(f"Training Seq2Seq completato in {total_training_time:.1f} secondi.")

    # --- Logica Restituzione Modello (come prima) ---
    final_model_to_return = model
    if save_strategy == 'migliore':
        if best_model_state_dict:
            try:
                best_model_state_dict_on_device = {k: v.to(device) for k, v in best_model_state_dict.items()}
                model.load_state_dict(best_model_state_dict_on_device)
                final_model_to_return = model
                st.success(f"Strategia 'migliore': Caricato modello Seq2Seq con Val Loss minima ({best_val_loss:.6f}).")
            except Exception as e_load_best: st.error(f"Errore caricamento stato Seq2Seq migliore: {e_load_best}.")
        elif val_loader: st.warning("Strategia 'migliore' Seq2Seq: Nessun miglioramento Val Loss. Restituito modello finale.")
        else: st.warning("Strategia 'migliore' Seq2Seq: Nessuna validazione. Restituito modello finale.")
    elif save_strategy == 'finale': st.info("Strategia 'finale' Seq2Seq: Restituito modello ultima epoca.")

    return final_model_to_return, train_losses, val_losses


# --- Funzioni Helper Download (Invariate) ---
def get_table_download_link(df, filename="data.csv"):
    """Genera link download per DataFrame come CSV (UTF-8 con BOM per Excel)."""
    csv_buffer = io.StringIO()
    # Usa encoding utf-8-sig per includere il BOM
    df.to_csv(csv_buffer, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    csv_buffer.seek(0)
    b64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8-sig')).decode() # Encode come utf-8-sig
    return f'<a href="data:text/csv;charset=utf-8-sig;base64,{b64}" download="{filename}">Scarica CSV</a>'

def get_binary_file_download_link(file_object, filename, text):
    """Genera link download per oggetto file binario (es: BytesIO)."""
    file_object.seek(0); b64 = base64.b64encode(file_object.getvalue()).decode()
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None: mime_type = 'application/octet-stream'
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'

def get_plotly_download_link(fig, filename_base, text_html="Scarica HTML", text_png="Scarica PNG"):
    """Genera link download per grafico Plotly (HTML, PNG se kaleido è installato)."""
    href_html = ""; href_png = ""
    try:
        # Download HTML
        buf_html = io.StringIO(); fig.write_html(buf_html, include_plotlyjs='cdn')
        buf_html.seek(0); b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="{filename_base}.html">{text_html}</a>'
    except Exception as e_html: st.warning(f"Errore gen. HTML {filename_base}: {e_html}", icon="📄")

    try:
        # Download PNG (richiede kaleido: pip install kaleido)
        import importlib
        if importlib.util.find_spec("kaleido"):
            buf_png = io.BytesIO(); fig.write_image(buf_png, format="png", scale=2)
            buf_png.seek(0); b64_png = base64.b64encode(buf_png.getvalue()).decode()
            href_png = f'<a href="data:image/png;base64,{b64_png}" download="{filename_base}.png">{text_png}</a>'
        else:
             print("Nota: 'kaleido' non installato, download PNG grafico disabilitato.")
    except Exception as e_png: st.warning(f"Errore gen. PNG {filename_base}: {e_png}", icon="🖼️")

    return f"{href_html} {href_png}".strip()

def get_download_link_for_file(filepath, link_text=None):
    """Genera link download per un file su disco."""
    if not os.path.exists(filepath): return f"<i>File non trovato: {os.path.basename(filepath)}</i>"
    link_text = link_text or f"Scarica {os.path.basename(filepath)}"
    try:
        with open(filepath, "rb") as f: file_content = f.read()
        b64 = base64.b64encode(file_content).decode("utf-8")
        filename = os.path.basename(filepath)
        mime_type, _ = mimetypes.guess_type(filepath)
        if mime_type is None: mime_type = 'application/octet-stream'
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e: st.error(f"Errore link per {filename}: {e}"); return f"<i>Errore link</i>"

# --- Funzione Estrazione ID GSheet (Invariata) ---
def extract_sheet_id(url):
    """Estrae l'ID del foglio Google da un URL."""
    patterns = [r'/spreadsheets/d/([a-zA-Z0-9-_]+)', r'/d/([a-zA-Z0-9-_]+)/']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

# --- Funzione Estrazione Etichetta Stazione (Invariata) ---
def get_station_label(col_name, short=False):
    """Restituisce etichetta leggibile per colonna sensore, usando STATION_COORDS."""
    if col_name in STATION_COORDS:
        info = STATION_COORDS[col_name]
        loc_id = info.get('location_id')
        s_type = info.get('type', '')
        if loc_id:
            if short:
                # Conta quanti sensori dello stesso tipo ci sono nella stessa location_id
                sensors_at_loc = [sc['type'] for sc_name, sc in STATION_COORDS.items() if sc.get('location_id') == loc_id]
                if len(sensors_at_loc) > 1:
                    # Se più sensori, aggiungi tipo abbreviato
                    type_abbr = {'Pioggia': 'P', 'Livello': 'L', 'Umidità': 'U'}.get(s_type, '?')
                    label = f"{loc_id} ({type_abbr})"
                else:
                    # Altrimenti solo il nome della location
                    label = loc_id
                # Tronca se troppo lunga
                return label[:25] + ('...' if len(label) > 25 else '')
            else: # Non short, restituisci il nome completo della location
                return info.get('name', loc_id) # Usa 'name' se presente, altrimenti location_id
        # Fallback se location_id non c'è in STATION_COORDS[col_name]
        label = info.get('name', col_name) # Usa 'name' se presente, altrimenti col_name originale
        return label[:25] + ('...' if len(label) > 25 else '') if short else label

    # Fallback se col_name non è in STATION_COORDS
    parts = col_name.split(' - '); label = col_name.split(' (')[0].strip()
    if len(parts) > 1:
        location = parts[0].strip(); measurement = parts[1].split(' (')[0].strip();
        label = f"{location} ({measurement[0]})" if short else f"{location} - {measurement}"
    return label[:25] + ('...' if len(label) > 25 else '') if short else label

# --- Inizializzazione Session State (Invariata) ---
if 'active_model_name' not in st.session_state: st.session_state.active_model_name = None
if 'active_config' not in st.session_state: st.session_state.active_config = None
if 'active_model' not in st.session_state: st.session_state.active_model = None
if 'active_device' not in st.session_state: st.session_state.active_device = None
if 'active_scalers' not in st.session_state: st.session_state.active_scalers = None # Ora può essere dizionario o tupla
if 'df' not in st.session_state: st.session_state.df = None # Dati CSV storici
# Default feature columns (usate se non specificate nel config LSTM)
if 'feature_columns' not in st.session_state:
     st.session_state.feature_columns = [
         'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
         'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
         HUMIDITY_COL_NAME, # Aggiunta colonna umidità ai default
         'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
         'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
         'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
         'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
         'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
     ]
if 'date_col_name_csv' not in st.session_state: st.session_state.date_col_name_csv = 'Data e Ora' # Assicurati sia corretto per il tuo CSV
if 'dashboard_thresholds' not in st.session_state: st.session_state.dashboard_thresholds = DEFAULT_THRESHOLDS.copy()
# ... (altri stati sessione dashboard/alert invariati) ...
if 'last_dashboard_data' not in st.session_state: st.session_state.last_dashboard_data = None
if 'last_dashboard_error' not in st.session_state: st.session_state.last_dashboard_error = None
if 'last_dashboard_fetch_time' not in st.session_state: st.session_state.last_dashboard_fetch_time = None
if 'active_alerts' not in st.session_state: st.session_state.active_alerts = []
# Stato per simulazione Seq2Seq
if 'seq2seq_past_data_gsheet' not in st.session_state: st.session_state.seq2seq_past_data_gsheet = None
if 'seq2seq_last_ts_gsheet' not in st.session_state: st.session_state.seq2seq_last_ts_gsheet = None
# Stati per simulazione LSTM GSheet
if 'imported_sim_data_gs_df_lstm' not in st.session_state: st.session_state.imported_sim_data_gs_df_lstm = None
if 'imported_sim_start_time_gs_lstm' not in st.session_state: st.session_state.imported_sim_start_time_gs_lstm = None


# ==============================================================================
# --- LAYOUT STREAMLIT ---
# ==============================================================================
st.title('🌊 Dashboard e Modello Predittivo Idrologico (Seq2Seq Capable)')

# --- Sidebar ---
st.sidebar.header('Impostazioni')

# --- Caricamento Dati Storici CSV (Logica Invariata) ---
st.sidebar.subheader('Dati Storici (per Analisi/Training)')
uploaded_data_file = st.sidebar.file_uploader('Carica CSV Dati Storici (Opzionale)', type=['csv'], key="data_uploader")
# ... (Logica caricamento/pulizia CSV invariata) ...
# Assicurati che la pulizia gestisca bene la colonna HUMIDITY_COL_NAME
df = None; df_load_error = None; data_source_info = ""
data_path_to_load = None; is_uploaded = False
if uploaded_data_file is not None:
    data_path_to_load = uploaded_data_file; is_uploaded = True
    data_source_info = f"File caricato: **{uploaded_data_file.name}**"
elif os.path.exists(DEFAULT_DATA_PATH):
    # Forza ricarica da file solo se non è già in sessione O se è stato appena caricato un file
    if 'df' not in st.session_state or st.session_state.df is None or st.session_state.get('uploaded_file_processed') is None:
        data_path_to_load = DEFAULT_DATA_PATH; is_uploaded = False
        data_source_info = f"File default: **{DEFAULT_DATA_PATH}**"
        st.session_state['uploaded_file_processed'] = False # Flag per indicare che usiamo il default
    else:
        data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"
elif 'df' not in st.session_state or st.session_state.df is None:
     df_load_error = f"'{DEFAULT_DATA_PATH}' non trovato. Carica un CSV."

# Logica per decidere se processare il file
process_file = False
if is_uploaded and st.session_state.get('uploaded_file_processed') != uploaded_data_file.name:
    process_file = True # Processa il nuovo file caricato
elif data_path_to_load == DEFAULT_DATA_PATH and not st.session_state.get('uploaded_file_processed'):
    process_file = True # Processa il default se non è stato caricato un file o se il flag non è True
elif 'df' not in st.session_state or st.session_state.df is None:
    process_file = True # Processa se non c'è df in sessione

if process_file and data_path_to_load:
    try:
        # ... (read_csv con encoding detection) ...
        read_args = {'sep': ';', 'decimal': ',', 'low_memory': False}
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1']
        df_temp = None
        file_obj = data_path_to_load # Può essere path stringa o file object
        for enc in encodings_to_try:
             try:
                 if hasattr(file_obj, 'seek'): file_obj.seek(0) # Reset file pointer if it's a file object
                 df_temp = pd.read_csv(file_obj, encoding=enc, **read_args)
                 print(f"CSV letto con encoding {enc}")
                 break
             except UnicodeDecodeError: continue
             except Exception as read_e: raise read_e
        if df_temp is None: raise ValueError(f"Impossibile leggere CSV con encodings: {encodings_to_try}")

        date_col_csv = st.session_state.date_col_name_csv
        if date_col_csv not in df_temp.columns: raise ValueError(f"Colonna data CSV '{date_col_csv}' mancante.")

        # ... (Conversione data robusta) ...
        try:
            # Tenta il formato specifico prima
            df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
        except ValueError:
             try:
                 # Fallback a inferenza
                 df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], errors='coerce')
                 st.sidebar.warning(f"Formato data CSV non standard ('{date_col_csv}'). Tentativo di inferenza.")
             except Exception as e_date_csv_infer:
                 raise ValueError(f"Errore conversione data CSV '{date_col_csv}': {e_date_csv_infer}")

        df_temp = df_temp.dropna(subset=[date_col_csv])
        if df_temp.empty: raise ValueError("Nessuna riga valida post pulizia data CSV.")
        df_temp = df_temp.sort_values(by=date_col_csv).reset_index(drop=True)

        # ... (Pulizia numerica, usa st.session_state.feature_columns come riferimento) ...
        current_f_cols_state = st.session_state.feature_columns # Colonne attese
        features_to_clean = [col for col in df_temp.columns if col != date_col_csv] # Pulisci tutte le colonne numeriche presenti

        for col in features_to_clean:
             if pd.api.types.is_object_dtype(df_temp[col]) or pd.api.types.is_string_dtype(df_temp[col]):
                  col_str = df_temp[col].astype(str).str.strip()
                  # Gestisci separatori e N/A in modo più robusto
                  col_str = col_str.replace(['N/A', '', '-', 'None', 'null', 'NaN', 'nan'], np.nan, regex=False)
                  col_str = col_str.str.replace('.', '', regex=False) # Rimuovi separatore migliaia
                  col_str = col_str.str.replace(',', '.', regex=False) # Sostituisci virgola decimale
                  df_temp[col] = pd.to_numeric(col_str, errors='coerce')
             elif pd.api.types.is_numeric_dtype(df_temp[col]):
                 # Già numerico, non fare nulla
                 pass
             else:
                 # Tenta conversione forzata per altri tipi
                 df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')


        # ... (Fill NaN con info) ...
        numeric_cols = df_temp.select_dtypes(include=np.number).columns
        n_nan_before_fill = df_temp[numeric_cols].isnull().sum().sum()
        if n_nan_before_fill > 0:
             st.sidebar.caption(f"Trovati {n_nan_before_fill} NaN/non numerici nel CSV. Eseguito ffill/bfill.")
             df_temp[numeric_cols] = df_temp[numeric_cols].fillna(method='ffill').fillna(method='bfill')
             n_nan_after_fill = df_temp[numeric_cols].isnull().sum().sum()
             if n_nan_after_fill > 0:
                 st.sidebar.error(f"NaN residui ({n_nan_after_fill}) dopo fill CSV. Riempiti con 0.")
                 df_temp[numeric_cols] = df_temp[numeric_cols].fillna(0)

        st.session_state.df = df_temp
        if is_uploaded: st.session_state['uploaded_file_processed'] = uploaded_data_file.name
        else: st.session_state['uploaded_file_processed'] = True # Mark default as processed
        st.sidebar.success(f"Dati CSV caricati e processati ({len(st.session_state.df)} righe).")
        df = st.session_state.df

    except Exception as e:
        df = None; st.session_state.df = None
        st.session_state['uploaded_file_processed'] = None # Reset flag on error
        df_load_error = f'Errore caricamento/processamento CSV: {e}'
        st.sidebar.error(f"Errore CSV: {df_load_error}")
        st.error(traceback.format_exc()) # Mostra traceback per debug

# Recupera df da sessione se non caricato ora
if df is None and 'df' in st.session_state:
    df = st.session_state.df
    if data_source_info == "": # Update info if df came from session state
        if st.session_state.get('uploaded_file_processed') and isinstance(st.session_state.get('uploaded_file_processed'), str):
             data_source_info = f"File caricato: **{st.session_state['uploaded_file_processed']}** (da sessione)"
        else:
             data_source_info = f"File default: **{DEFAULT_DATA_PATH}** (da sessione)"

# Mostra info sorgente dati
if data_source_info: st.sidebar.caption(data_source_info)
if df_load_error and not data_source_info: st.sidebar.error(df_load_error)

data_ready_csv = df is not None


# --- Selezione Modello (Logica aggiornata per tipo modello) ---
st.sidebar.divider()
st.sidebar.subheader("Modello Predittivo (per Simulazione)")

# --- find_available_models è aggiornata per identificare il tipo ---
available_models_dict = find_available_models(MODELS_DIR)
model_display_names = sorted(list(available_models_dict.keys()))
MODEL_CHOICE_UPLOAD = "Carica File Manualmente..." # Solo LSTM per ora
MODEL_CHOICE_NONE = "-- Nessun Modello Selezionato --"
selection_options = [MODEL_CHOICE_NONE] + model_display_names # Rimosso upload da qui, gestito separatamente

# Gestione stato selezione
current_selection_name = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

# Gestione reset se modello attivo non più valido (es. file cancellato)
if current_selection_name not in selection_options and current_selection_name != MODEL_CHOICE_UPLOAD:
    st.session_state.active_model_name = MODEL_CHOICE_NONE
    current_selection_name = MODEL_CHOICE_NONE
    st.session_state.active_config = None; st.session_state.active_model = None
    st.session_state.active_device = None; st.session_state.active_scalers = None

# Determina indice per selectbox
try:
    current_index = selection_options.index(current_selection_name)
except ValueError:
    # Se era selezionato "Upload", mantieni la selezione visuale su "None"
    current_index = 0

selected_model_display_name = st.sidebar.selectbox(
    "Modello Pre-Addestrato:", selection_options, index=current_index,
    key="model_selector_predefined",
    help="Scegli un modello pre-addestrato (LSTM o Seq2Seq)."
)

# --- Upload Manuale (solo LSTM) come sezione separata ---
with st.sidebar.expander("Carica Modello LSTM Manualmente", expanded=(current_selection_name == MODEL_CHOICE_UPLOAD)):
     is_upload_active = False
     m_f = st.file_uploader('.pth', type=['pth'], key="up_pth")
     sf_f = st.file_uploader('.joblib (Features Scaler)', type=['joblib'], key="up_scf_lstm") # Key diversa
     st_f = st.file_uploader('.joblib (Target Scaler)', type=['joblib'], key="up_sct_lstm") # Key diversa

     if m_f and sf_f and st_f:
        st.caption("Configura parametri LSTM:")
        c1, c2 = st.columns(2)
        # Parametri specifici LSTM
        iw_up = c1.number_input("In Win Steps", 6*2, 168*2, 48, 2, key="up_in_steps_lstm") # Steps
        ow_up = c1.number_input("Out Win Steps", 1, 72*2, 24, 1, key="up_out_steps_lstm") # Steps
        hs_up = c2.number_input("Hidden", 16, 1024, 128, 16, key="up_hid_lstm")
        nl_up = c2.number_input("Layers", 1, 8, 2, 1, key="up_lay_lstm")
        dr_up = c2.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="up_drop_lstm")
        # Selezione feature/target basata sulle colonne globali (o del CSV se caricato)
        available_cols_for_upload = list(df.columns.drop(st.session_state.date_col_name_csv, errors='ignore')) if data_ready_csv else st.session_state.feature_columns
        default_targets_upload = [c for c in available_cols_for_upload if 'Livello' in c][:1]
        targets_up = st.multiselect("Target", available_cols_for_upload, default=default_targets_upload, key="up_targets_lstm")
        features_up = st.multiselect("Features (Input)", available_cols_for_upload, default=available_cols_for_upload, key="up_features_lstm")

        if targets_up and features_up:
            if st.button("Carica Modello Manuale", key="load_manual_lstm"):
                is_upload_active = True
                # Se l'utente carica manualmente, deseleziona il modello pre-addestrato
                selected_model_display_name = MODEL_CHOICE_UPLOAD
        else:
             st.caption("Scegli features e target.")
     else:
         st.caption("Carica tutti e 3 i file (.pth, scaler features, scaler target).")


# --- Logica Caricamento Modello (Aggiornata) ---
config_to_load = None; model_to_load = None; device_to_load = None
scalers_to_load = None; load_error_sidebar = False; model_type_loaded = None

# Priorità: Upload manuale ha precedenza se il bottone è stato premuto
if is_upload_active:
    st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
    temp_cfg_up = {"input_window": iw_up, "output_window": ow_up, "hidden_size": hs_up,
                 "num_layers": nl_up, "dropout": dr_up,
                 "feature_columns": features_up, # USA quelle selezionate qui
                 "target_columns": targets_up,
                 "name": "uploaded_lstm_model", "display_name": "Modello LSTM Caricato",
                 "model_type": "LSTM"} # Aggiunto tipo
    # Carica modello LSTM
    model_to_load, device_to_load = load_specific_model(m_f, temp_cfg_up)
    # Carica scaler LSTM (passa direttamente i percorsi/file)
    temp_model_info_up = {"scaler_features_path": sf_f, "scaler_targets_path": st_f} # Info fittizia per scaler
    scalers_tuple_up = load_specific_scalers(temp_cfg_up, temp_model_info_up)
    if model_to_load and scalers_tuple_up and isinstance(scalers_tuple_up, tuple) and len(scalers_tuple_up)==2:
        config_to_load = temp_cfg_up
        scalers_to_load = scalers_tuple_up # Salva la tupla (scaler_features, scaler_targets)
        model_type_loaded = "LSTM"
    else: load_error_sidebar = True; st.error("Errore caricamento modello/scaler LSTM manuale.")

# Altrimenti, carica il modello pre-addestrato selezionato
elif selected_model_display_name != MODEL_CHOICE_NONE:
    # Se la selezione è cambiata rispetto allo stato attivo, carica il nuovo modello
    if selected_model_display_name != st.session_state.get('active_model_name'):
        st.session_state.active_model_name = selected_model_display_name # Aggiorna subito lo stato
        if selected_model_display_name in available_models_dict:
            model_info = available_models_dict[selected_model_display_name]
            config_to_load = load_model_config(model_info["config_path"])

            if config_to_load:
                # Aggiungi info percorso e tipo al config in memoria
                config_to_load["pth_path"] = model_info["pth_path"]
                config_to_load["config_name"] = model_info["config_name"]
                config_to_load["display_name"] = selected_model_display_name
                model_type_loaded = model_info.get("model_type", "LSTM") # Tipo da find_available_models
                config_to_load["model_type"] = model_type_loaded # Assicura sia nel config

                # Carica modello specifico (LSTM o Seq2Seq)
                model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)

                # Carica scaler specifici (la funzione gestisce 1, 2 o 3 scaler)
                scalers_to_load = load_specific_scalers(config_to_load, model_info)

                # Verifica caricamento
                if not (model_to_load and scalers_to_load):
                    load_error_sidebar = True; config_to_load = None # Reset se fallisce
            else:
                load_error_sidebar = True # Errore caricamento config
        else:
             st.sidebar.error(f"Modello '{selected_model_display_name}' non trovato (inconsistenza?).")
             load_error_sidebar = True
    # Se la selezione NON è cambiata, usa i dati già in sessione (se presenti)
    elif st.session_state.get('active_model') and st.session_state.get('active_config'):
         config_to_load = st.session_state.active_config
         model_to_load = st.session_state.active_model
         device_to_load = st.session_state.active_device
         scalers_to_load = st.session_state.active_scalers
         model_type_loaded = config_to_load.get("model_type", "LSTM")

# Se la selezione è "None", resetta lo stato attivo
elif selected_model_display_name == MODEL_CHOICE_NONE and st.session_state.get('active_model_name') != MODEL_CHOICE_NONE:
     st.session_state.active_model_name = MODEL_CHOICE_NONE
     st.session_state.active_config = None; st.session_state.active_model = None
     st.session_state.active_device = None; st.session_state.active_scalers = None


# Salva nello stato sessione SOLO se un modello è stato caricato con successo ORA
if config_to_load and model_to_load and device_to_load and scalers_to_load:
    # Verifica che scalers_to_load non sia None (o None, None per LSTM fallito)
    scalers_valid = False
    if isinstance(scalers_to_load, dict): # Seq2Seq
        scalers_valid = all(scalers_to_load.values())
    elif isinstance(scalers_to_load, tuple) and len(scalers_to_load) == 2: # LSTM
        scalers_valid = all(scalers_to_load)

    if scalers_valid:
        st.session_state.active_config = config_to_load
        st.session_state.active_model = model_to_load
        st.session_state.active_device = device_to_load
        st.session_state.active_scalers = scalers_to_load # Salva tupla o dizionario
        # active_model_name è già stato aggiornato all'inizio del blocco di caricamento


# Mostra feedback basato sullo stato sessione ATTUALE
active_config_sess = st.session_state.get('active_config')
active_model_sess = st.session_state.get('active_model')
active_model_name_sess = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)

if active_model_sess and active_config_sess:
    cfg = active_config_sess
    model_type_sess = cfg.get('model_type', 'LSTM')
    display_feedback_name = cfg.get("display_name", active_model_name_sess)
    # Mostra info diverse per tipo
    st.sidebar.success(f"Modello ATTIVO: '{display_feedback_name}' ({model_type_sess})")
    if model_type_sess == "Seq2Seq":
         st.sidebar.caption(f"In={cfg['input_window_steps']}s, Fore={cfg['forecast_window_steps']}s, Out={cfg['output_window_steps']}s")
    else: # LSTM
         st.sidebar.caption(f"In={cfg['input_window']}s, Out={cfg['output_window']}s") # Steps
elif load_error_sidebar and active_model_name_sess not in [MODEL_CHOICE_NONE]:
     st.sidebar.error(f"Caricamento modello '{active_model_name_sess}' fallito.")
elif active_model_name_sess == MODEL_CHOICE_UPLOAD and not active_model_sess:
     st.sidebar.info("Completa caricamento manuale modello LSTM.")
elif active_model_name_sess == MODEL_CHOICE_NONE:
     st.sidebar.info("Nessun modello selezionato per la simulazione.")


# --- Configurazione Soglie Dashboard (Invariata) ---
st.sidebar.divider()
st.sidebar.subheader("Configurazione Soglie Dashboard")
# ... (Codice expander soglie invariato) ...
with st.sidebar.expander("Modifica Soglie di Allerta", expanded=False):
    temp_thresholds = st.session_state.dashboard_thresholds.copy()
    monitorable_cols = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
    cols_thresh = st.columns(2)
    col_idx_thresh = 0
    for col in monitorable_cols:
        with cols_thresh[col_idx_thresh % 2]:
            label_short = get_station_label(col, short=True)
            is_level = 'Livello' in col or '(m)' in col or '(mt)' in col
            is_humidity = 'Umidit' in col # Check Umidità
            # BUG FIX: Originale aveva logica step/format invertita
            # Correggo: step 1.0 per umidità e pioggia, 0.1 per livelli
            step = 0.1 if is_level else 1.0
            # Correggo: format %.1f per tutti tranne livelli (che vogliono %.2f o simile?)
            # Usiamo %.1f per Pioggia/Umidità, %.2f per Livello
            fmt = "%.2f" if is_level else "%.1f"
            min_v = 0.0
            current_threshold = st.session_state.dashboard_thresholds.get(col, DEFAULT_THRESHOLDS.get(col, 0.0))
            new_threshold = st.number_input(
                label=f"{label_short}", value=float(current_threshold), min_value=min_v, step=step, format=fmt,
                key=f"thresh_{col}", help=f"Soglia per: {col}"
            )
            if new_threshold != current_threshold: temp_thresholds[col] = new_threshold
        col_idx_thresh += 1
    if st.button("Salva Soglie", key="save_thresholds", type="primary"):
        st.session_state.dashboard_thresholds = temp_thresholds.copy()
        st.success("Soglie aggiornate!"); time.sleep(0.5); st.rerun()


# --- Menu Navigazione (Invariato) ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
model_ready = active_model_sess is not None and active_config_sess is not None
# data_ready_csv definito prima

radio_options = ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello']
# ... (Logica captions, disabilitazione, selezione pagina invariata) ...
requires_model = ['Simulazione']
requires_csv = ['Analisi Dati Storici', 'Allenamento Modello']
radio_captions = []
disabled_options = []
default_page_idx = 0
for i, opt in enumerate(radio_options):
    caption = ""; disabled = False
    if opt == 'Dashboard': caption = "Monitoraggio GSheet"
    elif opt == 'Simulazione':
        if not model_ready: caption = "Richiede Modello attivo"; disabled = True
        else: caption = f"Esegui previsioni ({active_config_sess.get('model_type', 'LSTM')})"
    elif opt == 'Analisi Dati Storici':
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Esplora dati CSV"
    elif opt == 'Allenamento Modello':
        if not data_ready_csv: caption = "Richiede Dati CSV"; disabled = True
        else: caption = "Allena un nuovo modello"
    radio_captions.append(caption); disabled_options.append(disabled)
    if opt == 'Dashboard': default_page_idx = i

current_page_key = 'current_page'
selected_page = st.session_state.get(current_page_key, radio_options[default_page_idx])
try:
    current_page_index_in_options = radio_options.index(selected_page)
    if disabled_options[current_page_index_in_options]:
        st.sidebar.warning(f"Pagina '{selected_page}' non disponibile. Reindirizzato a Dashboard.")
        selected_page = radio_options[default_page_idx]
        st.session_state[current_page_key] = selected_page
except ValueError:
    selected_page = radio_options[default_page_idx]
    st.session_state[current_page_key] = selected_page
current_idx_display = radio_options.index(selected_page)

chosen_page = st.sidebar.radio(
    'Scegli una funzionalità:', options=radio_options, captions=radio_captions,
    index=current_idx_display, key='page_selector_radio', disabled=False, # Abilita sempre il radio
    # Gestiamo il blocco della pagina scelta nella logica principale
)

# Se l'utente sceglie una pagina disabilitata, riportalo alla dashboard
chosen_page_index = radio_options.index(chosen_page)
if disabled_options[chosen_page_index]:
    st.sidebar.error(f"La pagina '{chosen_page}' non è disponibile. {radio_captions[chosen_page_index]}.")
    page = radio_options[default_page_idx] # Imposta la pagina sulla dashboard
    st.session_state[current_page_key] = page # Aggiorna lo stato della sessione
    # Potrebbe essere utile un rerun qui, ma aspettiamo la fine della sidebar
    # st.rerun()
else:
    page = chosen_page # Usa la pagina scelta
    if chosen_page != selected_page:
        st.session_state[current_page_key] = chosen_page
        st.rerun() # Rerun se la pagina valida è cambiata


# ==============================================================================
# --- Logica Pagine Principali ---
# ==============================================================================

# Recupera variabili attive da session state
active_config = st.session_state.get('active_config')
active_model = st.session_state.get('active_model')
active_device = st.session_state.get('active_device')
active_scalers = st.session_state.get('active_scalers') # Tupla o Dizionario
active_model_type = active_config.get("model_type", "LSTM") if active_config else "LSTM"

df_current_csv = st.session_state.get('df', None) # Dati CSV
date_col_name_csv = st.session_state.date_col_name_csv


# --- PAGINA DASHBOARD ---
if page == 'Dashboard':
    st.header(f'📊 Dashboard Monitoraggio Idrologico')
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        st.error("🚨 Errore Configurazione: Credenziali Google mancanti nei secrets.")
        st.stop()

    now_ts = time.time()
    cache_time_key = int(now_ts // DASHBOARD_REFRESH_INTERVAL_SECONDS)
    df_dashboard, error_msg, actual_fetch_time = fetch_gsheet_dashboard_data(
        cache_time_key, GSHEET_ID, GSHEET_RELEVANT_COLS,
        GSHEET_DATE_COL, GSHEET_DATE_FORMAT, DASHBOARD_HISTORY_ROWS
    )
    st.session_state.last_dashboard_data = df_dashboard
    st.session_state.last_dashboard_error = error_msg
    if df_dashboard is not None or error_msg is None: # Salva tempo solo se fetch OK o nessun errore
        st.session_state.last_dashboard_fetch_time = actual_fetch_time

    # --- Visualizzazione (Status, Refresh Button, Errori) ---
    col_status, col_refresh_btn = st.columns([3, 1])
    with col_status:
        last_fetch_dt_sess = st.session_state.get('last_dashboard_fetch_time')
        if last_fetch_dt_sess:
             fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess
             fetch_secs_ago = int(fetch_time_ago.total_seconds())
             status_text = f"Dati GSheet ({DASHBOARD_HISTORY_ROWS} righe) alle: {last_fetch_dt_sess.strftime('%d/%m/%Y %H:%M:%S')} ({fetch_secs_ago}s fa). Auto-refresh: {DASHBOARD_REFRESH_INTERVAL_SECONDS}s."
             st.caption(status_text)
        else: st.caption("In attesa recupero dati GSheet...")
    with col_refresh_btn:
        if st.button("🔄 Forza Aggiorna", key="dash_refresh_button"):
            # Clear cache specific functions
            fetch_gsheet_dashboard_data.clear()
            fetch_sim_gsheet_data.clear()
            st.success("Cache GSheet pulita. Ricaricamento..."); time.sleep(0.5); st.rerun()

    if st.session_state.last_dashboard_error:
        error_msg_display = st.session_state.last_dashboard_error
        if any(x in error_msg_display for x in ["API", "Foglio", "Credenziali"]): st.error(f"🚨 Errore Grave GSheet: {error_msg_display}")
        else: st.warning(f"⚠️ Attenzione Dati GSheet: {error_msg_display}")

    # --- Visualizzazione Dati e Grafici (usa df_dashboard dalla sessione) ---
    df_dashboard_sess = st.session_state.get('last_dashboard_data')
    if df_dashboard_sess is not None and not df_dashboard_sess.empty:
        # --- Stato ultimo rilevamento ---
        latest_row_data = df_dashboard_sess.iloc[-1]
        last_update_time = latest_row_data.get(GSHEET_DATE_COL)
        if pd.notna(last_update_time) and isinstance(last_update_time, pd.Timestamp):
             # ... (logica calcolo tempo trascorso come prima) ...
             time_now_italy = datetime.now(italy_tz)
             # Assicurati che last_update_time sia timezone-aware
             if last_update_time.tzinfo is None:
                 last_update_time = italy_tz.localize(last_update_time)
             else:
                 last_update_time = last_update_time.tz_convert(italy_tz)

             time_delta = time_now_italy - last_update_time; minutes_ago = int(time_delta.total_seconds() // 60)
             time_str = last_update_time.strftime('%d/%m/%Y %H:%M:%S %Z')
             if minutes_ago < 2: time_ago_str = "pochi istanti fa"
             elif minutes_ago < 60: time_ago_str = f"{minutes_ago} min fa"
             else: time_ago_str = f"circa {minutes_ago // 60} ore fa"
             st.success(f"**Ultimo rilevamento:** {time_str} ({time_ago_str})")
             if minutes_ago > 90: # Aumentato a 1.5 ore per warning
                 st.warning(f"⚠️ Attenzione: Ultimo dato ricevuto oltre {minutes_ago} minuti fa.")
        else: st.warning("⚠️ Timestamp ultimo rilevamento non valido o mancante.")

        st.divider()
        # --- Tabella Valori Attuali e Soglie ---
        st.subheader("Tabella Valori Attuali")
        cols_to_monitor = [col for col in GSHEET_RELEVANT_COLS if col != GSHEET_DATE_COL]
        table_rows = []; current_alerts = []
        for col_name in cols_to_monitor:
            current_value = latest_row_data.get(col_name)
            threshold = st.session_state.dashboard_thresholds.get(col_name)
            alert_active = False; value_numeric = np.nan; value_display = "N/D"; unit = ""
            if 'Pioggia' in col_name: unit = '(mm)'
            elif 'Livello' in col_name: unit = '(m)'
            elif 'Umidit' in col_name: unit = '(%)' # Aggiunto unità umidità

            if pd.notna(current_value) and isinstance(current_value, (int, float, np.number)):
                 value_numeric = float(current_value)
                 # --- CORREZIONE APPLICATA QUI ---
                 fmt_spec = ".1f" if unit in ['(mm)', '(%)'] else ".2f"  # Usa .1f per mm/%, .2f per livelli (o altro)
                 try:
                     # Usa la specifica di formato corretta nella f-string
                     value_display = f"{value_numeric:{fmt_spec}} {unit}".strip()
                 except ValueError as e:
                     st.error(f"Errore formattazione valore {value_numeric} con spec '{fmt_spec}' per colonna '{col_name}': {e}", icon="⚠️")
                     value_display = f"{value_numeric} {unit}".strip() # Fallback
                 # --- FINE CORREZIONE ---
                 if threshold is not None and isinstance(threshold, (int, float, np.number)) and value_numeric >= float(threshold):
                      alert_active = True; current_alerts.append((col_name, value_numeric, threshold))
            elif pd.notna(current_value): value_display = f"{current_value} (?)" # Valore non numerico

            status = "🔴 ALLERTA" if alert_active else ("✅ OK" if pd.notna(value_numeric) else "⚪ N/D")
            threshold_display = f"{float(threshold):.1f}" if threshold is not None else "-"
            table_rows.append({"Sensore": get_station_label(col_name, short=True), "Nome Completo": col_name, "Valore Numerico": value_numeric,
                               "Valore Attuale": value_display, "Soglia": threshold_display, "Soglia Numerica": float(threshold) if threshold else None, "Stato": status})
        df_display = pd.DataFrame(table_rows)
        def highlight_threshold(row): # ... (funzione stile invariata) ...
            style = [''] * len(row); threshold_val = row['Soglia Numerica']; current_val = row['Valore Numerico']
            if pd.notna(threshold_val) and pd.notna(current_val) and current_val >= threshold_val: style = ['background-color: rgba(255, 0, 0, 0.15); color: black; font-weight: bold;'] * len(row)
            return style
        st.dataframe(df_display.style.apply(highlight_threshold, axis=1), column_order=["Sensore", "Valore Attuale", "Soglia", "Stato"], hide_index=True, use_container_width=True)
        st.session_state.active_alerts = current_alerts

        st.divider()
        # --- Grafico Comparativo Configurabile ---
        st.subheader("Grafico Comparativo Storico")
        # ... (Logica multiselect e plot comparativo invariata) ...
        sensor_options_compare = {get_station_label(col, short=True): col for col in cols_to_monitor}
        default_selection_labels = [label for label, col in sensor_options_compare.items() if 'Livello' in col][:3] or list(sensor_options_compare.keys())[:2]
        selected_labels_compare = st.multiselect("Seleziona sensori da confrontare:", options=list(sensor_options_compare.keys()), default=default_selection_labels, key="compare_select_multi")
        selected_cols_compare = [sensor_options_compare[label] for label in selected_labels_compare]
        if selected_cols_compare:
            fig_compare = go.Figure()
            x_axis_data = df_dashboard_sess[GSHEET_DATE_COL]
            for col in selected_cols_compare: fig_compare.add_trace(go.Scatter(x=x_axis_data, y=df_dashboard_sess[col], mode='lines', name=get_station_label(col, short=True)))
            fig_compare.update_layout(title=f"Andamento Storico Comparato (ultime {DASHBOARD_HISTORY_ROWS//2} ore)", xaxis_title='Data e Ora', yaxis_title='Valore', height=500, hovermode="x unified")
            st.plotly_chart(fig_compare, use_container_width=True)
            # ... (Link download) ...
            compare_filename_base = f"compare_{'_'.join(sl.replace(' ','_') for sl in selected_labels_compare)}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.markdown(get_plotly_download_link(fig_compare, compare_filename_base), unsafe_allow_html=True)

        st.divider()
        # --- Grafici Individuali ---
        st.subheader("Grafici Individuali Storici")
        # ... (Logica plot grafici individuali invariata) ...
        num_cols_individual = 3; graph_cols = st.columns(num_cols_individual); col_idx_graph = 0
        x_axis_data_indiv = df_dashboard_sess[GSHEET_DATE_COL]
        for col_name in cols_to_monitor:
            with graph_cols[col_idx_graph % num_cols_individual]:
                threshold_individual = st.session_state.dashboard_thresholds.get(col_name)
                label_individual = get_station_label(col_name, short=True)
                unit = '(mm)' if 'Pioggia' in col_name else ('(m)' if 'Livello' in col_name else ('(%)' if 'Umidit' in col_name else ''))
                yaxis_title_individual = f"Valore {unit}".strip()
                fig_individual = go.Figure(); fig_individual.add_trace(go.Scatter(x=x_axis_data_indiv, y=df_dashboard_sess[col_name], mode='lines', name=label_individual))
                if threshold_individual is not None: fig_individual.add_hline(y=threshold_individual, line_dash="dash", line_color="red", annotation_text=f"Soglia ({float(threshold_individual):.1f})")
                fig_individual.update_layout(title=f"{label_individual}", xaxis_title=None, yaxis_title=yaxis_title_individual, height=300, hovermode="x unified", showlegend=False, margin=dict(t=40, b=30, l=50, r=10))
                fig_individual.update_yaxes(rangemode='tozero')
                st.plotly_chart(fig_individual, use_container_width=True)
                # ... (Link download) ...
                ind_filename_base = f"sensor_{label_individual.replace(' ','_').replace('(','').replace(')','')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                st.markdown(get_plotly_download_link(fig_individual, ind_filename_base, text_html="HTML", text_png="PNG"), unsafe_allow_html=True)
            col_idx_graph += 1

        st.divider()
        # --- Riepilogo Alert ---
        active_alerts_sess = st.session_state.get('active_alerts', [])
        if active_alerts_sess:
             st.warning("**🚨 ALLERTE ATTIVE (Valori >= Soglia) 🚨**")
             # ... (Logica markdown alert invariata) ...
             alert_md = ""
             sorted_alerts = sorted(active_alerts_sess, key=lambda x: get_station_label(x[0], short=False))
             for col, val, thr in sorted_alerts:
                 label_alert = get_station_label(col, short=False); sensor_type_alert = STATION_COORDS.get(col, {}).get('type', '')
                 type_str = f" ({sensor_type_alert})" if sensor_type_alert else ""
                 unit = '(mm)' if 'Pioggia' in col else ('(m)' if 'Livello' in col else ('(%)' if 'Umidit' in col else ''))
                 # Usa già la sintassi corretta f-string qui
                 val_fmt = f"{val:.1f}" if unit in ['(mm)', '(%)'] else f"{val:.2f}"
                 thr_fmt = f"{float(thr):.1f}" if isinstance(thr, (int, float, np.number)) else str(thr)
                 alert_md += f"- **{label_alert}{type_str}**: Valore **{val_fmt}{unit}** >= Soglia **{thr_fmt}{unit}**\n"
             st.markdown(alert_md)
        else: st.success("✅ Nessuna soglia superata nell'ultimo rilevamento.")

    elif df_dashboard_sess is not None and df_dashboard_sess.empty:
        st.warning("Recupero dati GSheet OK, ma nessun dato trovato.")
    elif not st.session_state.last_dashboard_error: # Se non ci sono errori e il df è None, fetch non è ancora avvenuto
        st.info("Recupero dati dashboard in corso...")

    # --- Auto Refresh JS (Invariato) ---
    try:
        component_key = f"dashboard_auto_refresh_{DASHBOARD_REFRESH_INTERVAL_SECONDS}"
        js_code = f"""
        (function() {{
            const intervalIdKey = 'streamlit_auto_refresh_interval_id_{component_key}';
            // Clear previous interval if it exists
            if (window[intervalIdKey]) {{
                clearInterval(window[intervalIdKey]);
            }}
            // Set new interval
            window[intervalIdKey] = setInterval(function() {{
                // Check if Streamlit hook is available and trigger rerun
                if (window.streamlitHook && typeof window.streamlitHook.rerunScript === 'function') {{
                    console.log('Triggering Streamlit rerun via JS timer ({component_key})');
                    window.streamlitHook.rerunScript(null);
                }} else {{
                    console.warn('Streamlit hook not available for JS auto-refresh.');
                    // Optionally clear interval if hook is consistently unavailable
                    // clearInterval(window[intervalIdKey]);
                }}
            }}, {DASHBOARD_REFRESH_INTERVAL_SECONDS * 1000});
            // Return the interval ID (optional)
            // return window[intervalIdKey];
        }})();
        """
        streamlit_js_eval(js_expressions=js_code, key=component_key, want_output=False)
    except Exception as e_js: st.warning(f"Impossibile impostare auto-refresh: {e_js}")

# --- PAGINA SIMULAZIONE (Logica Biforcata LSTM/Seq2Seq) ---
elif page == 'Simulazione':
    st.header(f'🧪 Simulazione Idrologica ({active_model_type})') # Indica tipo modello
    if not model_ready:
        st.warning("⚠️ Seleziona un Modello attivo (dalla sidebar) per usare la Simulazione.")
        st.stop() # Blocca esecuzione pagina se modello non pronto

    # Recupera config e parametri specifici del modello attivo
    if active_model_type == "Seq2Seq":
        input_steps = active_config["input_window_steps"]
        forecast_steps = active_config["forecast_window_steps"]
        output_steps = active_config["output_window_steps"]
        past_feature_cols = active_config["all_past_feature_columns"]
        forecast_input_cols = active_config["forecast_input_columns"]
        target_columns_model = active_config["target_columns"]
        st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}** (Seq2Seq)")
        st.caption(f"Input Storico: {input_steps} steps ({input_steps*0.5:.1f}h) | Input Forecast: {forecast_steps} steps ({forecast_steps*0.5:.1f}h) | Output: {output_steps} steps ({output_steps*0.5:.1f}h)")
        with st.expander("Dettagli Colonne Seq2Seq"):
             st.write("**Feature Storiche (Encoder Input):**")
             st.caption(", ".join(past_feature_cols))
             st.write("**Feature Forecast (Decoder Input):**")
             st.caption(", ".join(forecast_input_cols))
             st.write("**Target (Output):**")
             st.caption(", ".join(target_columns_model))
    else: # LSTM Standard
        input_steps = active_config["input_window"] # Steps
        output_steps = active_config["output_window"] # Steps
        feature_columns_current_model = active_config.get("feature_columns", st.session_state.feature_columns) # Usa globali se mancano
        target_columns_model = active_config["target_columns"]
        st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}** (LSTM)")
        st.caption(f"Input: {input_steps} steps ({input_steps*0.5:.1f}h) | Output: {output_steps} steps ({output_steps*0.5:.1f}h)")
        with st.expander("Dettagli Colonne LSTM"):
             st.write("**Feature Input:**")
             st.caption(", ".join(feature_columns_current_model))
             st.write("**Target (Output):**")
             st.caption(", ".join(target_columns_model))


    # --- Selezione Metodo Input Simulazione ---
    if active_model_type == "Seq2Seq":
         sim_method = 'Simula con Previsioni Meteo (Seq2Seq)'
         st.subheader(f"Preparazione Dati per Simulazione Seq2Seq")
    else: # LSTM
         sim_method_options = ['Manuale (Valori Costanti)', 'Importa da Google Sheet (Ultime Ore)', 'Orario Dettagliato (Tabella)']
         if data_ready_csv: sim_method_options.append('Usa Ultime Ore da CSV Caricato')
         sim_method = st.radio("Metodo preparazione dati input simulazione LSTM:", sim_method_options, key="sim_method_radio_select_lstm", horizontal=False)


    # --- Logica per Metodo Seq2Seq ---
    if sim_method == 'Simula con Previsioni Meteo (Seq2Seq)':
         # --- 1. Recupero dati storici passati da GSheet ---
         st.markdown("**Passo 1: Recupero Dati Storici (Input Encoder)**")
         sheet_url_sim_s2s = st.text_input("URL Foglio Google storico", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", key="sim_gsheet_url_s2s")
         sheet_id_sim_s2s = extract_sheet_id(sheet_url_sim_s2s)

         # Mappatura colonne (solo per recuperare i dati storici)
         # Qui potremmo voler mappare TUTTE le colonne GSheet rilevanti alle feature passate richieste
         column_mapping_gsheet_to_past_s2s = {
            # NOME_COLONNA_GSHEET : NOME_FEATURE_MODELLO_PASSATO
             'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
             'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
             'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
             'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
             'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME, # Confermata
             'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
             'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
             'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
             'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
             'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
             # Mappa anche la colonna data di GSheet se è diversa da quella richiesta dal modello
             GSHEET_DATE_COL: st.session_state.date_col_name_csv # Assumiamo che il modello usi la stessa colonna data del CSV
         }
         # Filtra mappatura solo per le colonne effettivamente richieste da `past_feature_cols` + data
         relevant_model_cols_for_mapping = past_feature_cols + [st.session_state.date_col_name_csv]
         column_mapping_gsheet_to_past_s2s_filtered = {
             gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_past_s2s.items()
             if model_col in relevant_model_cols_for_mapping
         }

         # Gestione imputazione per feature passate non trovate in GSheet
         past_features_set = set(past_feature_cols)
         mapped_past_features_target = set(column_mapping_gsheet_to_past_s2s_filtered.values()) - {st.session_state.date_col_name_csv} # Escludi data
         missing_past_features_in_map = list(past_features_set - mapped_past_features_target)
         imputed_values_sim_past_s2s = {}
         if missing_past_features_in_map:
             st.warning(f"Feature storiche non mappate da GSheet: {', '.join(missing_past_features_in_map)}. Saranno imputate.")
             for missing_f in missing_past_features_in_map:
                 default_val = 0.0
                 if data_ready_csv and missing_f in df_current_csv.columns and pd.notna(df_current_csv[missing_f].median()):
                     default_val = df_current_csv[missing_f].median() # Usa mediana da CSV se disponibile
                 imputed_values_sim_past_s2s[missing_f] = default_val

         # Bottone e logica fetch
         fetch_error_gsheet_s2s = None
         if st.button("Carica/Aggiorna Storico da GSheet", key="fetch_gsh_s2s_base"):
             if not sheet_id_sim_s2s: st.error("ID GSheet mancante."); fetch_error_gsheet_s2s = "ID GSheet mancante."
             else:
                  fetch_sim_gsheet_data.clear() # Pulisce cache prima di chiamare
                  with st.spinner("Recupero dati storici (passato)..."):
                      # Richiedi TUTTE le feature passate + data per poter selezionare dopo
                      required_cols_fetch = past_feature_cols + [st.session_state.date_col_name_csv]
                      imported_df_past, import_err_past, last_ts_past = fetch_sim_gsheet_data(
                           sheet_id_sim_s2s, input_steps, GSHEET_DATE_COL, GSHEET_DATE_FORMAT,
                           column_mapping_gsheet_to_past_s2s_filtered,
                           required_cols_fetch, # Richiedi tutte le feature passate + data
                           imputed_values_sim_past_s2s
                      )
                  if import_err_past:
                      st.error(f"Recupero storico Seq2Seq fallito: {import_err_past}")
                      st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = import_err_past
                  elif imported_df_past is not None:
                      # Seleziona e riordina SOLO le colonne necessarie per l'encoder
                      try:
                          final_past_df = imported_df_past[past_feature_cols]
                          st.success(f"Recuperate e processate {len(final_past_df)} righe storiche.")
                          st.session_state.seq2seq_past_data_gsheet = final_past_df # Salva solo le colonne giuste
                          st.session_state.seq2seq_last_ts_gsheet = last_ts_past if last_ts_past else datetime.now(italy_tz)
                          fetch_error_gsheet_s2s = None
                      except KeyError as e_cols:
                          st.error(f"Errore selezione colonne storiche dopo fetch: {e_cols}")
                          st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = f"Errore colonne: {e_cols}"
                  else: st.error("Recupero storico Seq2Seq non riuscito (risultato vuoto)."); fetch_error_gsheet_s2s = "Errore sconosciuto recupero dati."

         # Mostra stato caricamento dati base
         past_data_loaded_s2s = st.session_state.seq2seq_past_data_gsheet is not None
         if past_data_loaded_s2s:
             st.caption("Dati storici base (input encoder) caricati.")
             with st.expander("Mostra dati storici base (Input Encoder)"):
                 st.dataframe(st.session_state.seq2seq_past_data_gsheet.round(3))
         elif fetch_error_gsheet_s2s: st.warning(f"Impossibile procedere ({fetch_error_gsheet_s2s}).")
         else: st.info("Clicca il bottone sopra per caricare i dati storici base da Google Sheet.")

         # --- 2. Input Previsioni Utente (Input Decoder) ---
         if past_data_loaded_s2s:
             st.markdown(f"**Passo 2: Inserisci Previsioni Future ({forecast_steps} steps = {forecast_steps*0.5:.1f}h)**")
             st.caption(f"Inserisci i valori previsti per le feature: **{', '.join(forecast_input_cols)}**")

             # Data Editor per le previsioni future
             # Inizializza con valori sensati (es. 0 per pioggia, ultima umidità nota dal df storico caricato)
             forecast_df_initial = pd.DataFrame(index=range(forecast_steps), columns=forecast_input_cols)
             last_known_past_data = st.session_state.seq2seq_past_data_gsheet.iloc[-1]

             for col in forecast_input_cols:
                  if 'pioggia' in col.lower() or 'cumulata' in col.lower():
                      forecast_df_initial[col] = 0.0 # Default 0 pioggia
                  elif col in last_known_past_data.index:
                      # Usa l'ultimo valore noto dalla serie storica come default
                      last_val = last_known_past_data.get(col)
                      forecast_df_initial[col] = last_val if pd.notna(last_val) else 0.0
                  else: # Se la colonna forecast non era nel passato, metti 0
                      forecast_df_initial[col] = 0.0

             forecast_editor_key = "seq2seq_forecast_editor"
             edited_forecast_df = st.data_editor(
                 forecast_df_initial.round(2),
                 key=forecast_editor_key,
                 num_rows="fixed",
                 use_container_width=True,
                 column_config={ # Configurazione dinamica colonne forecast
                     col: st.column_config.NumberColumn(
                         label=get_station_label(col, short=True), # Usa etichetta breve
                         help=f"Valore previsto per {col}",
                         min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else None,
                         max_value=100.0 if 'umidit' in col.lower() else None,
                         format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f",
                         step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1),
                         required=True # Rendi tutte le colonne richieste
                     ) for col in forecast_input_cols
                 }
             )

             # Validazione editor forecast
             forecast_data_valid_s2s = False
             if edited_forecast_df is not None and not edited_forecast_df.isnull().any().any() and edited_forecast_df.shape == (forecast_steps, len(forecast_input_cols)):
                  forecast_data_valid_s2s = True
                  st.caption("Previsioni future inserite correttamente.")
             else:
                  st.warning("Completa o correggi la tabella delle previsioni future. Assicurati che tutti i valori siano numerici.")

             # --- 3. Esecuzione Simulazione Seq2Seq ---
             st.divider()
             st.markdown("**Passo 3: Esegui Simulazione Seq2Seq**")
             can_run_s2s_sim = past_data_loaded_s2s and forecast_data_valid_s2s

             if st.button("🚀 Esegui Simulazione Seq2Seq", disabled=not can_run_s2s_sim, type="primary", key="run_s2s_sim_button"):
                  if not can_run_s2s_sim: st.error("Mancano dati storici o previsioni valide.")
                  else:
                       with st.spinner("Simulazione Seq2Seq in corso..."):
                           # Prendi dati pronti
                           past_data_np = st.session_state.seq2seq_past_data_gsheet[past_feature_cols].astype(float).values
                           future_forecast_np = edited_forecast_df[forecast_input_cols].astype(float).values

                           # Chiama predict_seq2seq
                           predictions_s2s = predict_seq2seq(
                               active_model, past_data_np, future_forecast_np,
                               active_scalers, # Dizionario {"past":..., "forecast":..., "targets":...}
                               active_config, active_device
                           )

                           # Mostra risultati
                           if predictions_s2s is not None:
                                output_steps_actual = predictions_s2s.shape[0] # Usa shape reale output
                                total_hours_output_actual = output_steps_actual * 0.5
                                st.subheader(f'📊 Risultato Simulazione Seq2Seq: Prossime {total_hours_output_actual:.1f} ore ({output_steps_actual} steps)')
                                start_pred_time_s2s = st.session_state.get('seq2seq_last_ts_gsheet', datetime.now(italy_tz))
                                st.caption(f"Previsione calcolata a partire da: {start_pred_time_s2s.strftime('%d/%m/%Y %H:%M:%S %Z')}")

                                # Visualizza tabella e grafici (usa plot_predictions)
                                pred_times_s2s = [start_pred_time_s2s + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]
                                results_df_s2s = pd.DataFrame(predictions_s2s, columns=target_columns_model)
                                results_df_s2s.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_s2s])

                                # Rinomina colonne per leggibilità tabella
                                rename_dict = {'Ora Prevista': 'Ora Prevista'}
                                original_to_renamed_map = {}
                                for col in target_columns_model:
                                   unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', col) # Cerca unità in [] o ()
                                   unit = f"({unit_match.group(1) or unit_match.group(2)})" if unit_match else ""
                                   new_name = f"{get_station_label(col, short=True)} {unit}".strip()
                                   count = 1; final_name = new_name
                                   while final_name in rename_dict.values(): count += 1; final_name = f"{new_name}_{count}" # Evita duplicati
                                   rename_dict[col] = final_name; original_to_renamed_map[col] = final_name

                                # Crea copia prima di rinominare per download
                                results_df_s2s_display = results_df_s2s.copy()
                                results_df_s2s_display.rename(columns=rename_dict, inplace=True)
                                numeric_cols_renamed = list(original_to_renamed_map.values())

                                try:
                                    cols_in_df_to_round = [col for col in numeric_cols_renamed if col in results_df_s2s_display.columns]
                                    if cols_in_df_to_round: results_df_s2s_display[cols_in_df_to_round] = results_df_s2s_display[cols_in_df_to_round].round(3)
                                    st.dataframe(results_df_s2s_display, hide_index=True)
                                except Exception as e_round_display:
                                     st.warning(f"Errore arrotondamento tabella: {e_round_display}")
                                     st.dataframe(results_df_s2s_display, hide_index=True) # Mostra comunque

                                # Link Download (usa df originale non rinominato)
                                st.markdown(get_table_download_link(results_df_s2s, f"simulazione_seq2seq_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                                st.subheader('📈 Grafici Previsioni Simulate (Seq2Seq)')
                                figs_sim_s2s = plot_predictions(predictions_s2s, active_config, start_pred_time_s2s)
                                num_graph_cols = min(len(figs_sim_s2s), 3) # Max 3 colonne
                                sim_cols = st.columns(num_graph_cols)
                                for i, fig_sim in enumerate(figs_sim_s2s):
                                   with sim_cols[i % num_graph_cols]:
                                        target_col_name = target_columns_model[i]
                                        s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                        st.plotly_chart(fig_sim, use_container_width=True)
                                        st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_s2s_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                           else:
                                 st.error("Predizione simulazione Seq2Seq fallita o risultato non valido.")


    # --- Logica per Metodi LSTM Standard ---
    elif sim_method == 'Manuale (Valori Costanti)':
         st.subheader(f'Inserisci valori costanti per i {input_steps} steps di input (LSTM)')
         st.caption(f"Questi valori saranno ripetuti per tutta la finestra di input ({input_steps*0.5:.1f}h).")
         temp_sim_values_lstm = {}
         cols_manual_lstm = st.columns(3)
         col_idx_manual = 0
         input_valid_manual = True
         for feature in feature_columns_current_model:
             with cols_manual_lstm[col_idx_manual % 3]:
                 label = get_station_label(feature, short=True)
                 unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', feature)
                 unit = f"({unit_match.group(1) or unit_match.group(2)})" if unit_match else ""
                 fmt = "%.1f" if 'pioggia' in feature.lower() or 'umidit' in feature.lower() else "%.2f"
                 step = 0.5 if 'pioggia' in feature.lower() else (1.0 if 'umidit' in feature.lower() else 0.1)
                 try:
                     val = st.number_input(f"{label} {unit}", key=f"man_{feature}", format=fmt, step=step)
                     temp_sim_values_lstm[feature] = float(val)
                 except Exception as e_num_input:
                     st.error(f"Valore non valido per {label}: {e_num_input}")
                     input_valid_manual = False
             col_idx_manual += 1

         # --- Bottone Esecuzione Specifico per LSTM Manuale ---
         sim_data_input_manual = None # Prepara variabile per dati
         if input_valid_manual:
             try:
                 # Assicura ordine corretto delle feature
                 ordered_values = [temp_sim_values_lstm[feature] for feature in feature_columns_current_model]
                 sim_data_input_manual = np.tile(ordered_values, (input_steps, 1)).astype(float)
             except KeyError as ke: st.error(f"Errore: Feature '{ke}' mancante nella configurazione manuale."); input_valid_manual = False
             except Exception as e: st.error(f"Errore creazione dati costanti: {e}"); input_valid_manual = False

         input_ready_manual = sim_data_input_manual is not None and input_valid_manual
         if st.button('🚀 Esegui Simulazione LSTM (Manuale)', type="primary", disabled=(not input_ready_manual), key="sim_run_exec_lstm_manual"):
             if input_ready_manual:
                  with st.spinner('Simulazione LSTM (Manuale) in corso...'):
                       # Recupera scalers LSTM dalla tupla
                       scaler_features_lstm, scaler_targets_lstm = active_scalers
                       predictions_sim_lstm = predict(active_model, sim_data_input_manual, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                  # ... (Visualizzazione risultati come nel blocco Seq2Seq ma con predictions_sim_lstm) ...
                  if predictions_sim_lstm is not None:
                       output_steps_actual = predictions_sim_lstm.shape[0]
                       total_hours_output_actual = output_steps_actual * 0.5
                       st.subheader(f'📊 Risultato Simulazione LSTM (Manuale): Prossime {total_hours_output_actual:.1f} ore ({output_steps_actual} steps)')
                       start_pred_time_lstm = datetime.now(italy_tz) # Usa ora corrente
                       # ... (Codice visualizzazione tabella/grafici) ...
                       # Visualizza tabella e grafici (usa plot_predictions)
                       pred_times_lstm = [start_pred_time_lstm + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]
                       results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model)
                       results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                       # Rinomina per display
                       rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'}
                       original_to_renamed_map_lstm = {}
                       for col in target_columns_model:
                           unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit = f"({unit_match.group(1) or unit_match.group(2)})" if unit_match else ""
                           new_name = f"{get_station_label(col, short=True)} {unit}".strip(); count = 1; final_name = new_name
                           while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                           rename_dict_lstm[col] = final_name; original_to_renamed_map_lstm[col] = final_name
                       results_df_lstm_display = results_df_lstm.copy()
                       results_df_lstm_display.rename(columns=rename_dict_lstm, inplace=True)
                       numeric_cols_renamed_lstm = list(original_to_renamed_map_lstm.values())
                       try:
                           cols_in_df_to_round = [col for col in numeric_cols_renamed_lstm if col in results_df_lstm_display.columns]
                           if cols_in_df_to_round: results_df_lstm_display[cols_in_df_to_round] = results_df_lstm_display[cols_in_df_to_round].round(3)
                           st.dataframe(results_df_lstm_display, hide_index=True)
                       except Exception as e_round_display: st.warning(f"Err Tab:{e_round_display}"); st.dataframe(results_df_lstm_display, hide_index=True)
                       st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_manuale_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                       # Grafici
                       st.subheader('📈 Grafici Previsioni Simulate (LSTM Manuale)')
                       figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, start_pred_time_lstm)
                       num_graph_cols_lstm = min(len(figs_sim_lstm), 3)
                       sim_cols_lstm = st.columns(num_graph_cols_lstm)
                       for i, fig_sim in enumerate(figs_sim_lstm):
                           with sim_cols_lstm[i % num_graph_cols_lstm]:
                                target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                st.plotly_chart(fig_sim, use_container_width=True)
                                st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_manuale_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                  else: st.error("Predizione simulazione LSTM (Manuale) fallita.")
             else: st.error("Dati input manuali non pronti o invalidi.")


    elif sim_method == 'Importa da Google Sheet (Ultime Ore)':
         st.subheader(f'Importa gli ultimi {input_steps} steps da Google Sheet (LSTM)')
         sheet_url_sim_lstm = st.text_input("URL Foglio Google storico", f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/edit", key="sim_gsheet_url_lstm")
         sheet_id_sim_lstm = extract_sheet_id(sheet_url_sim_lstm)

         # Mappatura GSheet -> Colonne Feature Modello LSTM
         column_mapping_gsheet_to_lstm = {
             'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
             'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
             'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
             'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)',
             'Umidita\' Sensore 3452 (Montemurello)': HUMIDITY_COL_NAME,
             'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
             'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
             'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
             'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
             'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
             GSHEET_DATE_COL: st.session_state.date_col_name_csv
         }
         # Filtra mappatura solo per colonne richieste da feature_columns_current_model + data
         relevant_model_cols_lstm_map = feature_columns_current_model + [st.session_state.date_col_name_csv]
         column_mapping_gsheet_to_lstm_filtered = {
             gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_lstm.items()
             if model_col in relevant_model_cols_lstm_map
         }

         # Imputazione per feature LSTM non mappate
         lstm_features_set = set(feature_columns_current_model)
         mapped_lstm_features_target = set(column_mapping_gsheet_to_lstm_filtered.values()) - {st.session_state.date_col_name_csv}
         missing_lstm_features_in_map = list(lstm_features_set - mapped_lstm_features_target)
         imputed_values_sim_lstm_gs = {}
         if missing_lstm_features_in_map:
             st.warning(f"Feature LSTM non mappate da GSheet: {', '.join(missing_lstm_features_in_map)}. Saranno imputate.")
             for missing_f in missing_lstm_features_in_map:
                 default_val = 0.0
                 if data_ready_csv and missing_f in df_current_csv.columns and pd.notna(df_current_csv[missing_f].median()):
                     default_val = df_current_csv[missing_f].median()
                 imputed_values_sim_lstm_gs[missing_f] = default_val

         # Bottone Importazione
         fetch_error_gsheet_lstm = None
         if st.button("Importa e Prepara da Google Sheet (LSTM)", key="sim_run_gsheet_import_lstm"):
             if not sheet_id_sim_lstm: st.error("ID GSheet mancante."); fetch_error_gsheet_lstm = "ID GSheet mancante."
             else:
                 fetch_sim_gsheet_data.clear()
                 with st.spinner("Recupero dati storici LSTM da GSheet..."):
                     required_cols_lstm_fetch = feature_columns_current_model + [st.session_state.date_col_name_csv]
                     imported_df_lstm, import_err_lstm, last_ts_lstm = fetch_sim_gsheet_data(
                          sheet_id_sim_lstm, input_steps, GSHEET_DATE_COL, GSHEET_DATE_FORMAT,
                          column_mapping_gsheet_to_lstm_filtered,
                          required_cols_lstm_fetch,
                          imputed_values_sim_lstm_gs
                     )
                 if import_err_lstm:
                     st.error(f"Recupero GSheet per LSTM fallito: {import_err_lstm}")
                     st.session_state.imported_sim_data_gs_df_lstm = None; fetch_error_gsheet_lstm = import_err_lstm
                 elif imported_df_lstm is not None:
                     try:
                         final_lstm_df = imported_df_lstm[feature_columns_current_model] # Seleziona solo feature input
                         st.success(f"Recuperate e processate {len(final_lstm_df)} righe LSTM da GSheet.")
                         st.session_state.imported_sim_data_gs_df_lstm = final_lstm_df
                         st.session_state.imported_sim_start_time_gs_lstm = last_ts_lstm if last_ts_lstm else datetime.now(italy_tz)
                         fetch_error_gsheet_lstm = None
                     except KeyError as e_cols_lstm:
                         st.error(f"Errore selezione colonne LSTM dopo fetch: {e_cols_lstm}")
                         st.session_state.imported_sim_data_gs_df_lstm = None; fetch_error_gsheet_lstm = f"Errore colonne: {e_cols_lstm}"
                 else: st.error("Recupero GSheet per LSTM non riuscito."); fetch_error_gsheet_lstm = "Errore sconosciuto."
                 # st.rerun() # Forzare rerun per aggiornare stato bottone esecuzione

         # Mostra stato caricamento dati LSTM GSheet
         imported_df_lstm_gs = st.session_state.get("imported_sim_data_gs_df_lstm", None)
         if imported_df_lstm_gs is not None:
             st.caption("Dati LSTM (Input) importati da Google Sheet.")
             with st.expander("Mostra dati importati (Input LSTM)"):
                 st.dataframe(imported_df_lstm_gs.round(3))

         # Bottone Esecuzione Simulazione LSTM GSheet
         sim_data_input_lstm_gs = None
         sim_start_time_lstm_gs = None
         if isinstance(imported_df_lstm_gs, pd.DataFrame):
             try:
                 # Assicurati che le colonne siano nell'ordine corretto richiesto dal modello/scaler
                 sim_data_input_lstm_gs_ordered = imported_df_lstm_gs[feature_columns_current_model]
                 sim_data_input_lstm_gs = sim_data_input_lstm_gs_ordered.astype(float).values
                 sim_start_time_lstm_gs = st.session_state.get("imported_sim_start_time_gs_lstm", datetime.now(italy_tz))
                 if sim_data_input_lstm_gs.shape != (input_steps, len(feature_columns_current_model)):
                     st.error(f"Shape dati LSTM GSheet errata: {sim_data_input_lstm_gs.shape}, attesa ({input_steps}, {len(feature_columns_current_model)})")
                     sim_data_input_lstm_gs = None # Shape errata
             except KeyError as e_key_gs: st.error(f"Colonna mancante nei dati LSTM GSheet: {e_key_gs}"); sim_data_input_lstm_gs = None
             except Exception as e_prep_gs: st.error(f"Errore preparazione dati LSTM GSheet: {e_prep_gs}"); sim_data_input_lstm_gs = None

         input_ready_lstm_gs = sim_data_input_lstm_gs is not None and not np.isnan(sim_data_input_lstm_gs).any()

         if st.button('🚀 Esegui Simulazione LSTM (GSheet)', type="primary", disabled=(not input_ready_lstm_gs), key="sim_run_exec_lstm_gsheet"):
              if input_ready_lstm_gs:
                  with st.spinner('Simulazione LSTM (GSheet) in corso...'):
                       scaler_features_lstm, scaler_targets_lstm = active_scalers
                       predictions_sim_lstm = predict(active_model, sim_data_input_lstm_gs, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                  # ... (Visualizzazione risultati) ...
                  if predictions_sim_lstm is not None:
                       output_steps_actual = predictions_sim_lstm.shape[0]
                       total_hours_output_actual = output_steps_actual * 0.5
                       st.subheader(f'📊 Risultato Simulazione LSTM (GSheet): Prossime {total_hours_output_actual:.1f} ore ({output_steps_actual} steps)')
                       st.caption(f"Previsione calcolata a partire da: {sim_start_time_lstm_gs.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                       # ... (Codice visualizzazione tabella/grafici come per LSTM manuale, usando sim_start_time_lstm_gs) ...
                       pred_times_lstm = [sim_start_time_lstm_gs + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]
                       results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                       rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'}; original_to_renamed_map_lstm = {}
                       for col in target_columns_model:
                           unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit = f"({unit_match.group(1) or unit_match.group(2)})" if unit_match else ""
                           new_name = f"{get_station_label(col, short=True)} {unit}".strip(); count = 1; final_name = new_name
                           while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                           rename_dict_lstm[col] = final_name; original_to_renamed_map_lstm[col] = final_name
                       results_df_lstm_display = results_df_lstm.copy(); results_df_lstm_display.rename(columns=rename_dict_lstm, inplace=True)
                       numeric_cols_renamed_lstm = list(original_to_renamed_map_lstm.values())
                       try:
                           cols_in_df_to_round = [col for col in numeric_cols_renamed_lstm if col in results_df_lstm_display.columns]
                           if cols_in_df_to_round: results_df_lstm_display[cols_in_df_to_round] = results_df_lstm_display[cols_in_df_to_round].round(3)
                           st.dataframe(results_df_lstm_display, hide_index=True)
                       except Exception as e_round_display: st.warning(f"Err Tab:{e_round_display}"); st.dataframe(results_df_lstm_display, hide_index=True)
                       st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_gsheet_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                       st.subheader('📈 Grafici Previsioni Simulate (LSTM GSheet)')
                       figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, sim_start_time_lstm_gs)
                       num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                       for i, fig_sim in enumerate(figs_sim_lstm):
                           with sim_cols_lstm[i % num_graph_cols_lstm]:
                                target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                st.plotly_chart(fig_sim, use_container_width=True)
                                st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_gsheet_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                  else: st.error("Predizione simulazione LSTM (GSheet) fallita.")
              else: st.error("Dati input da GSheet non pronti o non importati.")


    elif sim_method == 'Orario Dettagliato (Tabella)':
         st.subheader(f'Inserisci dati orari dettagliati per i {input_steps} steps di input (LSTM)')
         st.caption(f"Inserisci i valori per ciascuno dei {input_steps} passi temporali ({input_steps*0.5:.1f}h) precedenti il momento della simulazione.")

         # Crea DataFrame iniziale per l'editor
         editor_df_initial = pd.DataFrame(index=range(input_steps), columns=feature_columns_current_model)
         # Prova a pre-compilare con gli ultimi dati CSV se disponibili
         if data_ready_csv and len(df_current_csv) >= input_steps:
             try:
                last_data_csv = df_current_csv[feature_columns_current_model].iloc[-input_steps:].reset_index(drop=True)
                editor_df_initial = last_data_csv.astype(float).round(2)
             except Exception as e_fill_csv:
                st.warning(f"Impossibile pre-compilare tabella con dati CSV: {e_fill_csv}")
                editor_df_initial = editor_df_initial.fillna(0.0) # Fallback a 0
         else:
             editor_df_initial = editor_df_initial.fillna(0.0) # Fallback a 0 se CSV non disponibile/sufficiente

         # Data Editor
         lstm_editor_key = "lstm_editor_sim"
         edited_lstm_df = st.data_editor(
             editor_df_initial,
             key=lstm_editor_key,
             num_rows="fixed",
             use_container_width=True,
             column_config={ # Configurazione dinamica
                 col: st.column_config.NumberColumn(
                     label=get_station_label(col, short=True),
                     help=f"Valore storico per {col}",
                     min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else None,
                     max_value=100.0 if 'umidit' in col.lower() else None,
                     format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f",
                     step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1),
                     required=True
                 ) for col in feature_columns_current_model
             }
         )

         # Validazione Editor
         sim_data_input_lstm_editor = None
         validation_passed_editor = False
         if edited_lstm_df is not None and not edited_lstm_df.isnull().any().any() and edited_lstm_df.shape == (input_steps, len(feature_columns_current_model)):
             try:
                 # Assicura ordine colonne e tipo float
                 sim_data_input_lstm_editor = edited_lstm_df[feature_columns_current_model].astype(float).values
                 validation_passed_editor = True
                 st.caption("Dati tabella inseriti correttamente.")
             except KeyError as e_key_edit: st.error(f"Errore colonna tabella: {e_key_edit}")
             except Exception as e_conv_edit: st.error(f"Errore conversione dati tabella: {e_conv_edit}")
         else:
             st.warning("Completa o correggi la tabella. Tutti i valori devono essere numerici.")

         input_ready_lstm_editor = sim_data_input_lstm_editor is not None and validation_passed_editor
         if st.button('🚀 Esegui Simulazione LSTM (Tabella)', type="primary", disabled=(not input_ready_lstm_editor), key="sim_run_exec_lstm_editor"):
              if input_ready_lstm_editor:
                  with st.spinner('Simulazione LSTM (Tabella) in corso...'):
                       scaler_features_lstm, scaler_targets_lstm = active_scalers
                       predictions_sim_lstm = predict(active_model, sim_data_input_lstm_editor, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                  # ... (Visualizzazione risultati) ...
                  if predictions_sim_lstm is not None:
                       output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                       st.subheader(f'📊 Risultato Simulazione LSTM (Tabella): Prossime {total_hours_output_actual:.1f} ore ({output_steps_actual} steps)')
                       start_pred_time_lstm = datetime.now(italy_tz) # Usa ora corrente
                       # ... (Codice visualizzazione tabella/grafici come LSTM Manuale) ...
                       pred_times_lstm = [start_pred_time_lstm + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]; results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                       rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'}; original_to_renamed_map_lstm = {}
                       for col in target_columns_model:
                           unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit = f"({unit_match.group(1) or unit_match.group(2)})" if unit_match else ""
                           new_name = f"{get_station_label(col, short=True)} {unit}".strip(); count = 1; final_name = new_name
                           while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                           rename_dict_lstm[col] = final_name; original_to_renamed_map_lstm[col] = final_name
                       results_df_lstm_display = results_df_lstm.copy(); results_df_lstm_display.rename(columns=rename_dict_lstm, inplace=True)
                       numeric_cols_renamed_lstm = list(original_to_renamed_map_lstm.values())
                       try:
                           cols_in_df_to_round = [col for col in numeric_cols_renamed_lstm if col in results_df_lstm_display.columns];
                           if cols_in_df_to_round: results_df_lstm_display[cols_in_df_to_round] = results_df_lstm_display[cols_in_df_to_round].round(3)
                           st.dataframe(results_df_lstm_display, hide_index=True)
                       except Exception as e_round_display: st.warning(f"Err Tab:{e_round_display}"); st.dataframe(results_df_lstm_display, hide_index=True)
                       st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_tabella_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                       st.subheader('📈 Grafici Previsioni Simulate (LSTM Tabella)')
                       figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, start_pred_time_lstm); num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                       for i, fig_sim in enumerate(figs_sim_lstm):
                           with sim_cols_lstm[i % num_graph_cols_lstm]:
                                target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                st.plotly_chart(fig_sim, use_container_width=True)
                                st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_tabella_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                  else: st.error("Predizione simulazione LSTM (Tabella) fallita.")
              else: st.error("Dati input da tabella non pronti o invalidi.")


    elif sim_method == 'Usa Ultime Ore da CSV Caricato':
         st.subheader(f"Usa gli ultimi {input_steps} steps dai dati CSV caricati (LSTM)")
         if not data_ready_csv: st.error("Dati CSV non caricati."); st.stop()
         if len(df_current_csv) < input_steps: st.error(f"Dati CSV insufficienti ({len(df_current_csv)} righe), richieste {input_steps}."); st.stop()

         # Estrazione dati CSV
         sim_data_input_lstm_csv = None
         sim_start_time_lstm_csv = None
         try:
             latest_csv_data_df = df_current_csv.iloc[-input_steps:]
             # Verifica colonne necessarie
             missing_cols_csv = [col for col in feature_columns_current_model if col not in latest_csv_data_df.columns]
             if missing_cols_csv:
                 st.error(f"Colonne modello LSTM mancanti nel CSV: {', '.join(missing_cols_csv)}")
             else:
                 # Assicura ordine colonne e tipo float
                 sim_data_input_lstm_csv_ordered = latest_csv_data_df[feature_columns_current_model]
                 sim_data_input_lstm_csv = sim_data_input_lstm_csv_ordered.astype(float).values
                 # Recupera timestamp dell'ultimo dato usato
                 last_csv_timestamp = df_current_csv[date_col_name_csv].iloc[-1]
                 if pd.notna(last_csv_timestamp): sim_start_time_lstm_csv = last_csv_timestamp
                 else: sim_start_time_lstm_csv = datetime.now(italy_tz) # Fallback
                 st.caption("Dati CSV pronti per la simulazione.")
                 with st.expander("Mostra dati CSV usati (Input LSTM)"):
                     st.dataframe(latest_csv_data_df[[date_col_name_csv] + feature_columns_current_model].round(3))

         except Exception as e_prep_csv:
             st.error(f"Errore preparazione dati da CSV: {e_prep_csv}")
             sim_data_input_lstm_csv = None

         input_ready_lstm_csv = sim_data_input_lstm_csv is not None and not np.isnan(sim_data_input_lstm_csv).any()
         if st.button('🚀 Esegui Simulazione LSTM (CSV)', type="primary", disabled=(not input_ready_lstm_csv), key="sim_run_exec_lstm_csv"):
             if input_ready_lstm_csv:
                  with st.spinner('Simulazione LSTM (CSV) in corso...'):
                       scaler_features_lstm, scaler_targets_lstm = active_scalers
                       predictions_sim_lstm = predict(active_model, sim_data_input_lstm_csv, scaler_features_lstm, scaler_targets_lstm, active_config, active_device)
                  # ... (Visualizzazione risultati) ...
                  if predictions_sim_lstm is not None:
                       output_steps_actual = predictions_sim_lstm.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                       st.subheader(f'📊 Risultato Simulazione LSTM (CSV): Prossime {total_hours_output_actual:.1f} ore ({output_steps_actual} steps)')
                       st.caption(f"Previsione calcolata a partire da: {sim_start_time_lstm_csv.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                       # ... (Codice visualizzazione tabella/grafici come LSTM Manuale, usando sim_start_time_lstm_csv) ...
                       pred_times_lstm = [sim_start_time_lstm_csv + timedelta(minutes=30 * (i + 1)) for i in range(output_steps_actual)]; results_df_lstm = pd.DataFrame(predictions_sim_lstm, columns=target_columns_model); results_df_lstm.insert(0, 'Ora Prevista', [t.strftime('%d/%m %H:%M') for t in pred_times_lstm])
                       rename_dict_lstm = {'Ora Prevista': 'Ora Prevista'}; original_to_renamed_map_lstm = {}
                       for col in target_columns_model:
                           unit_match = re.search(r'\[(.*?)\]|\((.*?)\)', col); unit = f"({unit_match.group(1) or unit_match.group(2)})" if unit_match else ""
                           new_name = f"{get_station_label(col, short=True)} {unit}".strip(); count = 1; final_name = new_name
                           while final_name in rename_dict_lstm.values(): count += 1; final_name = f"{new_name}_{count}"
                           rename_dict_lstm[col] = final_name; original_to_renamed_map_lstm[col] = final_name
                       results_df_lstm_display = results_df_lstm.copy(); results_df_lstm_display.rename(columns=rename_dict_lstm, inplace=True)
                       numeric_cols_renamed_lstm = list(original_to_renamed_map_lstm.values())
                       try:
                           cols_in_df_to_round = [col for col in numeric_cols_renamed_lstm if col in results_df_lstm_display.columns];
                           if cols_in_df_to_round: results_df_lstm_display[cols_in_df_to_round] = results_df_lstm_display[cols_in_df_to_round].round(3)
                           st.dataframe(results_df_lstm_display, hide_index=True)
                       except Exception as e_round_display: st.warning(f"Err Tab:{e_round_display}"); st.dataframe(results_df_lstm_display, hide_index=True)
                       st.markdown(get_table_download_link(results_df_lstm, f"simulazione_lstm_csv_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)
                       st.subheader('📈 Grafici Previsioni Simulate (LSTM CSV)')
                       figs_sim_lstm = plot_predictions(predictions_sim_lstm, active_config, sim_start_time_lstm_csv); num_graph_cols_lstm = min(len(figs_sim_lstm), 3); sim_cols_lstm = st.columns(num_graph_cols_lstm)
                       for i, fig_sim in enumerate(figs_sim_lstm):
                           with sim_cols_lstm[i % num_graph_cols_lstm]:
                                target_col_name = target_columns_model[i]; s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_col_name, short=False))
                                st.plotly_chart(fig_sim, use_container_width=True)
                                st.markdown(get_plotly_download_link(fig_sim, f"grafico_sim_lstm_csv_{s_name_file}_{datetime.now().strftime('%Y%m%d_%H%M')}"), unsafe_allow_html=True)
                  else: st.error("Predizione simulazione LSTM (CSV) fallita.")
             else: st.error("Dati input da CSV non pronti o invalidi.")


# --- PAGINA ANALISI DATI STORICI (Invariata) ---
elif page == 'Analisi Dati Storici':
    st.header('🔎 Analisi Dati Storici (da file CSV)')
    if not data_ready_csv:
        st.warning("⚠️ Dati Storici CSV non disponibili. Caricane uno dalla sidebar.")
    else:
        st.info(f"Dataset CSV caricato: **{len(df_current_csv)}** righe.")
        st.caption(f"Periodo: da **{df_current_csv[date_col_name_csv].min().strftime('%d/%m/%Y %H:%M')}** a **{df_current_csv[date_col_name_csv].max().strftime('%d/%m/%Y %H:%M')}**")

        st.subheader("Esplora Dati")
        st.dataframe(df_current_csv.head())
        st.markdown(get_table_download_link(df_current_csv, f"dati_storici_completi_{datetime.now().strftime('%Y%m%d')}.csv"), unsafe_allow_html=True)

        st.divider()
        st.subheader("Statistiche Descrittive")
        numeric_cols_analysis = df_current_csv.select_dtypes(include=np.number).columns
        st.dataframe(df_current_csv[numeric_cols_analysis].describe().round(2))

        st.divider()
        st.subheader("Visualizzazione Temporale")
        cols_to_plot = st.multiselect(
            "Seleziona colonne da visualizzare:",
            options=numeric_cols_analysis.tolist(),
            default=numeric_cols_analysis[:min(len(numeric_cols_analysis), 5)].tolist(), # Default prime 5
            key="analysis_plot_select"
        )
        if cols_to_plot:
            fig_analysis = go.Figure()
            for col in cols_to_plot:
                fig_analysis.add_trace(go.Scatter(x=df_current_csv[date_col_name_csv], y=df_current_csv[col], mode='lines', name=get_station_label(col, short=True)))
            fig_analysis.update_layout(
                title="Andamento Temporale Selezionato",
                xaxis_title="Data e Ora",
                yaxis_title="Valore",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig_analysis, use_container_width=True)
            analysis_filename_base = f"analisi_temporale_{datetime.now().strftime('%Y%m%d_%H%M')}"
            st.markdown(get_plotly_download_link(fig_analysis, analysis_filename_base), unsafe_allow_html=True)

        st.divider()
        st.subheader("Matrice di Correlazione")
        if len(numeric_cols_analysis) > 1:
            corr_matrix = df_current_csv[numeric_cols_analysis].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='viridis', # o 'RdBu', 'Viridis'
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(
                title='Matrice di Correlazione tra Variabili Numeriche',
                xaxis_tickangle=-45,
                height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Seleziona almeno due colonne numeriche per la matrice di correlazione.")


# --- PAGINA ALLENAMENTO MODELLO (Logica Biforcata LSTM/Seq2Seq) ---
elif page == 'Allenamento Modello':
    st.header('🎓 Allenamento Nuovo Modello')
    if not data_ready_csv:
        st.warning("⚠️ Dati Storici CSV non disponibili per allenare. Caricane uno dalla sidebar.")
        st.stop() # Blocca esecuzione pagina

    st.success(f"Dati CSV disponibili per l'allenamento: {len(df_current_csv)} righe.")
    st.subheader('Configurazione Addestramento')

    # Selezione Tipo Modello da Allenare
    train_model_type = st.radio("Tipo di Modello da Allenare:", ["LSTM Standard", "Seq2Seq (Encoder-Decoder)"], key="train_select_type", horizontal=True)

    # Input nome modello (comune)
    default_save_name = f"modello_{train_model_type.split()[0].lower()}_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
    save_name_input = st.text_input("Nome base per salvare il modello", default_save_name, key="train_save_filename")
    save_name = re.sub(r'[^\w-]', '_', save_name_input).strip('_') or "modello_default"
    if save_name != save_name_input: st.caption(f"Nome file corretto in: `{save_name}`")

    # Crea directory modelli se non esiste
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Parametri Specifici per Tipo ---
    if train_model_type == "LSTM Standard":
        st.markdown("**1. Seleziona Feature Input (LSTM):**")
        all_features_lstm = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
        default_features_lstm = [f for f in st.session_state.feature_columns if f in all_features_lstm] # Usa globali come default se presenti
        selected_features_train_lstm = st.multiselect("Seleziona Feature Input LSTM:",
                                                        options=all_features_lstm,
                                                        default=default_features_lstm,
                                                        key="train_lstm_feat")

        st.markdown("**2. Seleziona Target Output (LSTM - Livelli):**")
        level_options_lstm = [f for f in all_features_lstm if 'livello' in f.lower() or '[m]' in f.lower()]
        default_targets_lstm = level_options_lstm[:1] # Default primo livello trovato
        selected_targets_train_lstm = st.multiselect("Seleziona Target (Livelli) da prevedere:",
                                                      options=level_options_lstm,
                                                      default=default_targets_lstm,
                                                      key="train_lstm_target")

        st.markdown("**3. Parametri Modello e Training (LSTM):**")
        with st.expander("Parametri LSTM", expanded=True):
            c1_lstm, c2_lstm, c3_lstm = st.columns(3)
            with c1_lstm:
                iw_t_lstm = st.number_input("Finestra Input (steps 30min)", min_value=2, value=48, step=2, key="t_lstm_in") # Min 1 ora
                ow_t_lstm = st.number_input("Finestra Output (steps 30min)", min_value=1, value=24, step=1, key="t_lstm_out") # Min 1 step
                vs_t_lstm = st.slider("% Validazione", 0, 50, 20, 1, key="t_lstm_vs")
            with c2_lstm:
                hs_t_lstm = st.number_input("Hidden Size", min_value=8, value=128, step=8, key="t_lstm_hs")
                nl_t_lstm = st.number_input("Layers", min_value=1, value=2, step=1, key="t_lstm_nl")
                dr_t_lstm = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_lstm_dr")
            with c3_lstm:
                lr_t_lstm = st.number_input("Learning Rate", min_value=1e-6, value=0.001, format="%.5f", step=1e-4, key="t_lstm_lr")
                bs_t_lstm = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_lstm_bs")
                ep_t_lstm = st.number_input("Epoche", min_value=1, value=50, step=5, key="t_lstm_ep")

            device_option_lstm = st.radio("Device:", ['Auto', 'Forza CPU'], index=0, key='train_device_lstm', horizontal=True)
            save_choice_lstm = st.radio("Salva:", ['Migliore (Val)', 'Finale'], index=0, key='train_save_lstm', horizontal=True,
                                         help="Salva il modello con la loss di validazione minima o quello dell'ultima epoca.")

        # --- Bottone Avvio Training LSTM ---
        ready_to_train_lstm = bool(save_name and selected_features_train_lstm and selected_targets_train_lstm)
        if st.button("▶️ Addestra Modello LSTM", type="primary", disabled=not ready_to_train_lstm, key="train_run_lstm"):
             st.info(f"Avvio addestramento LSTM Standard per '{save_name}'...")
             # 1. Prepara Dati LSTM
             with st.spinner("Preparazione dati LSTM..."):
                  # Converti steps in ore per la funzione prepare_training_data
                  input_hours_lstm = iw_t_lstm / 2
                  output_hours_lstm = ow_t_lstm / 2
                  X_tr, y_tr, X_v, y_v, sc_f, sc_t = prepare_training_data(
                       df_current_csv.copy(), # Passa una copia per evitare modifiche
                       selected_features_train_lstm, selected_targets_train_lstm,
                       int(input_hours_lstm), int(output_hours_lstm), # Passa ORE
                       vs_t_lstm
                  )
             if X_tr is None: st.error("Preparazione dati LSTM fallita."); st.stop()

             # 2. Allena Modello LSTM
             trained_model_lstm, _, _ = train_model(
                   X_tr, y_tr, X_v, y_v, len(selected_features_train_lstm), len(selected_targets_train_lstm), ow_t_lstm, # Passa output STEPS
                   hs_t_lstm, nl_t_lstm, ep_t_lstm, bs_t_lstm, lr_t_lstm, dr_t_lstm,
                   save_strategy=('migliore' if 'Migliore' in save_choice_lstm else 'finale'),
                   preferred_device=('auto' if 'Auto' in device_option_lstm else 'cpu')
             )

             # 3. Salva Risultati LSTM
             if trained_model_lstm and sc_f and sc_t:
                  st.success("Addestramento LSTM completato!")
                  try:
                      base_path = os.path.join(MODELS_DIR, save_name)
                      m_path = f"{base_path}.pth"
                      torch.save(trained_model_lstm.state_dict(), m_path)

                      sf_path = f"{base_path}_features.joblib"
                      st_path = f"{base_path}_targets.joblib"
                      joblib.dump(sc_f, sf_path); joblib.dump(sc_t, st_path)

                      c_path = f"{base_path}.json"
                      config_save_lstm = {
                          "model_type": "LSTM", # Importante!
                          "input_window": iw_t_lstm, # Salva STEPS
                          "output_window": ow_t_lstm, # Salva STEPS
                          "hidden_size": hs_t_lstm, "num_layers": nl_t_lstm, "dropout": dr_t_lstm,
                          "feature_columns": selected_features_train_lstm,
                          "target_columns": selected_targets_train_lstm,
                          "training_date": datetime.now(italy_tz).isoformat(),
                          "val_split_percent": vs_t_lstm,
                          "learning_rate": lr_t_lstm,
                          "batch_size": bs_t_lstm,
                          "epochs": ep_t_lstm,
                          "display_name": save_name, # Usa lo stesso nome del file per default
                      }
                      with open(c_path, 'w', encoding='utf-8') as f: json.dump(config_save_lstm, f, indent=4)
                      st.success(f"Modello LSTM '{save_name}' salvato correttamente in '{MODELS_DIR}'.")
                      # Mostra link download
                      st.subheader("⬇️ Download File Modello LSTM")
                      col_dl_lstm1, col_dl_lstm2, col_dl_lstm3 = st.columns(3)
                      with col_dl_lstm1: st.markdown(get_download_link_for_file(m_path), unsafe_allow_html=True)
                      with col_dl_lstm2: st.markdown(get_download_link_for_file(c_path), unsafe_allow_html=True)
                      with col_dl_lstm3: st.markdown(get_download_link_for_file(sf_path, "Scaler Features"), unsafe_allow_html=True)
                      col_dl_lstm4, _, _ = st.columns(3)
                      with col_dl_lstm4: st.markdown(get_download_link_for_file(st_path, "Scaler Target"), unsafe_allow_html=True)
                      # Forza refresh della cache dei modelli disponibili
                      find_available_models.clear()

                  except Exception as e_save:
                      st.error(f"Errore durante il salvataggio del modello LSTM: {e_save}")
                      st.error(traceback.format_exc())

             # --- FIX: Questo else deve essere allineato con 'if trained_model_lstm and sc_f and sc_t:' ---
             else:
                 st.error("Addestramento LSTM fallito o scaler non generati.")

    elif train_model_type == "Seq2Seq (Encoder-Decoder)":
         st.markdown("**1. Seleziona Feature Storiche (Input Encoder):**")
         features_present_in_csv_s2s = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
         default_past_features_s2s = [f for f in st.session_state.feature_columns if f in features_present_in_csv_s2s]
         selected_past_features_s2s = st.multiselect("TUTTE le feature da considerare dal passato:",
                                                     options=features_present_in_csv_s2s,
                                                     default=default_past_features_s2s,
                                                     key="train_s2s_past_feat")

         st.markdown("**2. Seleziona Feature Forecast (Input Decoder):**")
         # Opzioni sono solo quelle selezionate come past features
         options_forecast = selected_past_features_s2s
         default_forecast_cols = [f for f in options_forecast if 'pioggia' in f.lower() or 'cumulata' in f.lower() or f == HUMIDITY_COL_NAME]
         selected_forecast_features_s2s = st.multiselect("Feature per cui fornirai PREVISIONI future:",
                                                          options=options_forecast,
                                                          default=default_forecast_cols,
                                                          key="train_s2s_forecast_feat",
                                                          help="Seleziona le feature che userai come input futuri nel decoder.")

         st.markdown("**3. Seleziona Target Output (Livelli):**")
         # Opzioni sono solo livelli tra le past features
         level_options_s2s = [f for f in selected_past_features_s2s if 'livello' in f.lower() or '[m]' in f.lower()]
         default_targets_s2s = level_options_s2s[:1] # Default primo livello trovato
         selected_targets_s2s = st.multiselect("Target (Livelli) da prevedere:",
                                               options=level_options_s2s,
                                               default=default_targets_s2s,
                                               key="train_s2s_target_feat")

         st.markdown("**4. Parametri Finestre e Training (Seq2Seq):**")
         with st.expander("Parametri Seq2Seq", expanded=True):
             c1_s2s, c2_s2s, c3_s2s = st.columns(3)
             with c1_s2s:
                 iw_steps_s2s = st.number_input("Input Storico (steps 30min)", min_value=2, value=48, step=2, key="t_s2s_in")
                 fw_steps_s2s = st.number_input("Input Forecast (steps 30min)", min_value=1, value=24, step=1, key="t_s2s_fore", help="Numero di passi futuri forniti al decoder.")
                 ow_steps_s2s = st.number_input("Output Previsione (steps 30min)", min_value=1, value=24, step=1, key="t_s2s_out", help="Numero di passi futuri da prevedere.")
                 if ow_steps_s2s > fw_steps_s2s: st.warning("Output Steps > Forecast Steps. Il modello ripeterà l'ultimo forecast durante la predizione.")
                 vs_t_s2s = st.slider("% Validazione", 0, 50, 20, 1, key="t_s2s_vs")
             with c2_s2s:
                 hs_t_s2s = st.number_input("Hidden Size (Encoder/Decoder)", min_value=8, value=128, step=8, key="t_s2s_hs")
                 nl_t_s2s = st.number_input("Layers (Encoder/Decoder)", min_value=1, value=2, step=1, key="t_s2s_nl")
                 dr_t_s2s = st.slider("Dropout (Encoder/Decoder)", 0.0, 0.7, 0.2, 0.05, key="t_s2s_dr")
             with c3_s2s:
                 lr_t_s2s = st.number_input("Learning Rate", min_value=1e-6, value=0.001, format="%.5f", step=1e-4, key="t_s2s_lr")
                 bs_t_s2s = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], 32, key="t_s2s_bs")
                 ep_t_s2s = st.number_input("Epoche", min_value=1, value=50, step=5, key="t_s2s_ep")

             device_option_s2s = st.radio("Device:", ['Auto', 'Forza CPU'], index=0, key='train_device_s2s', horizontal=True)
             save_choice_s2s = st.radio("Salva:", ['Migliore (Val)', 'Finale'], index=0, key='train_save_s2s', horizontal=True)


         # --- Bottone Avvio Training Seq2Seq ---
         ready_to_train_s2s = bool(save_name and selected_past_features_s2s and selected_forecast_features_s2s and selected_targets_s2s)
         if st.button("▶️ Addestra Modello Seq2Seq", type="primary", disabled=not ready_to_train_s2s, key="train_run_s2s"):
             st.info(f"Avvio addestramento Seq2Seq per '{save_name}'...")
             # 1. Prepara Dati Seq2Seq
             with st.spinner("Preparazione dati Seq2Seq..."):
                  data_tuple = prepare_training_data_seq2seq(
                       df_current_csv.copy(), # Passa copia
                       selected_past_features_s2s, selected_forecast_features_s2s, selected_targets_s2s,
                       iw_steps_s2s, fw_steps_s2s, ow_steps_s2s, vs_t_s2s
                  )
             if data_tuple is None or len(data_tuple) != 9 or data_tuple[0] is None: st.error("Preparazione dati Seq2Seq fallita."); st.stop()
             (X_enc_tr, X_dec_tr, y_tar_tr, X_enc_v, X_dec_v, y_tar_v,
              sc_past, sc_fore, sc_tar) = data_tuple

             # Verifica che gli scaler siano stati fittati
             if not all([sc_past, sc_fore, sc_tar]):
                  st.error("Errore: Scaler non generati correttamente durante la preparazione dati Seq2Seq.")
                  st.stop()

             # 2. Istanzia e Allena Modello Seq2Seq
             try:
                 enc = EncoderLSTM(len(selected_past_features_s2s), hs_t_s2s, nl_t_s2s, dr_t_s2s)
                 dec = DecoderLSTM(len(selected_forecast_features_s2s), hs_t_s2s, len(selected_targets_s2s), nl_t_s2s, dr_t_s2s)
             except Exception as e_init:
                 st.error(f"Errore inizializzazione modello Seq2Seq: {e_init}")
                 st.stop()

             # Passa encoder e decoder istanziati alla funzione di training
             trained_model_s2s, _, _ = train_model_seq2seq(
                  X_enc_tr, X_dec_tr, y_tar_tr, X_enc_v, X_dec_v, y_tar_v,
                  enc, dec, ow_steps_s2s, # Passa output STEPS
                  ep_t_s2s, bs_t_s2s, lr_t_s2s,
                  save_strategy=('migliore' if 'Migliore' in save_choice_s2s else 'finale'),
                  preferred_device=('auto' if 'Auto' in device_option_s2s else 'cpu')
             )

             # 3. Salva Risultati Seq2Seq
             if trained_model_s2s:
                  st.success("Addestramento Seq2Seq completato!")
                  try:
                      base_path = os.path.join(MODELS_DIR, save_name)
                      # Salva modello .pth (intero Seq2SeqHydro)
                      m_path = f"{base_path}.pth"
                      torch.save(trained_model_s2s.state_dict(), m_path)
                      # Salva i 3 scaler .joblib
                      sp_path = f"{base_path}_past_features.joblib"
                      sf_path = f"{base_path}_forecast_features.joblib"
                      st_path = f"{base_path}_targets.joblib"
                      joblib.dump(sc_past, sp_path); joblib.dump(sc_fore, sf_path); joblib.dump(sc_tar, st_path)
                      # Salva config .json specifico Seq2Seq
                      c_path = f"{base_path}.json"
                      config_save_s2s = {
                          "model_type": "Seq2Seq", # Importante!
                          "input_window_steps": iw_steps_s2s,
                          "forecast_window_steps": fw_steps_s2s,
                          "output_window_steps": ow_steps_s2s,
                          "hidden_size": hs_t_s2s, "num_layers": nl_t_s2s, "dropout": dr_t_s2s,
                          "all_past_feature_columns": selected_past_features_s2s,
                          "forecast_input_columns": selected_forecast_features_s2s,
                          "target_columns": selected_targets_s2s,
                          "training_date": datetime.now(italy_tz).isoformat(),
                          "val_split_percent": vs_t_s2s,
                          "learning_rate": lr_t_s2s,
                          "batch_size": bs_t_s2s,
                          "epochs": ep_t_s2s,
                          "display_name": save_name,
                      }
                      with open(c_path, 'w', encoding='utf-8') as f: json.dump(config_save_s2s, f, indent=4)
                      st.success(f"Modello Seq2Seq '{save_name}' salvato correttamente in '{MODELS_DIR}'.")
                      # Mostra link download per pth, 3 joblib, json
                      st.subheader("⬇️ Download File Modello Seq2Seq")
                      col_dl1, col_dl2, col_dl3 = st.columns(3)
                      with col_dl1: st.markdown(get_download_link_for_file(m_path), unsafe_allow_html=True)
                      with col_dl2: st.markdown(get_download_link_for_file(c_path), unsafe_allow_html=True)
                      with col_dl3: st.markdown(get_download_link_for_file(sp_path,"Scaler Passato"), unsafe_allow_html=True)
                      col_dl4, col_dl5, _ = st.columns(3)
                      with col_dl4: st.markdown(get_download_link_for_file(sf_path,"Scaler Forecast"), unsafe_allow_html=True)
                      with col_dl5: st.markdown(get_download_link_for_file(st_path,"Scaler Target"), unsafe_allow_html=True)
                      # Forza refresh della cache dei modelli disponibili
                      find_available_models.clear()

                  except Exception as e_save_s2s:
                      st.error(f"Errore durante il salvataggio del modello Seq2Seq: {e_save_s2s}")
                      st.error(traceback.format_exc())

             else: st.error("Addestramento Seq2Seq fallito.")


# --- Footer (invariato) ---
st.sidebar.divider()
st.sidebar.info('App Idrologica Dashboard & Predict © 2024')
