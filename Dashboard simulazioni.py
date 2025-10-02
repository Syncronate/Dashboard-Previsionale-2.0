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
from dilate_loss_src import dilate_loss
from numpy.lib.stride_tricks import sliding_window_view

# Aggiungi all'inizio del file, con gli altri import
try:
    from torch_geometric.nn import GCNConv
    from torch.nn import GRU
    PYG_AVAILABLE = True
except ImportError:
    # Nota: L'avviso all'utente verrà mostrato in seguito nella UI della sidebar
    # per garantire che venga eseguito nel contesto corretto di Streamlit.
    PYG_AVAILABLE = False

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

# --- NUOVA COSTANTE PER TABELLA INPUT TEST ---
N_INPUT_DISPLAY_STEPS = 10 # Numero di passi di input da mostrare nella tabella

# --- NUOVE COSTANTI PER LA FRASE DI ATTRIBUZIONE ---
ATTRIBUTION_PHRASE = "Previsione progettata ed elaborata da Alberto Bussaglia"
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
     'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)': {'lat': 43.7176, 'lon': 13.2189, 'name': 'Ponte Garibaldi (Senigallia)', 'type': 'Livello', 'location_id': 'Ponte Garibaldi'}
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
    # La tua classe è già scritta in modo generico, ma assicuriamoci di gestire il caso in cui i pesi non ci sono
    def __init__(self, *tensors):
        # Filtra via eventuali tensori None (come i pesi quando la loss pesata è disattivata)
        self.tensors = tuple(torch.tensor(t, dtype=torch.float32) for t in tensors if t is not None)
        assert all(self.tensors[0].shape[0] == tensor.shape[0] for tensor in self.tensors), "Size mismatch between tensors"

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2, num_quantiles=1):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window # Numero di *passi* da 30 min
        self.output_size = output_size
        self.num_quantiles = num_quantiles
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * output_window * num_quantiles) # Prevede tutti i passi insieme

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Usa output dell'ultimo step
        out = self.fc(out)
        # Reshape per separare quantili e target se necessario
        if self.num_quantiles > 1:
            # Shape: (batch, output_window, output_size * num_quantiles) -> (batch, output_window, output_size, num_quantiles)
            out = out.view(out.size(0), self.output_window, self.output_size, self.num_quantiles)
        else:
            out = out.view(out.size(0), self.output_window, self.output_size) # Reshape standard
        return out

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2, num_quantiles=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        self.num_quantiles = num_quantiles # <<< AGGIUNGI QUESTA LINEA
        self.lstm = nn.LSTM(forecast_input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * num_quantiles)

    def forward(self, x_forecast_step, hidden, cell):
        output, (hidden, cell) = self.lstm(x_forecast_step, (hidden, cell))
        prediction = self.fc(output.squeeze(1)) # Shape: (batch, output_size * num_quantiles)
        
        # <<< INIZIA BLOCCO AGGIUNTO >>>
        if self.num_quantiles > 1:
            # Reshape per separare i quantili: (batch, output_size, num_quantiles)
            prediction = prediction.view(prediction.size(0), self.output_size, self.num_quantiles)
        # <<< FINE BLOCCO AGGIUNTO >>>
            
        return prediction, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.permute(0, 2, 1)
        v_exp = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        scores = torch.bmm(v_exp, energy).squeeze(1)
        return torch.softmax(scores, dim=1)

class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2, num_quantiles=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_input_size = forecast_input_size
        self.output_size = output_size
        self.num_quantiles = num_quantiles
        
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(forecast_input_size + hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * num_quantiles)

    def forward(self, x_forecast_step, hidden, cell, encoder_outputs):
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context_vector = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat([x_forecast_step, context_vector], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        
        # <<< INIZIA BLOCCO AGGIUNTO >>>
        if self.num_quantiles > 1:
            # Reshape per separare i quantili: (batch, output_size, num_quantiles)
            prediction = prediction.view(prediction.size(0), self.output_size, self.num_quantiles)
        # <<< FINE BLOCCO AGGIUNTO >>>
            
        return prediction, hidden, cell, attn_weights

class Seq2SeqHydro(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0):
        batch_size = x_past.shape[0]
        target_output_size = self.decoder.output_size
        num_quantiles = self.decoder.num_quantiles
        
        # <<< MODIFICA QUI >>>
        # Il tensore di output ora deve avere spazio per la dimensione dei quantili
        outputs = torch.zeros(batch_size, self.output_window, target_output_size, num_quantiles).to(x_past.device)
        
        _, encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :]

        for t in range(self.output_window):
            # L'output ora avrà shape [batch, target_size, quantiles] o [batch, target_size]
            decoder_output_step, decoder_hidden, decoder_cell = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell
            )
            # Assicuriamoci che abbia la giusta dimensione per l'assegnazione
            if num_quantiles == 1 and decoder_output_step.ndim == 2:
                decoder_output_step = decoder_output_step.unsqueeze(-1) # Aggiungi la dim del quantile
                
            outputs[:, t, :, :] = decoder_output_step

            if t < self.output_window - 1:
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]
        
        # Se non ci sono quantili, rimuovi l'ultima dimensione per retrocompatibilità
        if num_quantiles == 1:
            outputs = outputs.squeeze(-1)

        return outputs, None

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps

    def forward(self, x_past, x_future_forecast, teacher_forcing_ratio=0.0):
        batch_size = x_past.shape[0]
        target_output_size = self.decoder.output_size
        num_quantiles = self.decoder.num_quantiles

        # <<< MODIFICA QUI >>>
        outputs = torch.zeros(batch_size, self.output_window, target_output_size, num_quantiles).to(x_past.device)
        attention_weights_history = torch.zeros(batch_size, self.output_window, x_past.shape[1]).to(x_past.device)

        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        decoder_input_step = x_future_forecast[:, 0:1, :]

        for t in range(self.output_window):
            decoder_output_step, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell, encoder_outputs
            )
            
            if num_quantiles == 1 and decoder_output_step.ndim == 2:
                decoder_output_step = decoder_output_step.unsqueeze(-1)

            outputs[:, t, :, :] = decoder_output_step
            attention_weights_history[:, t, :] = attn_weights.squeeze()

            if t < self.output_window - 1:
                decoder_input_step = x_future_forecast[:, t+1:t+2, :]
        
        if num_quantiles == 1:
            outputs = outputs.squeeze(-1)
        
        return outputs, attention_weights_history


class DecoderLSTMAutoregressive(nn.Module):
    def __init__(self, forecast_input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.forecast_input_size = forecast_input_size
        
        # NOTA: il modello autoregressivo non supporta i quantili con l'implementazione attuale
        # perché l'input del passo successivo non saprebbe quale quantile usare.
        # Per ora, lasciamo la logica ma con un avviso.
        self.num_quantiles = 1 # Forziamo a 1 per evitare errori
        
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(forecast_input_size + hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_forecast_step, hidden, cell, encoder_outputs):
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context_vector = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat([x_forecast_step, context_vector], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell, attn_weights

class Seq2SeqAutoregressive(nn.Module):
    def __init__(self, encoder, decoder, output_window_steps, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_window = output_window_steps
        self.device = device

    def forward(self, x_past, x_decoder_input_train, y_target_train, teacher_forcing_ratio=0.5):
        batch_size = x_past.shape[0]
        # L'output del decoder (e il target reale) ha solo la dimensione dei target
        target_dim = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, self.output_window, target_dim).to(self.device)
        
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(x_past)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        
        # L'input per il primo step del decoder è il primo elemento della sequenza
        # di input del decoder che abbiamo preparato (che contiene già l'ultimo valore reale)
        decoder_input_step = x_decoder_input_train[:, 0:1, :]

        for t in range(self.output_window):
            prediction_step, decoder_hidden, decoder_cell, _ = self.decoder(
                decoder_input_step, decoder_hidden, decoder_cell, encoder_outputs
            )
            
            outputs[:, t, :] = prediction_step

            # --- LOGICA CHIAVE: TEACHER FORCING VS AUTOREGRESSIONE ---
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Per il prossimo input, usiamo la VERITÀ dal dataset
                # x_decoder_input_train contiene già le feature meteo future E i target reali shiftati
                if t < self.output_window - 1:
                    decoder_input_step = x_decoder_input_train[:, t+1:t+2, :]
            else:
                # Per il prossimo input, usiamo la NOSTRA PREDIZIONE precedente
                if t < self.output_window - 1:
                    # Prendiamo le feature meteo future dal dataset
                    next_forecast_features = x_decoder_input_train[:, t+1:t+2, :-target_dim]
                    
                    # Combiniamo le feature meteo con la nostra ultima predizione
                    # .unsqueeze(1) per aggiungere la dimensione temporale (seq_len=1)
                    decoder_input_step = torch.cat([next_forecast_features, prediction_step.unsqueeze(1)], dim=2)
        
        return outputs, None # Non restituiamo attention weights nel training per semplicità


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class HydroTransformer(nn.Module):
    def __init__(self,
                 input_dim_encoder: int,
                 input_dim_decoder: int,
                 output_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_quantiles: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_quantiles = num_quantiles
        self.output_dim = output_dim
        
        self.encoder_embedding = nn.Linear(input_dim_encoder, d_model)
        self.decoder_embedding = nn.Linear(input_dim_decoder, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, output_dim * num_quantiles)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_padding_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None, teacher_forcing_ratio=0.0) -> torch.Tensor:
        src_embedded_no_pos = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded_no_pos = self.decoder_embedding(tgt) * math.sqrt(self.d_model)

        src_embedded = self.pos_encoder(src_embedded_no_pos.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_embedded = self.pos_encoder(tgt_embedded_no_pos.permute(1, 0, 2)).permute(1, 0, 2)

        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        output = self.transformer(
            src=src_embedded, tgt=tgt_embedded, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
        )

        prediction = self.fc_out(output)
        
        if self.num_quantiles > 1:
            prediction = prediction.view(prediction.shape[0], prediction.shape[1], self.output_dim, self.num_quantiles)

        return prediction, None


# AGGIUNGI QUESTA CLASSE NELLA SEZIONE DELLE DEFINIZIONI DEI MODELLI
class SpatioTemporalGNN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim, num_layers, output_window, output_dim, num_quantiles=1, dropout=0.2):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric non è installato. Impossibile istanziare SpatioTemporalGNN.")
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles

        self.gcn = GCNConv(num_features, hidden_dim)
        self.rnn = GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * num_nodes, output_window * output_dim * num_quantiles) # Modificato per usare tutti i nodi
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: (batch_size, seq_len, num_nodes, num_features)
        batch_size, seq_len, _, _ = x.shape
        
        gcn_outputs_over_time = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            x_t_flat = x_t.reshape(batch_size * self.num_nodes, self.num_features)
            
            edge_index_batch = edge_index.repeat(1, batch_size) + torch.arange(batch_size, device=x.device).repeat_interleave(edge_index.size(1)) * self.num_nodes
            edge_weight_batch = edge_weight.repeat(batch_size) if edge_weight is not None else None

            gcn_out = self.gcn(x_t_flat, edge_index_batch, edge_weight=edge_weight_batch)
            gcn_out = self.relu(gcn_out)
            gcn_out = gcn_out.view(batch_size, self.num_nodes, self.hidden_dim)
            gcn_outputs_over_time.append(gcn_out)
            
        x_gcn_seq = torch.stack(gcn_outputs_over_time, dim=1)
        
        # Reshape per RNN: (batch_size, seq_len, num_nodes * hidden_dim)
        # rnn_input = x_gcn_seq.reshape(batch_size, seq_len, self.num_nodes * self.hidden_dim)
        
        # Per semplicità qui usiamo direttamente l'ultimo output spaziale per la previsione
        last_gcn_out_flat = self.dropout_layer(x_gcn_seq[:, -1, :, :].reshape(batch_size, -1))

        # Il layer FC ora predice l'intera finestra di output in un colpo solo
        out = self.fc(last_gcn_out_flat)

        if self.num_quantiles > 1:
            out = out.view(batch_size, self.output_window, self.output_dim, self.num_quantiles)
        else:
            out = out.view(batch_size, self.output_window, self.output_dim)

        return out, None

# --- Loss Functions ---
class QuantileLoss(nn.Module):
    def __init__(self, quantiles, reduction='mean'):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, -1)
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # y_pred shape: (batch, seq_len, num_targets, num_quantiles)
        # y_true shape: (batch, seq_len, num_targets)
        num_targets = y_true.shape[2]
        y_pred_reshaped = y_pred.view(y_pred.shape[0], y_pred.shape[1], num_targets, -1)
        y_true_reshaped = y_true.unsqueeze(-1)
        quantiles_dev = self.quantiles.to(y_pred.device)
        error = y_true_reshaped - y_pred_reshaped
        loss = torch.max((quantiles_dev - 1) * error, quantiles_dev * error)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# SOSTITUISCI QUESTA CLASSE
class DynamicWeightedMSELoss(nn.Module):
    # NUOVI ARGOMENTI: threshold e scaler_targets per lavorare con i valori reali
    def __init__(self, threshold=0.0, exponent=2.0, reduction='mean', use_unscaled_targets=False):
        super().__init__()
        self.threshold = threshold
        self.exponent = exponent
        self.reduction = reduction
        self.use_unscaled_targets = use_unscaled_targets
        self.mse = nn.MSELoss(reduction='none') # Fondamentale che sia 'none'

    def forward(self, y_pred, y_true, scaler_targets=None):
        # Calcola la loss MSE base per ogni elemento
        per_element_loss = self.mse(y_pred, y_true)
        
        # Calcola i pesi dinamicamente, senza tracciare i gradienti per questa parte
        with torch.no_grad():
            # Decide se usare i target scalati (più semplice) o quelli reali (più preciso)
            base_targets = y_true
            if self.use_unscaled_targets and scaler_targets is not None:
                # Converte temporaneamente i target in numpy, li de-scala e li riporta a tensor
                y_true_unscaled_np = scaler_targets.inverse_transform(y_true.cpu().numpy().reshape(-1, y_true.shape[-1]))
                y_true_unscaled = torch.from_numpy(y_true_unscaled_np.reshape(y_true.shape)).to(y_pred.device)
                base_targets = y_true_unscaled

            # Calcola l'eccesso sopra la soglia, assicurandosi che non sia mai negativo
            excess = torch.clamp(base_targets - self.threshold, min=0)
            
            # Calcola il peso: 1.0 (peso base) + l'eccesso elevato all'esponente
            # Esempio: soglia=2, esponente=1. Valore=3 -> eccesso=1 -> peso=1+1^1=2
            # Esempio: soglia=2, esponente=1. Valore=4 -> eccesso=2 -> peso=1+2^1=3
            weights = 1.0 + torch.pow(excess, self.exponent)

        # Applica i pesi alla loss di ogni elemento
        # Se la loss ha più dimensioni (es. per step temporale), assicuriamoci che i pesi facciano broadcasting corretto
        if per_element_loss.ndim > weights.ndim:
             weights = weights.unsqueeze(-1).expand_as(per_element_loss)
             
        weighted_loss = per_element_loss * weights
        
        # Applica la riduzione finale (mean, sum, or none)
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        return weighted_loss

class DilateLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.1, reduction='mean', device='cpu'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        num_targets = y_true.shape[2]
        total_loss_per_batch_item = torch.zeros(batch_size).to(self.device)
        for i in range(num_targets):
            y_pred_i = y_pred[:, :, i].unsqueeze(2)
            y_true_i = y_true[:, :, i].unsqueeze(2)
            loss_i, _, _ = dilate_loss(y_true_i, y_pred_i, alpha=self.alpha, gamma=self.gamma, device=self.device)
            total_loss_per_batch_item += loss_i
        avg_loss_per_batch_item = total_loss_per_batch_item / num_targets
        if self.reduction == 'mean':
            final_loss = avg_loss_per_batch_item.mean()
        elif self.reduction == 'sum':
            final_loss = avg_loss_per_batch_item.sum()
        else:
            final_loss = avg_loss_per_batch_item
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            return torch.tensor(1e6, device=self.device, requires_grad=True)
        return final_loss

# AGGIUNGI QUESTA NUOVA CLASSE NELLA SEZIONE "Loss Functions"

class WeightedQuantileLoss(nn.Module):
    def __init__(self, quantiles, threshold=0.0, exponent=1.0):
        super().__init__()
        self.quantiles = quantiles
        self.threshold = threshold
        self.exponent = exponent
        self.base_quantile_loss = QuantileLoss(quantiles, reduction='none')

    def forward(self, y_pred, y_true):
        per_element_loss = self.base_quantile_loss(y_pred, y_true)
        with torch.no_grad():
            excess = torch.clamp(y_true - self.threshold, min=0)
            weights = 1.0 + torch.pow(excess, self.exponent)
        weights_expanded = weights.unsqueeze(-1).expand_as(per_element_loss)
        weighted_loss = per_element_loss * weights_expanded
        # IMPORTANTE: Restituiamo la loss per campione, non la media!
        return weighted_loss

# --- Funzioni Utilità Modello/Dati ---
# MODIFICATA: Aggiunti argomenti e logica per i pesi
def prepare_training_data(df, feature_columns, target_columns, input_window, output_window,
                          lag_config=None, cumulative_config=None,
                          use_weighted_loss=False, dummy_col_name='Variabile Dummy', post_modification_weight=1.0): # NUOVI ARGOMENTI
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati LSTM Standard...")
    # ... (tutta la logica di feature engineering rimane invariata) ...
    original_feature_columns = list(feature_columns)
    engineered_feature_names = []

    if lag_config:
        for col, lag_periods_hours in lag_config.items():
            if col in df.columns:
                for lag_hr in lag_periods_hours:
                    lag_steps = lag_hr * 2
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
                    window_steps = win_hr * 2
                    new_col_name = f"{col}_cum{win_hr}h"
                    df[new_col_name] = df[col].rolling(window=window_steps, min_periods=1).sum()
                    engineered_feature_names.append(new_col_name)
                    print(f"Created cumulative feature: {new_col_name}")
            else:
                st.warning(f"Colonna '{col}' specificata in cumulative_config non trovata nel DataFrame.")

    current_feature_columns = original_feature_columns + engineered_feature_names
    
    # NUOVA LOGICA: Assicurati che la colonna dummy sia presente se necessario
    if use_weighted_loss:
        if dummy_col_name not in df.columns:
            st.error(f"Loss pesata attivata, ma la colonna dummy '{dummy_col_name}' non è stata trovata nel DataFrame.")
            return None, None, None, None, None
        # Aggiungiamo temporaneamente la colonna dummy per la creazione delle sequenze, se non è già una feature
        cols_for_sequencing = list(set(current_feature_columns + target_columns + [dummy_col_name]))
    else:
        cols_for_sequencing = list(set(current_feature_columns + target_columns))
        
    if engineered_feature_names:
        cols_present_in_df_to_fill = [c for c in cols_for_sequencing if c in df.columns]
        
        nan_count_before_bfill = df[cols_present_in_df_to_fill].isnull().sum().sum()
        if nan_count_before_bfill > 0:
            df[cols_present_in_df_to_fill] = df[cols_present_in_df_to_fill].fillna(method='bfill')
            nan_count_after_bfill = df[cols_present_in_df_to_fill].isnull().sum().sum()
            st.warning(f"NaNs from feature engineering: {nan_count_before_bfill} before bfill, {nan_count_after_bfill} after bfill. ")
    
    try:
        missing_features = [col for col in current_feature_columns if col not in df.columns]
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_features:
            st.error(f"Errore: Feature columns mancanti: {missing_features}")
            return None, None, None, None, None
        if missing_targets:
            st.error(f"Errore: Target columns mancanti: {missing_targets}")
            return None, None, None, None, None
        
        if df[cols_for_sequencing].isnull().any().any():
             st.warning("NaN trovati nelle colonne rilevanti PRIMA della creazione sequenze LSTM.")
    except Exception as e:
        st.error(f"Errore controllo colonne in prepare_training_data: {e}")
        return None, None, None, None, None

    X, y = [], []
    sample_weights = [] # NUOVO: Lista per i pesi
    total_len = len(df)
    input_steps = input_window * 2
    output_steps = output_window * 2
    required_len = input_steps + output_steps

    if total_len < required_len:
         st.error(f"Dati LSTM insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
         return None, None, None, None, None

    for i in range(total_len - required_len + 1):
        feature_window_data = df.iloc[i : i + input_steps][current_feature_columns]
        target_window_data = df.iloc[i + input_steps : i + required_len][target_columns]

        if feature_window_data.isnull().any().any() or target_window_data.isnull().any().any():
            continue
            
        X.append(feature_window_data.values)
        y.append(target_window_data.values)
        
        # --- INIZIO NUOVA LOGICA PER I PESI ---
        if use_weighted_loss:
            # Usiamo il valore della dummy all'inizio del periodo di target
            dummy_value = df.iloc[i + input_steps][dummy_col_name]
            if dummy_value == 1:
                sample_weights.append(post_modification_weight)
            else:
                sample_weights.append(1.0)
        # --- FINE NUOVA LOGICA ---

    if not X or not y:
        st.error("Errore creazione sequenze X/y LSTM."); return None, None, None, None, None
        
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)
    sample_weights = np.array(sample_weights, dtype=np.float32) if use_weighted_loss else None # NUOVO: Converti in array

    if X.size == 0 or y.size == 0:
        st.error("Dati X o y vuoti prima di scaling LSTM."); return None, None, None, None, None

    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    num_sequences, seq_len_in, num_features_X = X.shape
    num_sequences_y, seq_len_out, num_targets_y = y.shape
    
    if num_features_X != len(current_feature_columns):
        st.error(f"Errore numero feature in X ({num_features_X}) vs colonne attese ({len(current_feature_columns)}).")
        return None, None, None, None, None

    if seq_len_in != input_steps or seq_len_out != output_steps:
        st.error(f"Errore shape sequenze LSTM: In={seq_len_in} (atteso {input_steps}), Out={seq_len_out} (atteso {output_steps})")
        return None, None, None, None, None

    X_flat = X.reshape(-1, num_features_X); y_flat = y.reshape(-1, num_targets_y)
    try:
        X_scaled_flat = scaler_features.fit_transform(X_flat)
        y_scaled_flat = scaler_targets.fit_transform(y_flat)
    except ValueError as ve_scale:
        if "Input contains NaN" in str(ve_scale):
             st.error(f"Errore scaling LSTM: Input contiene NaN. Dettagli: {ve_scale}")
        else:
             st.error(f"Errore scaling LSTM: {ve_scale}")
        return None, None, None, None, None
    except Exception as e_scale:
        st.error(f"Errore scaling LSTM generico: {e_scale}"); return None, None, None, None, None

    X_scaled = X_scaled_flat.reshape(num_sequences, seq_len_in, num_features_X)
    y_scaled = y_scaled_flat.reshape(num_sequences_y, seq_len_out, num_targets_y)

    print(f"Dati LSTM pronti: X_scaled shape={X_scaled.shape}, y_scaled shape={y_scaled.shape}.")
    # MODIFICATO: Restituisce anche i pesi
    return X_scaled, y_scaled, sample_weights, scaler_features, scaler_targets


def prepare_training_data_seq2seq(df, past_feature_cols, forecast_feature_cols, target_cols,
                                 input_window_steps, forecast_window_steps, output_window_steps,
                                 lag_config_past=None, cumulative_config_past=None,
                                 use_weighted_loss=False, dummy_col_name='Variabile Dummy', post_modification_weight=1.0):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq/Transformer (Ottimizzata)...")
    print(f"Input Steps: {input_window_steps}, Forecast Steps: {forecast_window_steps}, Output Steps: {output_window_steps}")
    
    # --- 1. Feature Engineering (invariato) ---
    df_processed = df.copy()
    engineered_past_feature_names = []
    if lag_config_past:
        for col, lag_periods_hours in lag_config_past.items():
            if col in df_processed.columns:
                for lag_hr in lag_periods_hours:
                    lag_steps = lag_hr * 2
                    new_col_name = f"{col}_lag{lag_hr}h"
                    df_processed[new_col_name] = df_processed[col].shift(lag_steps)
                    engineered_past_feature_names.append(new_col_name)
    
    if cumulative_config_past:
        for col, window_periods_hours in cumulative_config_past.items():
            if col in df_processed.columns:
                for win_hr in window_periods_hours:
                    window_steps = win_hr * 2
                    new_col_name = f"{col}_cum{win_hr}h"
                    df_processed[new_col_name] = df_processed[col].rolling(window=window_steps, min_periods=1).sum()
                    engineered_past_feature_names.append(new_col_name)
                    
    current_past_feature_cols = past_feature_cols + engineered_past_feature_names
    
    # --- 2. Estrazione Dati in NumPy (OTTIMIZZAZIONE CHIAVE) ---
    all_needed_cols = list(set(current_past_feature_cols + forecast_feature_cols + target_cols))
    if use_weighted_loss:
        if dummy_col_name not in df_processed.columns:
            st.error(f"Loss pesata attivata, ma la colonna dummy '{dummy_col_name}' non è stata trovata.")
            return None, None, None, None, None, None, None
        all_needed_cols.append(dummy_col_name)
    
    if engineered_past_feature_names:
        df_processed[engineered_past_feature_names] = df_processed[engineered_past_feature_names].fillna(method='bfill').fillna(method='ffill')

    past_features_np = df_processed[current_past_feature_cols].values.astype(np.float32)
    forecast_features_np = df_processed[forecast_feature_cols].values.astype(np.float32)
    target_np = df_processed[target_cols].values.astype(np.float32)
    dummy_np = df_processed[dummy_col_name].values.astype(np.float32) if use_weighted_loss else None

    # --- 3. Creazione delle Sequenze con Viste (OTTIMIZZAZIONE CHIAVE) ---
    total_len = len(df_processed)
    required_len = input_window_steps + output_window_steps
    if total_len < required_len:
        st.error(f"Dati insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
        return None, None, None, None, None, None, None

    X_encoder_view = sliding_window_view(past_features_np, (input_window_steps, past_features_np.shape[1]))
    X_decoder_view = sliding_window_view(forecast_features_np, (forecast_window_steps, forecast_features_np.shape[1]))
    y_target_view = sliding_window_view(target_np, (output_window_steps, target_np.shape[1]))

    num_samples = total_len - required_len + 1
    # Le viste hanno shape: (num_windows, 1, window_steps, num_features). Rimuoviamo SOLO la dim singleton centrale.
    X_encoder = X_encoder_view[:num_samples, 0, :, :]
    X_decoder = X_decoder_view[input_window_steps:input_window_steps + num_samples, 0, :, :]
    y_target = y_target_view[input_window_steps:input_window_steps + num_samples, 0, :, :]

    sample_weights = None
    if use_weighted_loss:
        dummy_values_at_target_start = dummy_np[input_window_steps : input_window_steps + num_samples]
        sample_weights = np.where(dummy_values_at_target_start == 1, post_modification_weight, 1.0).astype(np.float32)

    # --- 4. Filtro NaN Vettorizzato (OTTIMIZZAZIONE CHIAVE) ---
    nan_mask_enc = np.isnan(X_encoder).any(axis=(1, 2))
    nan_mask_dec = np.isnan(X_decoder).any(axis=(1, 2))
    nan_mask_tar = np.isnan(y_target).any(axis=(1, 2))
    valid_mask = ~ (nan_mask_enc | nan_mask_dec | nan_mask_tar)
    
    X_encoder = X_encoder[valid_mask]
    X_decoder = X_decoder[valid_mask]
    y_target = y_target[valid_mask]
    if use_weighted_loss:
        sample_weights = sample_weights[valid_mask]
        
    if X_encoder.shape[0] == 0:
        st.error("Nessun campione valido trovato dopo la rimozione dei NaN.")
        return None, None, None, None, None, None, None

    # --- 5. Scaling ---
    scaler_past_features = MinMaxScaler()
    scaler_forecast_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    X_enc_scaled = scaler_past_features.fit_transform(X_encoder.reshape(-1, X_encoder.shape[-1])).reshape(X_encoder.shape)
    X_dec_scaled = scaler_forecast_features.fit_transform(X_decoder.reshape(-1, X_decoder.shape[-1])).reshape(X_decoder.shape)
    y_tar_scaled = scaler_targets.fit_transform(y_target.reshape(-1, y_target.shape[-1])).reshape(y_target.shape)

    st.success(f"Preparazione dati ottimizzata completata. Creati {X_encoder.shape[0]} campioni validi.")
    print(f"Shapes Finali: Encoder={X_enc_scaled.shape}, Decoder={X_dec_scaled.shape}, Target={y_tar_scaled.shape}")
    
    return (X_enc_scaled, X_dec_scaled, y_tar_scaled, sample_weights,
            scaler_past_features, scaler_forecast_features, scaler_targets)


def prepare_training_data_seq2seq_autoregressive(df, past_feature_cols, forecast_feature_cols, target_cols,
                                                  input_window_steps, output_window_steps):
    """
    Prepara i dati per un modello Seq2Seq AUTOREGRESSIVO.
    L'input del decoder (X_decoder) conterrà sia le feature future (meteo)
    sia il valore del target al passo precedente.
    """
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati Seq2Seq Autoregressivo...")

    # Combiniamo tutte le colonne necessarie per creare le sequenze
    all_needed_cols = list(set(past_feature_cols + forecast_feature_cols + target_cols))
    df_processed = df[all_needed_cols].copy()
    
    # Gestione NaN (semplificata, puoi usare la tua logica più complessa)
    df_processed.fillna(method='ffill', inplace=True)
    df_processed.fillna(method='bfill', inplace=True)

    X_encoder_list, X_decoder_list, y_target_list = [], [], []
    
    total_len = len(df_processed)
    required_len = input_window_steps + output_window_steps

    if total_len < required_len:
        st.error(f"Dati insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
        return None, None, None, None, None, None

    for i in range(total_len - required_len + 1):
        # 1. Input Encoder: dati storici (invariato)
        past_data = df_processed.iloc[i : i + input_window_steps][past_feature_cols].values
        
        # 2. Dati reali del target nella finestra di output
        target_sequence = df_processed.iloc[i + input_window_steps : i + required_len][target_cols].values
        
        # 3. Feature future (es. meteo) nella finestra di output
        forecast_features = df_processed.iloc[i + input_window_steps : i + required_len][forecast_feature_cols].values

        # --- PARTE CHIAVE: CREAZIONE INPUT AUTOREGRESSIVO ---
        # Creiamo i target "shiftati"
        # Il primo input per il decoder sarà l'ultimo valore REALE della sequenza di input
        last_known_target = df_processed.iloc[i + input_window_steps - 1][target_cols].values.reshape(1, -1)
        
        # Concateniamo l'ultimo valore noto con i target reali della finestra (escluso l'ultimo)
        # Questo è l'input perfetto per il Teacher Forcing
        shifted_targets = np.vstack([last_known_target, target_sequence[:-1, :]])
        
        # Combiniamo le feature future (meteo) con i target shiftati
        decoder_input = np.concatenate([forecast_features, shifted_targets], axis=1)
        # ----------------------------------------------------

        # Aggiungiamo solo se non ci sono NaN
        if not np.isnan(past_data).any() and not np.isnan(decoder_input).any() and not np.isnan(target_sequence).any():
            X_encoder_list.append(past_data)
            X_decoder_list.append(decoder_input)
            y_target_list.append(target_sequence)

    if not X_encoder_list:
        st.error("Nessun campione valido creato. Controlla i NaN nei tuoi dati.")
        return None, None, None, None, None, None

    # Conversione in array NumPy
    X_encoder_np = np.array(X_encoder_list, dtype=np.float32)
    X_decoder_np = np.array(X_decoder_list, dtype=np.float32)
    y_target_np = np.array(y_target_list, dtype=np.float32)

    # Scaling (fondamentale)
    scaler_past_features = MinMaxScaler()
    scaler_forecast_decoder_input = MinMaxScaler() # Un solo scaler per l'input combinato del decoder
    scaler_targets = MinMaxScaler()

    X_enc_scaled = scaler_past_features.fit_transform(X_encoder_np.reshape(-1, X_encoder_np.shape[-1])).reshape(X_encoder_np.shape)
    X_dec_scaled = scaler_forecast_decoder_input.fit_transform(X_decoder_np.reshape(-1, X_decoder_np.shape[-1])).reshape(X_decoder_np.shape)
    y_tar_scaled = scaler_targets.fit_transform(y_target_np.reshape(-1, y_target_np.shape[-1])).reshape(y_target_np.shape)

    st.success(f"Preparazione dati autoregressiva completata. Creati {X_enc_scaled.shape[0]} campioni.")
    
    return (X_enc_scaled, X_dec_scaled, y_tar_scaled,
            scaler_past_features, scaler_forecast_decoder_input, scaler_targets)


# SOSTITUISCI la funzione prepare_training_data_gnn con questa versione CORRETTA E OTTIMIZZATA

def prepare_training_data_gnn(df, node_columns, target_columns, feature_mapping,
                              input_window_steps, output_window_steps, progress_bar=None,
                              use_weighted_loss=False, dummy_col_name='Variabile Dummy', post_modification_weight=1.0):
    if progress_bar:
        progress_bar.progress(0.0, text="Avvio preparazione dati GNN ottimizzata...")
    
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Preparazione dati GNN...")
    
    num_nodes = len(node_columns)
    num_features_per_node = [1 + len(feature_mapping.get(node, [])) for node in node_columns]
    if len(set(num_features_per_node)) > 1:
        st.error("Errore di configurazione: Tutti i nodi devono avere lo stesso numero totale di feature.")
        return None, None, None, None, None, -1
    num_features = num_features_per_node[0] if num_features_per_node else 1
    
    if progress_bar:
        progress_bar.progress(0.1, text="Fase 1/5: Estrazione dati da DataFrame...")
    
    all_feature_columns = []
    for node_name in node_columns:
        all_feature_columns.append(node_name)
        all_feature_columns.extend(feature_mapping.get(node_name, []))
    
    if use_weighted_loss:
        if dummy_col_name not in df.columns:
            st.error(f"Loss pesata attivata, ma la colonna dummy '{dummy_col_name}' non è stata trovata.")
            return None, None, None, None, None, -1
        if dummy_col_name not in all_feature_columns:
             # Aggiungiamo la colonna dummy al df per l'estrazione, ma non alle feature del modello
             all_needed_cols = list(set(all_feature_columns + target_columns + [dummy_col_name]))
        else:
             all_needed_cols = list(set(all_feature_columns + target_columns))
    else:
        all_needed_cols = list(set(all_feature_columns + target_columns))

    # Estrazione di tutte le colonne necessarie in una volta
    data_np = df[all_needed_cols].values.astype(np.float32)
    df_temp = pd.DataFrame(data_np, columns=all_needed_cols) # DF temporaneo per slicing facile

    feature_data_np = df_temp[all_feature_columns].values
    full_feature_matrix = feature_data_np.reshape(len(df), num_nodes, num_features, order='C')
    
    if progress_bar:
        progress_bar.progress(0.3, text="Fase 2/5: Creazione sequenze di input (X)...")
    
    total_len = len(df)
    required_len = input_window_steps + output_window_steps
    if total_len < required_len:
        st.error(f"Dati insufficienti ({total_len} righe) per creare sequenze (richieste {required_len} righe).")
        return None, None, None, None, None, -1
    
    num_samples = total_len - required_len + 1

    shape = (num_samples, input_window_steps, num_nodes, num_features)
    strides = (full_feature_matrix.strides[0],) + full_feature_matrix.strides
    X = np.lib.stride_tricks.as_strided(full_feature_matrix, shape=shape, strides=strides)

    if progress_bar:
        progress_bar.progress(0.5, text="Fase 3/5: Creazione sequenze di target (y) e pesi...")
        
    target_data_np = df_temp[target_columns].values
    shape_y = (num_samples, output_window_steps, len(target_columns))
    strides_y = (target_data_np.strides[0],) + target_data_np.strides
    y = np.lib.stride_tricks.as_strided(target_data_np[input_window_steps:], shape=shape_y, strides=strides_y)
    
    sample_weights = None
    if use_weighted_loss:
        dummy_data_np = df_temp[dummy_col_name].values
        dummy_values_at_target_start = dummy_data_np[input_window_steps : input_window_steps + num_samples]
        sample_weights = np.where(dummy_values_at_target_start == 1, post_modification_weight, 1.0).astype(np.float32)

    if progress_bar:
        progress_bar.progress(0.7, text="Fase 4/5: Filtraggio campioni non validi (NaN)...")
        
    X, y = np.copy(X), np.copy(y)
    nan_mask_X = np.isnan(X).any(axis=(1, 2, 3))
    nan_mask_y = np.isnan(y).any(axis=(1, 2))
    valid_mask = ~ (nan_mask_X | nan_mask_y)
    
    X = X[valid_mask]; y = y[valid_mask]
    if use_weighted_loss: sample_weights = sample_weights[valid_mask]

    if X.shape[0] == 0:
        st.error("Nessun campione valido trovato dopo la rimozione dei NaN.")
        if progress_bar: progress_bar.empty()
        return None, None, None, None, None, -1

    if progress_bar:
        progress_bar.progress(0.9, text="Fase 5/5: Scaling dei dati...")
        
    scaler_features = MinMaxScaler(); scaler_targets = MinMaxScaler()
    X_scaled = scaler_features.fit_transform(X.reshape(-1, num_features)).reshape(X.shape)
    y_scaled = scaler_targets.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    
    if progress_bar:
        progress_bar.progress(1.0, text="Preparazione dati completata!")
        pytime.sleep(1)
    
    st.success(f"Preparazione dati GNN completata. Creati {X.shape[0]} campioni validi.")
    print(f"Dati GNN pronti: X_scaled={X_scaled.shape}, y_scaled={y_scaled.shape}, Weights={sample_weights.shape if sample_weights is not None else 'None'}")
    return X_scaled, y_scaled, sample_weights, scaler_features, scaler_targets, num_features

def train_model_gnn(X_scaled_full, y_scaled_full, sample_weights_full, model, edge_index, edge_weights,
                    epochs=50, batch_size=32, learning_rate=0.001,
                    save_strategy='migliore', preferred_device='auto',
                    n_splits_cv=3, loss_function_name="MSELoss", training_mode='standard', quantiles=None,
                    split_method="Temporale", validation_size=0.2, use_weighted_loss=False, 
                    use_magnitude_loss=False, target_threshold=0.5, weight_exponent=1.0):
    device = torch.device('cuda' if ('auto' in preferred_device.lower() and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float32).to(device) if edge_weights is not None else None

    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training GNN (Weighted: {use_weighted_loss}, loss={loss_function_name})...")
    
    reduction_strategy = 'none' if use_weighted_loss or (loss_function_name == "DynamicWeightedMSE" and use_magnitude_loss) or (training_mode == 'quantile' and use_magnitude_loss) else 'mean'
    criterion = None
    if training_mode == 'quantile':
        if use_magnitude_loss:
            st.info(f"Loss Quantile Pesata per Magnitudo attivata con Soglia: {target_threshold}, Esponente: {weight_exponent}")
            criterion = WeightedQuantileLoss(
                quantiles=quantiles, 
                threshold=target_threshold, 
                exponent=weight_exponent
            )
        else:
            criterion = QuantileLoss(quantiles=quantiles, reduction=reduction_strategy)
    elif loss_function_name == "HuberLoss":
        criterion = nn.HuberLoss(reduction=reduction_strategy)
    elif loss_function_name == "DynamicWeightedMSE":
        st.info(f"Loss Dinamica attivata con Soglia (scalata): {target_threshold}, Esponente: {weight_exponent}")
        criterion = DynamicWeightedMSELoss(
            threshold=target_threshold, 
            exponent=weight_exponent, 
            reduction=reduction_strategy,
            use_unscaled_targets=False 
        )
    elif loss_function_name == "DilateLoss":
        criterion = DilateLoss(alpha=0.5, gamma=0.1, device=device, reduction='none')
    else:
        criterion = nn.MSELoss(reduction=reduction_strategy)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    weights_train, weights_val = None, None
    if "Temporale" in split_method:
        tscv = TimeSeriesSplit(n_splits=n_splits_cv)
        train_indices, val_indices = list(tscv.split(X_scaled_full))[-1]
        
        X_train, y_train = X_scaled_full[train_indices], y_scaled_full[train_indices]
        X_val, y_val = X_scaled_full[val_indices], y_scaled_full[val_indices]
        if use_weighted_loss and sample_weights_full is not None:
            weights_train = sample_weights_full[train_indices]
            weights_val = sample_weights_full[val_indices]
    else: # Logica Casuale
        st.info(f"Suddivisione casuale con {validation_size*100:.0f}% dei dati per la validazione.")
        arrays_to_split = (X_scaled_full, y_scaled_full)
        if use_weighted_loss and sample_weights_full is not None:
            arrays_to_split += (sample_weights_full,)

        split_results = train_test_split(*arrays_to_split, test_size=validation_size, shuffle=True, random_state=42)
        
        X_train, X_val, y_train, y_val = split_results[0], split_results[1], split_results[2], split_results[3]
        if use_weighted_loss and sample_weights_full is not None:
            weights_train, weights_val = split_results[4], split_results[5]

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, weights_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size) if len(X_val) > 0 else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    progress_bar = st.progress(0.0, text="Training GNN: Inizio...")
    status_text = st.empty()
    loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        if v_loss and any(v is not None for v in v_loss):
            fig.add_trace(go.Scatter(y=v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training GNN per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            if use_weighted_loss:
                X_batch, y_batch, weights_batch = batch
                weights_batch = weights_batch.to(device)
            else:
                X_batch, y_batch = batch
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(X_batch, edge_index_tensor, edge_weight=edge_weights_tensor)
            # --- INIZIO DEL NUOVO BLOCCO DI CALCOLO LOSS (DA SOSTITUIRE) ---

            # `criterion` calcola la loss per campione (pesata per magnitudo se attivata).
            loss_per_sample = criterion(outputs, y_batch)

            # Se la pesatura temporale (dummy) è attiva, la applichiamo.
            if use_weighted_loss and 'weights_batch' in locals() and weights_batch is not None:
                # Assicuriamoci che le dimensioni siano compatibili prima della moltiplicazione.
                # Se la loss ha più dimensioni (es. per step temporale/quantili), facciamo la media su quelle.
                if loss_per_sample.ndim > 1:
                    loss_per_sample = loss_per_sample.mean(dim=tuple(range(1, loss_per_sample.ndim)))
                
                # Moltiplica la loss (già pesata per magnitudo) per il peso temporale.
                final_loss_per_sample = loss_per_sample * weights_batch
                loss = final_loss_per_sample.mean() # Calcola la media finale per ottenere uno scalare.
            else:
                # Se NON c'è pesatura temporale, dobbiamo comunque ridurre la loss per campione a uno scalare.
                # Semplicemente calcoliamo la media.
                loss = loss_per_sample.mean()

            # --- FINE DEL NUOVO BLOCCO DI CALCOLO LOSS ---

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        
        train_losses.append(epoch_train_loss / len(train_loader.dataset))
        
        epoch_val_loss = None
        if val_loader:
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs, _ = model(X_batch, edge_index_tensor, edge_weight=edge_weights_tensor)
                    # --- INIZIO BLOCCO DI VALIDAZIONE CORRETTO (DA SOSTITUIRE) ---

                    # Calcola la loss per campione (sarà un tensore se reduction='none')
                    loss_per_sample_val = criterion(outputs, y_batch) # Usa y_tar per seq2seq

                    # Indipendentemente da tutto, dobbiamo ridurre la loss di validazione a uno scalare
                    # facendo la media. La validazione non usa MAI pesi.
                    val_loss_unweighted = loss_per_sample_val.mean()

                    # Ora val_loss_unweighted è garantito essere uno scalare.
                    epoch_val_loss_sum += val_loss_unweighted.item() * X_batch.size(0) # Usa x_enc.size(0) per seq2seq

                    # --- FINE BLOCCO DI VALIDAZIONE CORRETTO ---
            epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            scheduler.step(epoch_val_loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_losses.append(None)

        val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "N/A"
        status_text.markdown(f"Epoca {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_loss_str}")

        progress_bar.progress((epoch + 1) / epochs)
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

    if save_strategy == 'migliore' and best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        st.success(f"Caricato modello con Val Loss migliore: {best_val_loss:.6f}")

    return model, (train_losses, []), (val_losses, [])

def predict_gnn(model, input_data_multi_feature, scalers, config, device, uncertainty_passes=1):
    model.eval()
    if uncertainty_passes > 1 and config.get("training_mode") != "quantile":
        model.train()

    scaler_features, scaler_targets = scalers
    edge_index = torch.tensor(config['edge_index'], dtype=torch.long).to(device)
    edge_weights = torch.tensor(config['edge_weights'], dtype=torch.float32).to(device) if config.get('edge_weights') else None
    
    # MODIFICATO: Leggi il numero di feature dalla config
    num_features = config.get("num_features", 1)
    
    predictions_list = []
    
    with torch.no_grad():
        for _ in range(uncertainty_passes):
            # L'input_data ora ha shape (seq_len, num_nodes, num_features)
            # Lo scaler si aspetta (n_samples, n_features)
            inp_norm = scaler_features.transform(input_data_multi_feature.reshape(-1, num_features))
            inp_reshaped = inp_norm.reshape(input_data_multi_feature.shape)
            inp_tens = torch.FloatTensor(inp_reshaped).unsqueeze(0).to(device)
            
            output, _ = model(inp_tens, edge_index, edge_weight=edge_weights)
            out_np = output.cpu().numpy().squeeze(0)
            
            # Gestione Quantile vs Standard
            if config.get("training_mode") == "quantile":
                # out_np shape: (output_window, num_targets, num_quantiles)
                num_targets = len(config['target_columns'])
                # We need to reshape for inverse_transform: each target must be a column
                preds_scaled_flat = out_np.reshape(-1, num_targets)
                preds_flat = scaler_targets.inverse_transform(preds_scaled_flat)
                preds = preds_flat.reshape(out_np.shape)
            else:
                preds = scaler_targets.inverse_transform(out_np)
            predictions_list.append(preds)
            
    predictions_array = np.array(predictions_list)
    mean_pred = np.mean(predictions_array, axis=0)
    std_pred = np.std(predictions_array, axis=0) if uncertainty_passes > 1 and config.get("training_mode") != "quantile" else None

    # Ensure model is back in eval mode if it was changed
    model.eval()
    return mean_pred, std_pred

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

            if model_type == "Seq2SeqAutoregressivo":
                required_keys = ["input_window_steps", "output_window_steps", "hidden_size", "num_layers", "dropout",
                                 "all_past_feature_columns", "forecast_input_columns", "target_columns"]
                s_past_p = os.path.join(models_dir, f"{base}_past_features.joblib")
                s_dec_in_p = os.path.join(models_dir, f"{base}_forecast_decoder_input.joblib")
                s_targ_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(s_past_p) and os.path.exists(s_dec_in_p) and os.path.exists(s_targ_p)):
                    model_info.update({
                        "scaler_past_features_path": s_past_p,
                        "scaler_forecast_decoder_input_path": s_dec_in_p,
                        "scaler_targets_path": s_targ_p,
                        "model_type": model_type
                    })
                    valid_model = True
            elif model_type == "Seq2Seq" or model_type == "Seq2SeqAttention":
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
                        "model_type": model_type
                    })
                    valid_model = True
            elif model_type == "Transformer":
                required_keys = [
                    "input_window_steps", "output_window_steps", "forecast_window_steps",
                    "d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward", "dropout",
                    "all_past_feature_columns", "forecast_input_columns", "target_columns"
                ]
                s_past_p = os.path.join(models_dir, f"{base}_past_features.joblib")
                s_fore_p = os.path.join(models_dir, f"{base}_forecast_features.joblib")
                s_targ_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if (all(k in config_data for k in required_keys) and
                    os.path.exists(s_past_p) and os.path.exists(s_fore_p) and os.path.exists(s_targ_p)):
                    model_info.update({
                        "scaler_past_features_path": s_past_p,
                        "scaler_forecast_features_path": s_fore_p,
                        "scaler_targets_path": s_targ_p,
                        "model_type": "Transformer"
                    })
                    valid_model = True
            elif model_type == "SpatioTemporalGNN":
                required_keys = ["input_window_steps", "output_window_steps", "hidden_dim", "node_order", "edge_index", "target_columns"]
                scf_p = os.path.join(models_dir, f"{base}_features.joblib")
                sct_p = os.path.join(models_dir, f"{base}_targets.joblib")
                if all(k in config_data for k in required_keys) and os.path.exists(scf_p) and os.path.exists(sct_p):
                     model_info.update({"scaler_features_path": scf_p, "scaler_targets_path": sct_p, "model_type": "SpatioTemporalGNN"})
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
            st.warning(f"Modello '{base}' ignorato: errore lettura/config JSON: {e_cfg}")
            valid_model = False

        if valid_model:
            available[name] = model_info
        else:
            print(f"Modello '{base}' ignorato: file mancanti o config incompleta per tipo '{model_type}'.")
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
    training_mode = config.get("training_mode", "standard")
    num_quantiles = 1
    if training_mode == 'quantile':
        quantiles = config.get("quantiles", [])
        if not quantiles:
            st.error("Training mode è 'quantile' ma la chiave 'quantiles' è mancante o vuota nella config.")
            return None, None
        num_quantiles = len(quantiles)

    try:
        model = None
        if model_type == "Seq2SeqAutoregressivo":
            enc_input_size = len(config["all_past_feature_columns"])
            # The decoder input is now forecast features + target features
            dec_input_size = len(config["forecast_input_columns"]) + len(config["target_columns"])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]; layers = config["num_layers"]; drop = config["dropout"]
            out_win = config["output_window_steps"]
            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            # Use the new autoregressive decoder
            decoder = DecoderLSTMAutoregressive(dec_input_size, hidden, dec_output_size, layers, drop).to(device)
            # Use the new autoregressive Seq2Seq model
            model = Seq2SeqAutoregressive(encoder, decoder, out_win, device).to(device)
        elif model_type == "Seq2Seq":
            enc_input_size = len(config["all_past_feature_columns"])
            dec_input_size = len(config["forecast_input_columns"])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]; layers = config["num_layers"]; drop = config["dropout"]
            out_win = config["output_window_steps"]
            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            decoder = DecoderLSTM(dec_input_size, hidden, dec_output_size, layers, drop, num_quantiles=num_quantiles).to(device)
            model = Seq2SeqHydro(encoder, decoder, out_win).to(device)
        elif model_type == "Seq2SeqAttention":
            enc_input_size = len(config["all_past_feature_columns"])
            dec_input_size = len(config["forecast_input_columns"])
            dec_output_size = len(config["target_columns"])
            hidden = config["hidden_size"]; layers = config["num_layers"]; drop = config["dropout"]
            out_win = config["output_window_steps"]
            encoder = EncoderLSTM(enc_input_size, hidden, layers, drop).to(device)
            decoder = DecoderLSTMWithAttention(dec_input_size, hidden, dec_output_size, layers, drop, num_quantiles=num_quantiles).to(device)
            model = Seq2SeqWithAttention(encoder, decoder, out_win).to(device)
        elif model_type == "Transformer":
            enc_input_size = len(config["all_past_feature_columns"])
            dec_input_size = len(config["forecast_input_columns"])
            dec_output_size = len(config["target_columns"])
            
            model = HydroTransformer(
                input_dim_encoder=enc_input_size,
                input_dim_decoder=dec_input_size,
                output_dim=dec_output_size,
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_encoder_layers=config["num_encoder_layers"],
                num_decoder_layers=config["num_decoder_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
                num_quantiles=num_quantiles
            ).to(device)
        elif model_type == "SpatioTemporalGNN":
            if not PYG_AVAILABLE: raise RuntimeError("PyTorch Geometric non trovato.")
            
            # MODIFICATO: Leggi il numero di feature dalla config
            num_features_from_config = config.get("num_features", 1) # Default a 1 per retrocompatibilità
            
            model = SpatioTemporalGNN(
                num_nodes=len(config["node_order"]),
                num_features=num_features_from_config, # <-- Usa il valore dalla config
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                output_window=config["output_window_steps"],
                output_dim=len(config["target_columns"]),
                num_quantiles=num_quantiles,
                dropout=config.get("dropout", 0.2)
            ).to(device)
        else: # LSTM Standard
            f_cols_lstm = config.get("feature_columns")
            if not f_cols_lstm:
                 raise ValueError(f"Chiave 'feature_columns' mancante nella config per il modello LSTM '{config.get('display_name', 'N/A')}'.")

            input_size_lstm = len(f_cols_lstm)
            target_size_lstm = len(config["target_columns"])
            out_win_lstm = config["output_window"]
            model = HydroLSTM(input_size_lstm, config["hidden_size"], target_size_lstm,
                              out_win_lstm, config["num_layers"], config["dropout"], num_quantiles=num_quantiles).to(device)

        if isinstance(_model_path, str):
             if not os.path.exists(_model_path): raise FileNotFoundError(f"File modello '{_model_path}' non trovato.")
             model.load_state_dict(torch.load(_model_path, map_location=device))
        elif hasattr(_model_path, 'getvalue'):
             _model_path.seek(0)
             model.load_state_dict(torch.load(_model_path, map_location=device))
        else: raise TypeError("Percorso modello non valido.")
        model.eval()
        print(f"Modello '{config.get('display_name', 'N/A')}' (Tipo: {model_type}, Mode: {training_mode}) caricato su {device}.")
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
         elif hasattr(path, 'getvalue'): path.seek(0); return joblib.load(path)
         else: raise TypeError(f"Percorso scaler non valido: {type(path)}")

    try:
        if model_type == "Seq2SeqAutoregressivo":
            scaler_past = _load_joblib(model_info["scaler_past_features_path"])
            scaler_forecast_decoder_input = _load_joblib(model_info["scaler_forecast_decoder_input_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            print(f"Scaler {model_type} caricati.")
            return {"past": scaler_past, "forecast_decoder_input": scaler_forecast_decoder_input, "targets": scaler_targets}
        elif model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
            scaler_past = _load_joblib(model_info["scaler_past_features_path"])
            scaler_forecast = _load_joblib(model_info["scaler_forecast_features_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            print(f"Scaler {model_type} caricati.")
            return {"past": scaler_past, "forecast": scaler_forecast, "targets": scaler_targets}
        else: # LSTM Standard
            scaler_features = _load_joblib(model_info["scaler_features_path"])
            scaler_targets = _load_joblib(model_info["scaler_targets_path"])
            print(f"Scaler LSTM caricati.")
            return scaler_features, scaler_targets
    except Exception as e:
        st.error(f"Errore caricamento scaler (Tipo: {model_type}): {e}")
        st.error(traceback.format_exc())
        if model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]: return None
        else: return None, None

# --- Funzioni Predict (Standard e Seq2Seq) ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict LSTM: Modello, scaler o config mancanti."); return None

    model_type = config.get("model_type", "LSTM")
    if model_type != "LSTM": st.error(f"Funzione predict chiamata su modello non LSTM (tipo: {model_type})"); return None

    training_mode = config.get("training_mode", "standard")
    num_quantiles = len(config["quantiles"]) if training_mode == 'quantile' else 1
    
    output_steps = config["output_window"]
    num_targets = len(config["target_columns"])

    model.eval()
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inp_tens)

        # Il forward del modello ora restituisce (batch, seq, targets, quantiles) se num_quantiles > 1
        # o (batch, seq, targets) se num_quantiles == 1
        out_np = output.cpu().numpy().squeeze(0) # -> (seq, targets, quantiles) or (seq, targets)
        
        if training_mode == 'quantile':
            # out_np shape: (output_steps, num_targets, num_quantiles)
            preds_scaled_flat = out_np.reshape(-1, num_targets)
            preds_flat = scaler_targets.inverse_transform(preds_scaled_flat)
            preds = preds_flat.reshape(output_steps, num_targets, num_quantiles)
        else:
            # out_np shape: (output_steps, num_targets)
            preds = scaler_targets.inverse_transform(out_np)
            
        return preds
    except Exception as e:
        st.error(f"Errore durante predict LSTM: {e}")
        st.error(traceback.format_exc()); return None

def predict_seq2seq(model, past_data, future_forecast_data, scalers, config, device):
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]):
         st.error("Predict Seq2Seq: Input mancanti."); return None, None

    training_mode = config.get("training_mode", "standard")
    num_quantiles = len(config["quantiles"]) if training_mode == 'quantile' else 1
    
    output_steps_model = config["output_window_steps"]
    num_targets = len(config["target_columns"])

    scaler_past = scalers.get("past"); scaler_forecast = scalers.get("forecast"); scaler_targets = scalers.get("targets")
    if not all([scaler_past, scaler_forecast, scaler_targets]):
        st.error("Predict Seq2Seq: Scaler mancanti nel dizionario fornito."); return None, None

    model.eval()
    try:
        past_norm = scaler_past.transform(past_data)
        future_norm = scaler_forecast.transform(future_forecast_data)
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device)
        future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)

        with torch.no_grad(): 
            output, attention_weights = model(past_tens, future_tens, teacher_forcing_ratio=0.0)

        out_np = output.cpu().numpy().squeeze(0)

        if training_mode == 'quantile':
            # out_np shape: (output_steps, num_targets, num_quantiles)
            preds_scaled_flat = out_np.reshape(-1, num_targets)
            preds_flat = scaler_targets.inverse_transform(preds_scaled_flat)
            preds = preds_flat.reshape(output_steps_model, num_targets, num_quantiles)
        else:
            # out_np shape: (output_steps, num_targets)
            preds = scaler_targets.inverse_transform(out_np)
            
        return preds, attention_weights
    except Exception as e:
        st.error(f"Errore durante predict Seq2Seq: {e}")
        st.error(traceback.format_exc()); return None, None


def predict_with_uncertainty(model, input_data, scaler_features, scaler_targets, config, device, num_passes=25, progress_indicator=None):
    """Esegue la previsione LSTM più volte (MC Dropout) aggiornando un indicatore di progresso."""
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Predict (Uncertainty) LSTM: Modello, scaler o config mancanti.")
        return None, None

    model.train()
    predictions_list = []
    
    try:
        inp_norm = scaler_features.transform(input_data)
        inp_tens = torch.FloatTensor(inp_norm).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for i in range(num_passes):
                output = model(inp_tens)
                out_np = output.cpu().numpy().squeeze(0)
                preds_single_pass = scaler_targets.inverse_transform(out_np)
                predictions_list.append(preds_single_pass)
                
                if progress_indicator:
                    progress_percentage = (i + 1) / num_passes
                    progress_text = f"Calcolo Incertezza: Passaggio {i + 1}/{num_passes}"
                    progress_indicator.progress(progress_percentage, text=progress_text)

    except Exception as e:
        st.error(f"Errore durante il ciclo di predict_with_uncertainty (LSTM): {e}")
        st.error(traceback.format_exc())
        model.eval()
        return None, None

    model.eval()

    if not predictions_list:
        st.error("Nessuna previsione valida generata durante il MC Dropout (LSTM).")
        return None, None

    predictions_array = np.array(predictions_list)
    mean_prediction = np.mean(predictions_array, axis=0)
    std_prediction = np.std(predictions_array, axis=0)

    return mean_prediction, std_prediction


def predict_seq2seq_with_uncertainty(model, past_data, future_forecast_data, scalers, config, device, num_passes=25, progress_indicator=None):
    """Esegue la previsione Seq2Seq più volte (MC Dropout) aggiornando un indicatore di progresso."""
    if not all([model, past_data is not None, future_forecast_data is not None, scalers, config, device]):
        st.error("Predict (Uncertainty) Seq2Seq: Input mancanti.")
        return None, None, None

    model.train()
    predictions_list = []
    attention_list = []
    
    scaler_past = scalers.get("past")
    scaler_forecast = scalers.get("forecast")
    scaler_targets = scalers.get("targets")

    try:
        past_norm = scaler_past.transform(past_data)
        future_norm = scaler_forecast.transform(future_forecast_data)
        past_tens = torch.FloatTensor(past_norm).unsqueeze(0).to(device)
        future_tens = torch.FloatTensor(future_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            for i in range(num_passes):
                output, attention_weights = model(past_tens, future_tens, teacher_forcing_ratio=0.0)
                out_np = output.cpu().numpy().squeeze(0)
                preds_single_pass = scaler_targets.inverse_transform(out_np)
                predictions_list.append(preds_single_pass)
                if attention_weights is not None:
                    attention_list.append(attention_weights.cpu().numpy().squeeze(0))
                
                if progress_indicator:
                    progress_percentage = (i + 1) / num_passes
                    progress_text = f"Calcolo Incertezza: Passaggio {i + 1}/{num_passes}"
                    progress_indicator.progress(progress_percentage, text=progress_text)

    except Exception as e:
        st.error(f"Errore durante il ciclo di predict_seq2seq_with_uncertainty: {e}")
        st.error(traceback.format_exc())
        model.eval()
        return None, None, None

    model.eval()

    if not predictions_list:
        st.error("Nessuna previsione valida generata durante il MC Dropout (Seq2Seq).")
        return None, None, None
        
    predictions_array = np.array(predictions_list)
    mean_prediction = np.mean(predictions_array, axis=0)
    std_prediction = np.std(predictions_array, axis=0)

    mean_attention = None
    if attention_list:
        attention_array = np.array(attention_list)
        mean_attention = np.mean(attention_array, axis=0)
    
    return mean_prediction, std_prediction, mean_attention


def predict_seq2seq_autoregressive(model, past_data_np, future_forecast_features_np, 
                                   last_known_target_np, scalers, config, device):
    """
    Esegue la predizione in modalità puramente autoregressiva.
    """
    model.eval()
    
    # Recuperiamo gli scaler corretti
    scaler_past = scalers.get("past")
    # NOTA: Il nome 'forecast_decoder_input' deve corrispondere a come viene salvato e caricato lo scaler
    scaler_decoder_input = scalers.get("forecast_decoder_input") 
    scaler_targets = scalers.get("targets")

    if not all([scaler_past, scaler_decoder_input, scaler_targets]):
        st.error("Predict Seq2Seq Autoregressive: Uno o più scaler richiesti non trovati nel dizionario.")
        return None, None

    # Normalizziamo i dati storici per l'encoder
    past_data_norm = scaler_past.transform(past_data_np)
    past_tens = torch.FloatTensor(past_data_norm).unsqueeze(0).to(device)
    
    output_window = config["output_window_steps"]
    target_dim = len(config["target_columns"])
    
    outputs_norm = torch.zeros(output_window, target_dim).to(device)

    with torch.no_grad():
        encoder_outputs, encoder_hidden, encoder_cell = model.encoder(past_tens)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        # --- Primo Step ---
        # Combiniamo l'ultimo target reale (unscaled) con le prime feature future (unscaled)
        first_step_input_unscaled = np.concatenate([
            future_forecast_features_np[0:1, :],
            last_known_target_np.reshape(1, -1)
        ], axis=1)

        # Scaliamo l'input combinato per il primo step del decoder
        decoder_input_step_norm = torch.FloatTensor(
            scaler_decoder_input.transform(first_step_input_unscaled)
        ).unsqueeze(1).to(device)

        # --- Ciclo Autoregressivo ---
        for t in range(output_window):
            # Il decoder autoregressivo ora richiede anche encoder_outputs
            prediction_step_norm, decoder_hidden, decoder_cell, _ = model.decoder(
                decoder_input_step_norm, decoder_hidden, decoder_cell, encoder_outputs
            )
            
            outputs_norm[t] = prediction_step_norm.squeeze()

            if t < output_window - 1:
                # Prepariamo l'input per il prossimo step
                # 1. De-normalizziamo la predizione corrente per averla in scala originale
                prediction_step_unscaled = scaler_targets.inverse_transform(
                    prediction_step_norm.cpu().numpy().reshape(1, -1)
                )

                # 2. Prendiamo le prossime feature future (unscaled)
                next_forecast_features_unscaled = future_forecast_features_np[t+1:t+2, :]

                # 3. Combiniamo la feature futura con la predizione (entrambe unscaled)
                next_step_input_unscaled = np.concatenate([
                    next_forecast_features_unscaled,
                    prediction_step_unscaled
                ], axis=1)

                # 4. Scaliamo l'input combinato per il prossimo step del decoder
                decoder_input_step_norm = torch.FloatTensor(
                    scaler_decoder_input.transform(next_step_input_unscaled)
                ).unsqueeze(1).to(device)
                
    # De-normalizziamo l'output finale completo
    predictions = scaler_targets.inverse_transform(outputs_norm.cpu().numpy())
    return predictions, None # Restituisce None per i pesi di attention


def plot_attention_weights(attention_weights, input_labels, output_labels):
    fig = go.Figure(data=go.Heatmap(
                    z=attention_weights,
                    x=input_labels,
                    y=output_labels,
                    colorscale='Blues'))
    fig.update_layout(
        title='Mappa di Attenzione',
        xaxis_title='Passi Temporali di Input',
        yaxis_title='Passi Temporali di Output Previsti'
    )
    return fig

# SOSTITUISCI QUESTA INTERA FUNZIONE
def plot_predictions(predictions, config, start_time=None, actual_data=None, actual_data_label="Dati Reali CSV", uncertainty=None):
    if config is None or predictions is None: return []

    target_cols = config["target_columns"]
    output_steps = predictions.shape[0]
    attribution_text = ATTRIBUTION_PHRASE
    figs = []

    for i, sensor in enumerate(target_cols):
        fig = go.Figure()
        if start_time:
            # --- CORREZIONE QUI ---
            # La previsione inizia a 'start_time', non un passo dopo. Rimosso il "+ 1".
            time_steps_datetime = [start_time + timedelta(minutes=30 * step) for step in range(output_steps)]
            # --- FINE CORREZIONE ---
            x_axis, x_title = time_steps_datetime, "Data e Ora"
            x_tick_format = "%d/%m %H:%M"
        else:
            time_steps_relative = np.arange(1, output_steps + 1) * 0.5
            x_axis, x_title = time_steps_relative, "Ore Future (passi da 30 min)"
            x_tick_format = None
        x_axis_np = np.array(x_axis)

        station_name_graph = get_station_label(sensor, short=False)
        plot_title_base = f'Test: Previsto vs Reale - {station_name_graph}' if actual_data is not None else f'Previsione Probabilistica - {station_name_graph}'
        plot_title = f'{plot_title_base}<br><span style="font-size:10px;">{attribution_text}</span>'

        unit_match = re.search(r'\((.*?)\)|\[(.*?)\]', sensor)
        y_axis_unit = "m"; y_axis_title_h = "Livello H (m)"
        if unit_match:
            unit_content = unit_match.group(1) or unit_match.group(2)
            if unit_content:
                y_axis_unit = unit_content.strip()
                y_axis_title_h = f"Livello H ({y_axis_unit})"
        
        # Logica per gestire incertezza (Quantile o MC Dropout)
        if predictions.ndim == 3: # Modalità Quantile
            # predictions shape: (seq_len, num_targets, num_quantiles)
            num_quantiles_local = predictions.shape[2]
            lower_bound = predictions[:, i, 0]
            if num_quantiles_local >= 3:
                median_prediction = predictions[:, i, 1]
                upper_bound = predictions[:, i, 2]
            elif num_quantiles_local == 2:
                # Se sono presenti solo 2 quantili (lower, upper), usiamo la media come linea centrale
                upper_bound = predictions[:, i, 1]
                median_prediction = (lower_bound + upper_bound) / 2.0
            else:
                # Caso degrado: un solo quantile -> tratti come puntuale
                puntual_prediction = predictions[:, i, 0]
                fig.add_trace(go.Scatter(
                    x=x_axis_np, y=puntual_prediction, mode='lines+markers', name=f'Previsto H ({y_axis_unit})', yaxis='y1'
                ))
                figs.append(fig)
                continue

            fig.add_trace(go.Scatter(
                x=np.concatenate([x_axis_np, x_axis_np[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='Banda di Incertezza Quantile'
            ))
            fig.add_trace(go.Scatter(
                x=x_axis_np, y=median_prediction, mode='lines+markers', name=f'Previsto H (Mediana)', yaxis='y1'
            ))

        else: # Modalità Standard (Puntuale)
            puntual_prediction = predictions[:, i]
            if uncertainty is not None and uncertainty.shape == predictions.shape:
                upper_bound_mc = puntual_prediction + 1.96 * uncertainty[:, i]
                lower_bound_mc = puntual_prediction - 1.96 * uncertainty[:, i]
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_axis_np, x_axis_np[::-1]]),
                    y=np.concatenate([upper_bound_mc, lower_bound_mc[::-1]]),
                    fill='toself', fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='Incertezza (95% CI)'
                ))
            fig.add_trace(go.Scatter(
                x=x_axis_np, y=puntual_prediction, mode='lines+markers', name=f'Previsto H ({y_axis_unit})', yaxis='y1'
            ))

        if actual_data is not None:
            if actual_data.ndim == 2 and actual_data.shape[0] == output_steps and actual_data.shape[1] == len(target_cols):
                fig.add_trace(go.Scatter(
                    x=x_axis_np, y=actual_data[:, i], mode='lines', name=f'{actual_data_label} H ({y_axis_unit})',
                    line=dict(color='green', dash='dashdot'), yaxis='y1'
                ))
            else:
                st.warning(f"Shape 'actual_data' ({actual_data.shape}) non compatibile per {sensor}.")

        threshold_info = SIMULATION_THRESHOLDS.get(sensor, {})
        soglia_attenzione = threshold_info.get('attenzione')
        soglia_allerta = threshold_info.get('allerta')
        if soglia_attenzione is not None:
            fig.add_hline(y=soglia_attenzione, line_dash="dash", line_color="orange", annotation_text=f"Att.H({soglia_attenzione:.2f})", annotation_position="bottom right", layer="below")
        if soglia_allerta is not None:
            fig.add_hline(y=soglia_allerta, line_dash="dash", line_color="red", annotation_text=f"All.H({soglia_allerta:.2f})", annotation_position="top right", layer="below")

        sensor_info = STATION_COORDS.get(sensor)
        sensor_code_plot = sensor_info.get('sensor_code') if sensor_info else None
        has_discharge_data = False
        if sensor_code_plot and sensor_code_plot in RATING_CURVES:
            predicted_H_values = predictions[:, i, 1] if predictions.ndim == 3 else predictions[:, i]
            predicted_Q_values = calculate_discharge_vectorized(sensor_code_plot, predicted_H_values)
            valid_Q_mask = pd.notna(predicted_Q_values)
            if np.any(valid_Q_mask):
                has_discharge_data = True
                fig.add_trace(go.Scatter(
                    x=x_axis_np[valid_Q_mask], y=predicted_Q_values[valid_Q_mask], mode='lines', name='Portata Prevista Q (m³/s)',
                    line=dict(color='firebrick', dash='dot'), yaxis='y2'
                ))
        
        try:
            fig.update_layout(title=plot_title, height=400, margin=dict(l=60, r=60, t=70, b=50), hovermode="x unified", template="plotly_white")
            fig.update_xaxes(title_text=x_title)
            if x_tick_format: fig.update_xaxes(tickformat=x_tick_format)
            fig.update_yaxes(title=dict(text=y_axis_title_h, font=dict(color="#1f77b4")), tickfont=dict(color="#1f77b4"), side="left", rangemode='tozero')

            if has_discharge_data:
                fig.update_layout(
                     yaxis2=dict(title=dict(text="Portata Q (m³/s)", font=dict(color="firebrick")),
                                 tickfont=dict(color="firebrick"), overlaying="y", side="right", rangemode='tozero', showgrid=False),
                     legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
            else:
                 fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))

        except Exception as e_layout:
            st.error(f"Errore layout Plotly: {e_layout}")
        figs.append(fig)
    return figs

# --- Funzioni Fetch Dati Google Sheet (Dashboard e Simulazione) ---
@st.cache_data(ttl=DASHBOARD_REFRESH_INTERVAL_SECONDS, show_spinner="Recupero dati aggiornati dal foglio Google...")
def fetch_gsheet_dashboard_data(_cache_key_time, sheet_id, relevant_columns, date_col, date_format,
                                fetch_all=False, num_rows_default=DASHBOARD_HISTORY_ROWS):
    mode = "tutti i dati" if fetch_all else f"ultime {num_rows_default} righe"
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] ESECUZIONE fetch_gsheet_dashboard_data ({mode})")
    actual_fetch_time = datetime.now(italy_tz)
    try:
        if "GOOGLE_CREDENTIALS" not in st.secrets:
            return None, "Errore: Credenziali Google mancanti.", actual_fetch_time
        credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
        gc = gspread.authorize(credentials); sh = gc.open_by_key(sheet_id); worksheet = sh.sheet1
        all_values = worksheet.get_all_values()
        if not all_values or len(all_values) < 2: return None, "Errore: Foglio Google vuoto o con solo intestazione.", actual_fetch_time

        headers = all_values[0]
        if fetch_all: data_rows = all_values[1:]
        else: start_index = max(1, len(all_values) - num_rows_default); data_rows = all_values[start_index:]

        missing_cols = [col for col in relevant_columns if col not in headers]
        if missing_cols: return None, f"Errore: Colonne GSheet richieste mancanti: {', '.join(missing_cols)}", actual_fetch_time

        df = pd.DataFrame(data_rows, columns=headers)
        cols_to_select = [c for c in relevant_columns if c in df.columns]
        df = df[cols_to_select]

        error_parsing = []
        for col in df.columns:
            if col == date_col:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format if date_format else None, errors='coerce', infer_datetime_format=True)
                    if df[col].isnull().any(): error_parsing.append(f"Formato data non valido per '{col}'")
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

        error_message = "Attenzione conversione dati GSheet: " + " | ".join(error_parsing) if error_parsing else None
        return df, error_message, actual_fetch_time
    except gspread.exceptions.APIError as api_e:
        try: error_details = api_e.response.json(); error_msg = error_details.get('error', {}).get('message', str(api_e)); status = error_details.get('error', {}).get('code', 'N/A'); error_msg = f"Codice {status}: {error_msg}";
        except: error_msg = str(api_e)
        return None, f"Errore API Google Sheets: {error_msg}", actual_fetch_time
    except gspread.exceptions.SpreadsheetNotFound: return None, f"Errore: Foglio Google non trovato (ID: '{sheet_id}').", actual_fetch_time
    except Exception as e: return None, f"Errore imprevisto recupero dati GSheet: {type(e).__name__} - {e}\n{traceback.format_exc()}", actual_fetch_time

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
            except Exception as e_clean_num: st.warning(f"Problema pulizia GSheet colonna '{col}' per simulazione: {e_clean_num}."); df_mapped[col] = np.nan

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
             except Exception as e_date_clean: st.warning(f"Errore conversione/pulizia data GSheet simulazione '{date_col_model_name}': {e_date_clean}."); date_col_model_name = None

        try:
            cols_present_final = [c for c in required_model_cols_fetch if c in df_mapped.columns]
            df_final = df_mapped[cols_present_final].copy()
        except KeyError as e_key: return None, f"Errore selezione/ordine colonne finali simulazione: '{e_key}' mancante.", None

        numeric_cols_to_fill = df_final.select_dtypes(include=np.number).columns
        nan_count_before = df_final[numeric_cols_to_fill].isnull().sum().sum()
        if nan_count_before > 0:
             st.warning(f"Trovati {nan_count_before} valori NaN nei dati GSheet per simulazione. Applico ffill/bfill.")
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
# MODIFICATA: per accettare i pesi e calcolare la loss pesata
def train_model(X_scaled_full, y_scaled_full, sample_weights_full, # NUOVO ARGOMENTO
                input_size, output_size, output_window_steps,
                hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2,
                save_strategy='migliore', preferred_device='auto', 
                n_splits_cv=3, loss_function_name="MSELoss",
                training_mode='standard', quantiles=None,
                _model_to_continue_train=None,
                split_method="Temporale", validation_size=0.2,
                use_weighted_loss=False, use_magnitude_loss=False, target_threshold=0.5, weight_exponent=1.0):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training LSTM (Weighted Loss: {use_weighted_loss})...")
    
    device = torch.device('cuda' if ('auto' in preferred_device.lower() and torch.cuda.is_available()) else 'cpu')
    print(f"Training LSTM userà: {device}")

    # --- INIZIO NUOVA LOGICA PER CRITERION ---
    # Per la loss pesata, dobbiamo calcolare la loss per ogni campione, quindi 'reduction' deve essere 'none'.
    reduction_strategy = 'none' if use_weighted_loss or (loss_function_name == "DynamicWeightedMSE" and use_magnitude_loss) or (training_mode == 'quantile' and use_magnitude_loss) else 'mean'

    criterion = None
    if training_mode == 'quantile':
        if use_magnitude_loss:
            st.info(f"Loss Quantile Pesata per Magnitudo attivata con Soglia: {target_threshold}, Esponente: {weight_exponent}")
            criterion = WeightedQuantileLoss(
                quantiles=quantiles, 
                threshold=target_threshold, 
                exponent=weight_exponent
            )
        else:
            criterion = QuantileLoss(quantiles=quantiles, reduction=reduction_strategy)
    elif loss_function_name == "HuberLoss": 
        criterion = nn.HuberLoss(reduction=reduction_strategy)
    elif loss_function_name == "DynamicWeightedMSE":
        st.info(f"Loss Dinamica attivata con Soglia (scalata): {target_threshold}, Esponente: {weight_exponent}")
        criterion = DynamicWeightedMSELoss(
            threshold=target_threshold, 
            exponent=weight_exponent, 
            reduction=reduction_strategy,
            use_unscaled_targets=False 
        )
    elif loss_function_name == "DilateLoss":
        # DilateLoss è speciale e restituisce già la loss per campione del batch, quindi reduction='none' è implicito
        criterion = DilateLoss(alpha=0.5, gamma=0.1, device=device, reduction='none')
    else: 
        criterion = nn.MSELoss(reduction=reduction_strategy)
    # --- FINE NUOVA LOGICA PER CRITERION ---
    
    if _model_to_continue_train is not None: model = _model_to_continue_train.to(device)
    else:
        num_q = len(quantiles) if training_mode == 'quantile' else 1
        model = HydroLSTM(input_size, hidden_size, output_size, output_window_steps, num_layers, dropout, num_quantiles=num_q).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    weights_train, weights_val = None, None # NUOVO
    if "Temporale" in split_method:
        tscv = TimeSeriesSplit(n_splits=n_splits_cv)
        train_indices, val_indices = list(tscv.split(X_scaled_full))[-1]
        
        X_train, y_train = X_scaled_full[train_indices], y_scaled_full[train_indices]
        X_val, y_val = X_scaled_full[val_indices], y_scaled_full[val_indices]
        if use_weighted_loss and sample_weights_full is not None: # NUOVO
            weights_train = sample_weights_full[train_indices]
            weights_val = sample_weights_full[val_indices]
    else: # Logica Casuale
        st.info(f"Suddivisione casuale con {validation_size*100:.0f}% dei dati per la validazione.")
        
        # NUOVO: Aggiungi i pesi allo split se necessario
        arrays_to_split = (X_scaled_full, y_scaled_full)
        if use_weighted_loss and sample_weights_full is not None:
            arrays_to_split += (sample_weights_full,)

        split_results = train_test_split(
            *arrays_to_split,
            test_size=validation_size, 
            shuffle=True, 
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = split_results[0], split_results[1], split_results[2], split_results[3]
        if use_weighted_loss and sample_weights_full is not None:
             weights_train, weights_val = split_results[4], split_results[5]

    # MODIFICATO: Passa i pesi al DataLoader
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, weights_train), batch_size=batch_size, shuffle=True)
    # La validazione non richiede i pesi perché non facciamo backpropagation, ma potremmo usarli per un report più accurato
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size) if len(X_val) > 0 else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    progress_bar = st.progress(0.0, text="Training LSTM: Inizio...")
    status_text = st.empty()
    loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        if v_loss and any(v is not None for v in v_loss): fig.add_trace(go.Scatter(y=v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training LSTM per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        # MODIFICATO: Unpack del batch
        batch_items = train_loader
        for batch in batch_items:
            # Gestisce entrambi i casi (con e senza pesi)
            if use_weighted_loss:
                X_batch, y_batch, weights_batch = batch
                weights_batch = weights_batch.to(device)
            else:
                X_batch, y_batch = batch
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            # --- INIZIO DEL NUOVO BLOCCO DI CALCOLO LOSS (DA SOSTITUIRE) ---

            # `criterion` calcola la loss per campione (pesata per magnitudo se attivata).
            loss_per_sample = criterion(outputs, y_batch)

            # Se la pesatura temporale (dummy) è attiva, la applichiamo.
            if use_weighted_loss and 'weights_batch' in locals() and weights_batch is not None:
                # Assicuriamoci che le dimensioni siano compatibili prima della moltiplicazione.
                # Se la loss ha più dimensioni (es. per step temporale/quantili), facciamo la media su quelle.
                if loss_per_sample.ndim > 1:
                    loss_per_sample = loss_per_sample.mean(dim=tuple(range(1, loss_per_sample.ndim)))
                
                # Moltiplica la loss (già pesata per magnitudo) per il peso temporale.
                final_loss_per_sample = loss_per_sample * weights_batch
                loss = final_loss_per_sample.mean() # Calcola la media finale per ottenere uno scalare.
            else:
                # Se NON c'è pesatura temporale, dobbiamo comunque ridurre la loss per campione a uno scalare.
                # Semplicemente calcoliamo la media.
                loss = loss_per_sample.mean()

            # --- FINE DEL NUOVO BLOCCO DI CALCOLO LOSS ---

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        train_losses.append(epoch_train_loss / len(train_loader.dataset))
        
        epoch_val_loss = None
        if val_loader:
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    # --- INIZIO BLOCCO DI VALIDAZIONE CORRETTO (DA SOSTITUIRE) ---

                    # Calcola la loss per campione (sarà un tensore se reduction='none')
                    loss_per_sample_val = criterion(outputs, y_batch) # Usa y_tar per seq2seq

                    # Indipendentemente da tutto, dobbiamo ridurre la loss di validazione a uno scalare
                    # facendo la media. La validazione non usa MAI pesi.
                    val_loss_unweighted = loss_per_sample_val.mean()

                    # Ora val_loss_unweighted è garantito essere uno scalare.
                    epoch_val_loss_sum += val_loss_unweighted.item() * X_batch.size(0) # Usa x_enc.size(0) per seq2seq

                    # --- FINE BLOCCO DI VALIDAZIONE CORRETTO ---
            epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            scheduler.step(epoch_val_loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
             val_losses.append(None)

        val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "N/A"
        status_text.markdown(f"Epoca {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_loss_str}")

        progress_bar.progress((epoch + 1) / epochs)
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

    if save_strategy == 'migliore' and best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        st.success(f"Caricato modello con Val Loss migliore: {best_val_loss:.6f}")

    return model, (train_losses, []), (val_losses, [])


def train_model_seq2seq(X_enc_scaled_full, X_dec_scaled_full, y_tar_scaled_full, sample_weights_full,
                        model,
                        output_window_steps, epochs=50, batch_size=32, learning_rate=0.001,
                        save_strategy='migliore', preferred_device='auto', teacher_forcing_ratio_schedule=None,
                        n_splits_cv=3, loss_function_name="MSELoss",
                        training_mode='standard', quantiles=None,
                        split_method="Temporale", validation_size=0.2,
                        use_weighted_loss=False, use_magnitude_loss=False, target_threshold=0.5, weight_exponent=1.0):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training Seq2Seq (Weighted: {use_weighted_loss}, loss={loss_function_name})...")
    
    device = torch.device('cuda' if ('auto' in preferred_device.lower() and torch.cuda.is_available()) else 'cpu')
    model.to(device)

    reduction_strategy = 'none' if use_weighted_loss or (loss_function_name == "DynamicWeightedMSE" and use_magnitude_loss) or (training_mode == 'quantile' and use_magnitude_loss) else 'mean'
    criterion = None
    if training_mode == 'quantile':
        if use_magnitude_loss:
            st.info(f"Loss Quantile Pesata per Magnitudo attivata con Soglia: {target_threshold}, Esponente: {weight_exponent}")
            criterion = WeightedQuantileLoss(
                quantiles=quantiles, 
                threshold=target_threshold, 
                exponent=weight_exponent
            )
        else:
            criterion = QuantileLoss(quantiles=quantiles, reduction=reduction_strategy)
    elif loss_function_name == "HuberLoss": 
        criterion = nn.HuberLoss(reduction=reduction_strategy)
    elif loss_function_name == "DynamicWeightedMSE":
        st.info(f"Loss Dinamica attivata con Soglia (scalata): {target_threshold}, Esponente: {weight_exponent}")
        criterion = DynamicWeightedMSELoss(
            threshold=target_threshold, 
            exponent=weight_exponent, 
            reduction=reduction_strategy,
            use_unscaled_targets=False 
        )
    elif loss_function_name == "DilateLoss":
        criterion = DilateLoss(alpha=0.5, gamma=0.1, device=device, reduction='none')
    else: 
        criterion = nn.MSELoss(reduction=reduction_strategy)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    weights_train, weights_val = None, None
    if "Temporale" in split_method:
        tscv = TimeSeriesSplit(n_splits=n_splits_cv)
        train_indices, val_indices = list(tscv.split(X_enc_scaled_full))[-1]
        
        X_enc_train, X_dec_train, y_tar_train = X_enc_scaled_full[train_indices], X_dec_scaled_full[train_indices], y_tar_scaled_full[train_indices]
        X_enc_val, X_dec_val, y_tar_val = X_enc_scaled_full[val_indices], X_dec_scaled_full[val_indices], y_tar_scaled_full[val_indices]
        if use_weighted_loss and sample_weights_full is not None:
            weights_train = sample_weights_full[train_indices]
            weights_val = sample_weights_full[val_indices]
    else: 
        st.info(f"Suddivisione casuale con {validation_size*100:.0f}% dei dati per la validazione.")
        arrays_to_split = (X_enc_scaled_full, X_dec_scaled_full, y_tar_scaled_full)
        if use_weighted_loss and sample_weights_full is not None:
            arrays_to_split += (sample_weights_full,)

        split_results = train_test_split(*arrays_to_split, test_size=validation_size, shuffle=True, random_state=42)
        
        X_enc_train, X_enc_val = split_results[0], split_results[1]
        X_dec_train, X_dec_val = split_results[2], split_results[3]
        y_tar_train, y_tar_val = split_results[4], split_results[5]
        if use_weighted_loss and sample_weights_full is not None:
            weights_train, weights_val = split_results[6], split_results[7]

    train_loader = DataLoader(TimeSeriesDataset(X_enc_train, X_dec_train, y_tar_train, weights_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_enc_val, X_dec_val, y_tar_val), batch_size=batch_size) if len(X_enc_val) > 0 else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    progress_bar = st.progress(0.0, text="Training Seq2Seq: Inizio...")
    status_text = st.empty()
    loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        if v_loss and any(v is not None for v in v_loss): fig.add_trace(go.Scatter(y=v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training Seq2Seq per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        current_tf_ratio = 0.5
        if teacher_forcing_ratio_schedule:
            start_tf, end_tf = teacher_forcing_ratio_schedule
            current_tf_ratio = max(end_tf, start_tf - (start_tf - end_tf) * epoch / (epochs -1 if epochs > 1 else 1))
        
        for batch in train_loader:
            if use_weighted_loss:
                x_enc, x_dec, y_tar, weights_batch = batch
                weights_batch = weights_batch.to(device)
            else:
                x_enc, x_dec, y_tar = batch

            x_enc, x_dec, y_tar = x_enc.to(device), x_dec.to(device), y_tar.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(x_enc, x_dec, teacher_forcing_ratio=current_tf_ratio)
            # --- INIZIO DEL NUOVO BLOCCO DI CALCOLO LOSS (DA SOSTITUIRE) ---

            # `criterion` calcola la loss per campione (pesata per magnitudo se attivata).
            loss_per_sample = criterion(outputs, y_tar)

            # Se la pesatura temporale (dummy) è attiva, la applichiamo.
            if use_weighted_loss and 'weights_batch' in locals() and weights_batch is not None:
                # Assicuriamoci che le dimensioni siano compatibili prima della moltiplicazione.
                # Se la loss ha più dimensioni (es. per step temporale/quantili), facciamo la media su quelle.
                if loss_per_sample.ndim > 1:
                    loss_per_sample = loss_per_sample.mean(dim=tuple(range(1, loss_per_sample.ndim)))
                
                # Moltiplica la loss (già pesata per magnitudo) per il peso temporale.
                final_loss_per_sample = loss_per_sample * weights_batch
                loss = final_loss_per_sample.mean() # Calcola la media finale per ottenere uno scalare.
            else:
                # Se NON c'è pesatura temporale, dobbiamo comunque ridurre la loss per campione a uno scalare.
                # Semplicemente calcoliamo la media.
                loss = loss_per_sample.mean()

            # --- FINE DEL NUOVO BLOCCO DI CALCOLO LOSS ---

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_enc.size(0)
        
        train_losses.append(epoch_train_loss / len(train_loader.dataset))
        
        epoch_val_loss = None
        if val_loader:
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for x_enc, x_dec, y_tar in val_loader:
                    x_enc, x_dec, y_tar = x_enc.to(device), x_dec.to(device), y_tar.to(device)
                    outputs, _ = model(x_enc, x_dec, teacher_forcing_ratio=0.0)
                    # --- INIZIO BLOCCO DI VALIDAZIONE CORRETTO (DA SOSTITUIRE) ---

                    # Calcola la loss per campione (sarà un tensore se reduction='none')
                    loss_per_sample_val = criterion(outputs, y_tar) # Usa y_tar per seq2seq

                    # Indipendentemente da tutto, dobbiamo ridurre la loss di validazione a uno scalare
                    # facendo la media. La validazione non usa MAI pesi.
                    val_loss_unweighted = loss_per_sample_val.mean()

                    # Ora val_loss_unweighted è garantito essere uno scalare.
                    epoch_val_loss_sum += val_loss_unweighted.item() * x_enc.size(0) # Usa x_enc.size(0) per seq2seq

                    # --- FINE BLOCCO DI VALIDAZIONE CORRETTO ---
            epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            scheduler.step(epoch_val_loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_losses.append(None)

        val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "N/A"
        status_text.markdown(f"Epoca {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_loss_str}")

        progress_bar.progress((epoch + 1) / epochs)
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

    if save_strategy == 'migliore' and best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        st.success(f"Caricato modello con Val Loss migliore: {best_val_loss:.6f}")

    return model, (train_losses, []), (val_losses, [])


def train_model_seq2seq_autoregressive(X_enc_scaled_full, X_dec_scaled_full, y_tar_scaled_full,
                        model,
                        output_window_steps, epochs=50, batch_size=32, learning_rate=0.001,
                        save_strategy='migliore', preferred_device='auto', teacher_forcing_ratio_schedule=None,
                        n_splits_cv=3, loss_function_name="MSELoss",
                        training_mode='standard', quantiles=None,
                        split_method="Temporale", validation_size=0.2,
                        use_magnitude_loss=False, target_threshold=0.5, weight_exponent=1.0):
    print(f"[{datetime.now(italy_tz).strftime('%H:%M:%S')}] Avvio training Seq2Seq Autoregressive (loss={loss_function_name})...")
    
    device = torch.device('cuda' if ('auto' in preferred_device.lower() and torch.cuda.is_available()) else 'cpu')
    model.to(device)

    reduction_strategy = 'none' if (loss_function_name == "DynamicWeightedMSE" and use_magnitude_loss) or (training_mode == 'quantile' and use_magnitude_loss) else 'mean'
    criterion = None
    if training_mode == 'quantile':
        if use_magnitude_loss:
            st.info(f"Loss Quantile Pesata per Magnitudo attivata con Soglia: {target_threshold}, Esponente: {weight_exponent}")
            criterion = WeightedQuantileLoss(
                quantiles=quantiles, 
                threshold=target_threshold, 
                exponent=weight_exponent
            )
        else:
            criterion = QuantileLoss(quantiles=quantiles, reduction=reduction_strategy)
    elif loss_function_name == "HuberLoss": 
        criterion = nn.HuberLoss(reduction=reduction_strategy)
    elif loss_function_name == "DynamicWeightedMSE":
        st.info(f"Loss Dinamica attivata con Soglia (scalata): {target_threshold}, Esponente: {weight_exponent}")
        criterion = DynamicWeightedMSELoss(
            threshold=target_threshold, 
            exponent=weight_exponent, 
            reduction=reduction_strategy,
            use_unscaled_targets=False 
        )
    elif loss_function_name == "DilateLoss":
        criterion = DilateLoss(alpha=0.5, gamma=0.1, device=device, reduction='none')
    else: 
        criterion = nn.MSELoss(reduction=reduction_strategy)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    
    if "Temporale" in split_method:
        tscv = TimeSeriesSplit(n_splits=n_splits_cv)
        train_indices, val_indices = list(tscv.split(X_enc_scaled_full))[-1]
        
        X_enc_train, X_dec_train, y_tar_train = X_enc_scaled_full[train_indices], X_dec_scaled_full[train_indices], y_tar_scaled_full[train_indices]
        X_enc_val, X_dec_val, y_tar_val = X_enc_scaled_full[val_indices], X_dec_scaled_full[val_indices], y_tar_scaled_full[val_indices]
    else: 
        st.info(f"Suddivisione casuale con {validation_size*100:.0f}% dei dati per la validazione.")
        X_enc_train, X_enc_val, X_dec_train, X_dec_val, y_tar_train, y_tar_val = train_test_split(
            X_enc_scaled_full, X_dec_scaled_full, y_tar_scaled_full, test_size=validation_size, shuffle=True, random_state=42
        )

    train_loader = DataLoader(TimeSeriesDataset(X_enc_train, X_dec_train, y_tar_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_enc_val, X_dec_val, y_tar_val), batch_size=batch_size) if len(X_enc_val) > 0 else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    progress_bar = st.progress(0.0, text="Training Seq2Seq Autoregressive: Inizio...")
    status_text = st.empty()
    loss_chart_placeholder = st.empty()

    def update_loss_chart(t_loss, v_loss, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=t_loss, mode='lines', name='Train Loss'))
        if v_loss and any(v is not None for v in v_loss): fig.add_trace(go.Scatter(y=v_loss, mode='lines', name='Validation Loss'))
        fig.update_layout(title='Andamento Loss', xaxis_title='Epoca', yaxis_title='Loss', height=300, margin=dict(t=30, b=0), template="plotly_white")
        placeholder.plotly_chart(fig, use_container_width=True)

    st.write(f"Inizio training Seq2Seq Autoregressive per {epochs} epoche su {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        current_tf_ratio = 0.5
        if teacher_forcing_ratio_schedule:
            start_tf, end_tf = teacher_forcing_ratio_schedule
            current_tf_ratio = max(end_tf, start_tf - (start_tf - end_tf) * epoch / (epochs -1 if epochs > 1 else 1))
        
        for x_enc, x_dec, y_tar in train_loader:
            x_enc, x_dec, y_tar = x_enc.to(device), x_dec.to(device), y_tar.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(x_enc, x_dec, y_tar, teacher_forcing_ratio=current_tf_ratio)
            
            loss_per_sample = criterion(outputs, y_tar)
            loss = loss_per_sample.mean()

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_enc.size(0)
        
        train_losses.append(epoch_train_loss / len(train_loader.dataset))
        
        epoch_val_loss = None
        if val_loader:
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for x_enc, x_dec, y_tar in val_loader:
                    x_enc, x_dec, y_tar = x_enc.to(device), x_dec.to(device), y_tar.to(device)
                    outputs, _ = model(x_enc, x_dec, y_tar, teacher_forcing_ratio=0.0) # TF a 0 in validazione
                    
                    loss_per_sample_val = criterion(outputs, y_tar)
                    val_loss_unweighted = loss_per_sample_val.mean()
                    epoch_val_loss_sum += val_loss_unweighted.item() * x_enc.size(0)
            
            epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            scheduler.step(epoch_val_loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_losses.append(None)

        val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "N/A"
        status_text.markdown(f"Epoca {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_loss_str}")

        progress_bar.progress((epoch + 1) / epochs)
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

    if save_strategy == 'migliore' and best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        st.success(f"Caricato modello con Val Loss migliore: {best_val_loss:.6f}")

    return model, (train_losses, []), (val_losses, [])

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
    try:
        html_filename = f"{safe_filename_base}.html"; buf_html = io.StringIO(); fig.write_html(buf_html, include_plotlyjs='cdn')
        buf_html.seek(0); b64_html = base64.b64encode(buf_html.getvalue().encode()).decode()
        href_html = f'<a href="data:text/html;base64,{b64_html}" download="{html_filename}">{text_html}</a>'
    except Exception as e_html: print(f"Errore download HTML {safe_filename_base}: {e_html}"); href_html = "<i>Errore HTML</i>"
    try:
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
            if len(sensors_at_loc) > 1: type_abbr = {'Pioggia': 'P', 'Livello': 'L', 'Umidità': 'U'}.get(s_type, '?'); label = f"{loc_id} ({type_abbr})"
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
        'Variabile Dummy', 'Seasonality_Sin', 'Seasonality_Cos',
        'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
        'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
        "Umidita' Sensore 3452 (Montemurello)",
        'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
        'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
        'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
        'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
        'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
        'Cumulata Sensore 1295 (Arcevia)_cumulata_1h', 'Cumulata Sensore 1295 (Arcevia)_cumulata_3h',
        'Cumulata Sensore 1295 (Arcevia)_media_3h', 'Cumulata Sensore 1295 (Arcevia)_dev_std_3h',
        'Cumulata Sensore 2637 (Bettolelle)_cumulata_1h', 'Cumulata Sensore 2637 (Bettolelle)_cumulata_3h',
        'Cumulata Sensore 2637 (Bettolelle)_media_3h', 'Cumulata Sensore 2637 (Bettolelle)_dev_std_3h',
        'Cumulata Sensore 2858 (Barbara)_cumulata_1h', 'Cumulata Sensore 2858 (Barbara)_cumulata_3h',
        'Cumulata Sensore 2858 (Barbara)_media_3h', 'Cumulata Sensore 2858 (Barbara)_dev_std_3h',
        'Cumulata Sensore 2964 (Corinaldo)_cumulata_1h', 'Cumulata Sensore 2964 (Corinaldo)_cumulata_3h',
        'Cumulata Sensore 2964 (Corinaldo)_media_3h', 'Cumulata Sensore 2964 (Corinaldo)_dev_std_3h'
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
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.caption("Logo non trovato.")
    except Exception as e: st.warning(f"Impossibile caricare il logo: {e}")
    st.header('Impostazioni')

    st.subheader('Dati Storici (per Analisi/Training/Test)')
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
                try:
                    df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%y %H.%M', errors='raise')
                except ValueError:
                    try:
                        df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], format='%d/%m/%Y %H:%M', errors='raise')
                    except ValueError:
                        st.caption(f"Formato data CSV non standard ('{date_col_csv}'). Tentativo di inferenza automatica.")
                        try:
                            df_temp[date_col_csv] = pd.to_datetime(df_temp[date_col_csv], errors='coerce', infer_datetime_format=True)
                        except Exception as e_date_csv_infer:
                            raise ValueError(f"Errore conversione data CSV '{date_col_csv}': {e_date_csv_infer}")
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
        cfg = active_config_sess
        model_type_sess = cfg.get('model_type', 'LSTM')
        display_feedback_name = cfg.get("display_name", active_model_name_sess)
        st.success(f"Modello Attivo: **{display_feedback_name}** ({model_type_sess})")
        
        # Gestisce i modelli che usano le chiavi con "_steps"
        if model_type_sess in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
            st.caption(f"Input: {cfg['input_window_steps']}s | Forecast: {cfg.get('forecast_window_steps', 'N/A')}s | Output: {cfg['output_window_steps']}s")
        elif model_type_sess == "SpatioTemporalGNN":
            # Caso specifico per GNN che usa le chiavi corrette
            st.caption(f"Input: {cfg['input_window_steps']}s | Output: {cfg['output_window_steps']}s")
        elif model_type_sess == "Seq2SeqAutoregressivo":
            # Caso specifico autoregressivo: non usa forecast_window_steps
            st.caption(f"Input: {cfg['input_window_steps']}s | Output: {cfg['output_window_steps']}s")
        # Gestisce il modello LSTM standard che usa chiavi diverse
        else:
            st.caption(f"Input: {cfg['input_window']}s | Output: {cfg['output_window']}s")
    elif load_error_sidebar and active_model_name_sess not in [MODEL_CHOICE_NONE]: st.error(f"Caricamento modello '{active_model_name_sess}' fallito.")
    elif active_model_name_sess == MODEL_CHOICE_UPLOAD and not active_model_sess: st.info("Completa il caricamento manuale del modello LSTM.")
    elif active_model_name_sess == MODEL_CHOICE_NONE: st.info("Nessun modello selezionato per la simulazione/test.")
    st.divider()

    st.subheader("Configurazione Soglie Dashboard")
    with st.expander("Modifica Soglie di Allerta (per Dashboard)"):
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
    radio_options = ['Dashboard', 'Simulazione', 'Test Modello su Storico', 'Analisi Dati Storici', 'Allenamento Modello', 'Post-Training Modello']
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
        elif opt == 'Post-Training Modello':
            caption = "Affina un modello esistente con nuovi dati"
            if not model_ready: caption = "Richiede Modello attivo"; disabled = True
            elif not data_ready_csv: caption = "Richiede Dati CSV per post-training"; disabled = True
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
    # ... (Il resto della pagina Dashboard rimane invariato) ...
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
             fetch_time_ago = datetime.now(italy_tz) - last_fetch_dt_sess; fetch_secs_ago = int(fetch_time_ago.total_seconds() // 60)
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
    if not model_ready:
        st.warning("Seleziona un Modello attivo dalla sidebar per eseguire la Simulazione.")
        st.stop()

    st.info(f"Simulazione con Modello Attivo: **{st.session_state.active_model_name}** ({active_model_type})")
    st.subheader("Impostazioni Simulazione e Incertezza")

    is_quantile_model = active_config.get("training_mode") == "quantile"
    if is_quantile_model:
        st.info("Il modello attivo è stato addestrato in modalità **Quantile Regression**. L'incertezza è data dai quantili predetti e l'opzione MC Dropout è disabilitata.")
        num_passes_sim = 1
    else:
        num_passes_sim = st.number_input(
            "Numero di passaggi per calcolo incertezza (MC Dropout):",
            min_value=1, max_value=100, value=25, step=1,
            key="sim_num_passes",
            help="Valore più alto = stima più stabile ma calcolo più lento. Se impostato a 1, non calcola l'incertezza. Consigliato: 20-30."
        )

    target_columns_model = active_config['target_columns']
    if active_model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
        st.caption(f"Input Storico: {active_config['input_window_steps']}s | Input Forecast: {active_config['forecast_window_steps']}s | Output: {active_config['output_window_steps']}s")
        with st.expander(f"Dettagli Colonne Modello {active_model_type}"):
             st.markdown("**Feature Storiche (Input Encoder):**"); st.caption(f"`{', '.join(active_config['all_past_feature_columns'])}`")
             st.markdown("**Feature Forecast (Decoder Input):**"); st.caption(f"`{', '.join(active_config['forecast_input_columns'])}`")
             st.markdown("**Target (Output - Livello H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        input_steps_model = active_config['input_window_steps']
        output_steps_model = active_config['output_window_steps']
        past_feature_cols_model = active_config['all_past_feature_columns']
        forecast_feature_cols_model = active_config['forecast_input_columns']
        forecast_steps_model = active_config['forecast_window_steps']
    elif active_model_type == "SpatioTemporalGNN":
        st.caption(f"Input: {active_config['input_window_steps']}s | Output: {active_config['output_window_steps']}s")
        with st.expander(f"Dettagli Colonne Modello {active_model_type}"):
            st.markdown("**Nodi del Grafo (Input):**"); st.caption(f"`{', '.join(active_config['node_order'])}`")
            st.markdown("**Target (Output - Livello H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        input_steps_model = active_config['input_window_steps']
        output_steps_model = active_config['output_window_steps']
        node_columns_model = active_config['node_order']
    else: # LSTM Standard
        feature_columns_model = active_config.get("feature_columns", st.session_state.feature_columns)
        st.caption(f"Input: {active_config['input_window']}s | Output: {active_config['output_window']}s")
        with st.expander("Dettagli Colonne Modello LSTM"):
             st.markdown("**Feature Input:**"); st.caption(f"`{', '.join(feature_columns_model)}`")
             st.markdown("**Target (Output - Livello H):**"); st.caption(f"`{', '.join(active_config['target_columns'])}`")
        input_steps_model = active_config['input_window']
        output_steps_model = active_config['output_window']

    st.divider()
    if active_model_type == "Seq2SeqAutoregressivo":
        st.subheader(f"Preparazione Dati Input Simulazione {active_model_type}")
        st.markdown("**Passo 1: Recupero Dati Storici (Input Encoder)**")
        st.caption(f"Verranno recuperati gli ultimi {input_steps_model} steps per le feature storiche e l'ultimo valore di target noto.")
        
        # Logica per caricare i dati storici (simile ad altri modelli)
        # ... (Questa parte può essere riutilizzata/adattata)
        
        st.markdown("**Passo 2: Inserisci Input Futuri (SOLO METEO)**")
        st.caption(f"Inserisci i valori per le feature meteo future: `{', '.join(forecast_feature_cols_model)}`")
        
        # UI per l'inserimento delle feature meteo future
        future_forecast_df_initial = pd.DataFrame(index=range(output_steps_model), columns=forecast_feature_cols_model)
        # ... (logica per pre-compilare con valori di default se si vuole) ...
        edited_future_forecast_df = st.data_editor(future_forecast_df_initial.round(2), key="autoregressive_forecast_editor", num_rows="fixed", use_container_width=True)

        st.divider()
        st.markdown("**Passo 3: Esegui Simulazione**")
        if st.button(f"Esegui Simulazione {active_model_type}", type="primary", key="run_autoregressive_sim_button"):
            with st.spinner(f"Esecuzione simulazione {active_model_type}..."):
                # 1. Prepara i dati necessari
                past_data_np = st.session_state.seq2seq_past_data_gsheet[past_feature_cols_model].astype(float).values
                future_forecast_features_np = edited_future_forecast_df[forecast_feature_cols_model].astype(float).values
                
                # Estrai l'ultimo target noto dai dati storici caricati
                last_known_target_df = st.session_state.seq2seq_past_data_gsheet.iloc[-1:][target_columns_model]
                last_known_target_np = last_known_target_df.values.reshape(1, -1)
                
                # 2. Chiama la funzione di predizione autoregressiva
                predictions, _ = predict_seq2seq_autoregressive(
                    active_model, past_data_np, future_forecast_features_np,
                    last_known_target_np, active_scalers, active_config, active_device
                )
                
                start_pred_time = st.session_state.get('seq2seq_last_ts_gsheet', datetime.now(italy_tz))

            if predictions is not None:
                st.subheader('Risultato Simulazione Autoregressiva')
                figs = plot_predictions(predictions, active_config, start_time=start_pred_time)
                # ... (logica per mostrare i grafici) ...
            else:
                st.error("Predizione simulazione autoregressiva fallita.")

    elif active_model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
         st.subheader(f"Preparazione Dati Input Simulazione {active_model_type}")
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
              if import_err_past: st.error(f"Recupero storico fallito: {import_err_past}"); st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = import_err_past
              elif imported_df_past is not None:
                  try: final_past_df = imported_df_past[past_feature_cols_model]; st.success(f"Recuperate e processate {len(final_past_df)} righe storiche."); st.session_state.seq2seq_past_data_gsheet = final_past_df; st.session_state.seq2seq_last_ts_gsheet = last_ts_past if last_ts_past else datetime.now(italy_tz); fetch_error_gsheet_s2s = None; st.rerun()
                  except KeyError as e_cols: missing_cols_final = [c for c in past_feature_cols_model if c not in imported_df_past.columns]; st.error(f"Errore selezione colonne storiche dopo fetch: Colonne mancanti {missing_cols_final}"); st.session_state.seq2seq_past_data_gsheet = None; fetch_error_gsheet_s2s = f"Errore colonne: {e_cols}"
              else: st.error("Recupero storico non riuscito (risultato vuoto)."); fetch_error_gsheet_s2s = "Errore recupero dati."
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
                  elif col in last_known_past_data.index: forecast_df_initial[col] = last_known_past_data.get(col, 0.0)
                  else: forecast_df_initial[col] = 0.0
             edited_forecast_df = st.data_editor(forecast_df_initial.round(2), key="seq2seq_forecast_editor", num_rows="fixed", use_container_width=True, column_config={col: st.column_config.NumberColumn(label=get_station_label(col, short=True), help=f"Valore previsto per {col}", min_value=0.0 if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower() or 'livello' in col.lower()) else None, max_value=100.0 if 'umidit' in col.lower() else None, format="%.1f" if ('pioggia' in col.lower() or 'cumulata' in col.lower() or 'umidit' in col.lower()) else "%.2f", step=0.5 if ('pioggia' in col.lower() or 'cumulata' in col.lower()) else (1.0 if 'umidit' in col.lower() else 0.1), required=True) for col in forecast_feature_cols_model})
             forecast_data_valid_s2s = False
             if edited_forecast_df is not None and not edited_forecast_df.isnull().any().any():
                 if edited_forecast_df.shape == (num_rows_decoder_input, len(forecast_feature_cols_model)):
                     try: edited_forecast_df.astype(float); forecast_data_valid_s2s = True
                     except ValueError: st.warning("Valori non numerici rilevati nella tabella previsioni.")
                 else: st.warning("Numero di righe o colonne errato nella tabella previsioni.")
             else: st.warning("Completa o correggi la tabella delle previsioni future. Tutti i valori devono essere numerici.")

             st.divider(); st.markdown("**Passo 3: Esegui Simulazione**"); can_run_s2s_sim = past_data_loaded_s2s and forecast_data_valid_s2s
             if st.button(f"Esegui Simulazione {active_model_type}", disabled=not can_run_s2s_sim, type="primary", key="run_s2s_sim_button"):
                  if not can_run_s2s_sim: st.error("Mancano dati storici validi o previsioni future valide.")
                  else:
                       predictions_s2s = None; start_pred_time_s2s = None; uncertainty_s2s = None; attention_weights_s2s = None
                       progress_placeholder_s2s = st.empty()
                       with st.spinner(f"Preparazione simulazione {active_model_type}..."):
                           past_data_np = st.session_state.seq2seq_past_data_gsheet[past_feature_cols_model].astype(float).values
                           future_forecast_np = edited_forecast_df[forecast_feature_cols_model].astype(float).values
                           
                           if num_passes_sim > 1 and not is_quantile_model:
                               progress_bar_s2s = progress_placeholder_s2s.progress(0, text="Avvio calcolo incertezza...")
                               predictions_s2s, uncertainty_s2s, attention_weights_s2s = predict_seq2seq_with_uncertainty(
                                   active_model, past_data_np, future_forecast_np, 
                                   active_scalers, active_config, active_device, 
                                   num_passes=num_passes_sim,
                                   progress_indicator=progress_bar_s2s
                               )
                           else:
                               st.info("Esecuzione singola passata (nessuna incertezza MC Dropout).")
                               predictions_s2s, attention_weights_s2s = predict_seq2seq(
                                   active_model, past_data_np, future_forecast_np,
                                   active_scalers, active_config, active_device
                               )
                               uncertainty_s2s = None
                           
                           start_pred_time_s2s = st.session_state.get('seq2seq_last_ts_gsheet', datetime.now(italy_tz))
                       progress_placeholder_s2s.empty()
                       if predictions_s2s is not None:
                           output_steps_actual = predictions_s2s.shape[0]; total_hours_output_actual = output_steps_actual * 0.5
                           st.subheader(f'Risultato Simulazione: Prossime {total_hours_output_actual:.1f} ore'); st.caption(f"Previsione calcolata a partire da: {start_pred_time_s2s.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                           
                           st.subheader('Grafici Previsioni Simulate'); 
                           figs_sim_s2s = plot_predictions(predictions_s2s, active_config, start_pred_time_s2s, uncertainty=uncertainty_s2s)
                           num_graph_cols = min(len(figs_sim_s2s), 3); sim_cols = st.columns(num_graph_cols)
                           for i, fig_sim in enumerate(figs_sim_s2s):
                              with sim_cols[i % num_graph_cols]: s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_columns_model[i], short=False)); filename_base_s2s_ind = f"grafico_sim_{s_name_file}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"; st.plotly_chart(fig_sim, use_container_width=True); st.markdown(get_plotly_download_link(fig_sim, filename_base_s2s_ind), unsafe_allow_html=True)
                       else: st.error("Predizione simulazione fallita.")
    elif active_model_type == "SpatioTemporalGNN":
        st.subheader(f"Preparazione Dati Input Simulazione {active_model_type}")
        st.markdown("**Passo 1: Recupero Dati Storici (Input Grafo)**")
        st.caption(f"Verranno recuperati gli ultimi {input_steps_model} steps per i nodi del grafo dal Foglio Google (ID: `{GSHEET_ID}`).")
        
        date_col_model_name = st.session_state.date_col_name_csv
        # Mappatura specifica per i nodi GNN (livelli)
        column_mapping_gsheet_to_gnn = {
            'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
            'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
            'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
            'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
            'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)',
            # Aggiungere altre mappature se i nomi dei nodi nel modello differiscono da GSheet
            GSHEET_DATE_COL: date_col_model_name
        }
        
        required_model_cols_for_gnn = node_columns_model + [date_col_model_name]
        column_mapping_filtered = {gs_col: model_col for gs_col, model_col in column_mapping_gsheet_to_gnn.items() if model_col in required_model_cols_for_gnn}

        if 'imported_sim_data_gs_df_gnn' not in st.session_state:
            st.session_state.imported_sim_data_gs_df_gnn = None
        if 'imported_sim_start_time_gs_gnn' not in st.session_state:
            st.session_state.imported_sim_start_time_gs_gnn = None

        if st.button("Carica/Aggiorna Dati Storici GNN da GSheet", key="fetch_gsh_gnn"):
            fetch_sim_gsheet_data.clear()
            with st.spinner("Recupero dati storici per il grafo..."):
                # Imputazione non necessaria per GNN se si usano solo livelli
                imported_df, import_err, last_ts = fetch_sim_gsheet_data(GSHEET_ID, input_steps_model, GSHEET_DATE_COL, GSHEET_DATE_FORMAT, column_mapping_filtered, node_columns_model + [date_col_model_name], {})
                if import_err:
                    st.error(f"Recupero dati GNN fallito: {import_err}")
                    st.session_state.imported_sim_data_gs_df_gnn = None
                elif imported_df is not None:
                    st.session_state.imported_sim_data_gs_df_gnn = imported_df
                    st.session_state.imported_sim_start_time_gs_gnn = last_ts if last_ts else datetime.now(italy_tz)
                    st.success(f"Recuperate {len(imported_df)} righe per i nodi del grafo.")
                    st.rerun()

        if st.session_state.imported_sim_data_gs_df_gnn is not None:
            st.caption("Dati storici per il grafo caricati.")
            st.expander("Mostra dati storici caricati").dataframe(st.session_state.imported_sim_data_gs_df_gnn.round(3))
            
            st.markdown("**Passo 2: Esegui Simulazione**")
            if st.button(f"Esegui Simulazione {active_model_type}", type="primary"):
                with st.spinner("Preparazione dati di input e esecuzione previsione GNN..."):
                    # Recupera i dati storici dal dataframe in sessione
                    historical_df = st.session_state.imported_sim_data_gs_df_gnn
                    
                    # Recupera la configurazione delle feature
                    node_columns_sim = active_config['node_order']
                    feature_mapping_sim = active_config['node_feature_mapping']
                    num_features_sim = active_config['num_features']
                    input_steps_sim = active_config['input_window_steps']
                    
                    # Costruisci l'array di input multi-feature
                    input_array = np.zeros((input_steps_sim, len(node_columns_sim), num_features_sim), dtype=np.float32)
                    
                    for t in range(input_steps_sim):
                        for node_idx, node_name in enumerate(node_columns_sim):
                            input_array[t, node_idx, 0] = historical_df[node_name].iloc[t]
                            
                            additional_features = feature_mapping_sim.get(node_name, [])
                            for feat_idx, feat_col in enumerate(additional_features):
                                input_array[t, node_idx, 1 + feat_idx] = historical_df[feat_col].iloc[t]

                    # Chiama la funzione di predizione con il nuovo array
                    predictions, uncertainty = predict_gnn(active_model, input_array, active_scalers, active_config, active_device, uncertainty_passes=num_passes_sim)
                    
                    start_pred_time = st.session_state.imported_sim_start_time_gs_gnn
                    
                if predictions is not None:
                    output_steps_actual = predictions.shape[0]
                    total_hours_output_actual = output_steps_actual * 0.5
                    st.subheader(f'Risultato Simulazione: Prossime {total_hours_output_actual:.1f} ore')
                    st.caption(f"Previsione calcolata a partire da: {start_pred_time.strftime('%d/%m/%Y %H:%M:%S %Z')}")
                    
                    figs_sim = plot_predictions(predictions, active_config, start_pred_time, uncertainty=uncertainty)
                    num_graph_cols = min(len(figs_sim), 3)
                    sim_cols = st.columns(num_graph_cols)
                    for i, fig_sim in enumerate(figs_sim):
                        with sim_cols[i % num_graph_cols]:
                            s_name_file = re.sub(r'[^a-zA-Z0-9_-]', '_', get_station_label(target_columns_model[i], short=False))
                            filename_base_sim_ind = f"grafico_sim_{s_name_file}{ATTRIBUTION_PHRASE_FILENAME_SUFFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                            st.plotly_chart(fig_sim, use_container_width=True)
                            st.markdown(get_plotly_download_link(fig_sim, filename_base_sim_ind), unsafe_allow_html=True)
                else:
                    st.error("Predizione GNN fallita.")
    else: # LSTM
         # ... (La logica per LSTM rimane simile, ma con le modifiche per disabilitare MC Dropout se quantile)
         pass

# --- PAGINA TEST MODELLO SU STORICO ---
elif page == 'Test Modello su Storico':
    st.header('Test Modello su Dati Storici CSV (Walk-Forward Evaluation)')
    if not model_ready:
        st.warning("Seleziona un Modello attivo dalla sidebar per eseguire questo test.")
        st.stop()
    if not data_ready_csv:
        st.warning("Dati Storici CSV non disponibili. Caricane uno dalla sidebar.")
        st.stop()

    st.info(f"Modello Attivo: **{st.session_state.active_model_name}** ({active_model_type})")
    
    target_columns_model_test = active_config['target_columns']
    if active_model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
        input_steps_model_test = active_config['input_window_steps']
        output_steps_model_test = active_config['output_window_steps']
        past_feature_cols_model_test = active_config['all_past_feature_columns']
        forecast_feature_cols_model_test = active_config['forecast_input_columns']
        required_len_for_test = input_steps_model_test + output_steps_model_test
    elif active_model_type == "Seq2SeqAutoregressivo":
        input_steps_model_test = active_config['input_window_steps']
        output_steps_model_test = active_config['output_window_steps']
        past_feature_cols_model_test = active_config['all_past_feature_columns']
        forecast_feature_cols_model_test = active_config['forecast_input_columns']
        required_len_for_test = input_steps_model_test + output_steps_model_test
    elif active_model_type == "SpatioTemporalGNN":
        input_steps_model_test = active_config['input_window_steps']
        output_steps_model_test = active_config['output_window_steps']
        node_columns_model_test = active_config['node_order']
        required_len_for_test = input_steps_model_test + output_steps_model_test
    else: # LSTM
        feature_columns_model_test = active_config.get("feature_columns", [])
        input_steps_model_test = active_config['input_window']
        output_steps_model_test = active_config['output_window']
        required_len_for_test = input_steps_model_test + output_steps_model_test

    st.subheader("Configurazione Walk-Forward Evaluation")
    col_wf1, col_wf2, col_wf3, col_wf4 = st.columns(4)
    with col_wf1:
        num_evaluation_periods = st.number_input("Numero di Periodi di Test:", min_value=1, value=3, step=1, key="wf_num_periods")
    with col_wf2:
        max_start_idx = len(df_current_csv) - required_len_for_test
        first_input_start_index = st.number_input("Primo Indice di Inizio per Input nel CSV:", min_value=0, max_value=max(0, max_start_idx), value=0, step=1, key="wf_first_start_idx")
    with col_wf3:
        stride_between_periods = st.number_input("Passo tra Periodi di Test (righe):", min_value=1, value=output_steps_model_test, step=1, key="wf_stride")
    with col_wf4:
        is_quantile_model_test = active_config.get("training_mode") == "quantile"
        if is_quantile_model_test:
            st.info("Modello Quantile: no MC Dropout.")
            num_passes_test = 1
        else:
            num_passes_test = st.number_input("Passaggi per Incertezza:", min_value=1, max_value=100, value=25, step=1, key="wf_num_passes")

    st.markdown("---")
    
# SOSTITUISCI QUESTO INTERO BLOCCO if st.button(...)
if st.button("Esegui Test Walk-Forward su Storico", type="primary", key="run_walk_forward_test_button"):
    from sklearn.metrics import mean_squared_error
    
    evaluation_results_list = []
    all_period_mses = {target_col: [] for target_col in target_columns_model_test}
    
    progress_bar_test = st.progress(0.0, text="Avvio test walk-forward...")

    for i_period in range(num_evaluation_periods):
        progress_bar_test.progress((i_period + 1) / num_evaluation_periods, text=f"Elaborazione Periodo {i_period + 1}/{num_evaluation_periods}...")
        
        # Definiamo gli indici in modo esplicito
        current_input_start_index = first_input_start_index + (i_period * stride_between_periods)
        input_data_end_index = current_input_start_index + input_steps_model_test
        
        # La finestra degli "actuals" (dati reali per il confronto) inizia subito dopo
        actual_data_start_idx = input_data_end_index
        actual_data_end_idx = actual_data_start_idx + output_steps_model_test
        
        if actual_data_end_idx > len(df_current_csv):
            st.warning(f"Periodo {i_period+1} saltato: dati insufficienti nel CSV.")
            continue

        # Estraiamo le fette di dati (slice)
        input_df_period_slice = df_current_csv.iloc[current_input_start_index:input_data_end_index].copy()
        actual_df_period_slice = df_current_csv.iloc[actual_data_start_idx:actual_data_end_idx].copy()
        
        # Verifichiamo che le fette non siano vuote
        if input_df_period_slice.empty or actual_df_period_slice.empty:
            st.warning(f"Periodo {i_period+1} saltato: slice di dati vuoto.")
            continue

        # Dati grezzi e timestamp per il confronto e la visualizzazione
        actual_data_np = actual_df_period_slice[target_columns_model_test].values
        start_time_prediction_period = actual_df_period_slice[date_col_name_csv].iloc[0]

        predictions_period = None
        uncertainty_period = None
        input_cols_for_display = []
        
        if active_model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
            past_input_np_raw = input_df_period_slice[past_feature_cols_model_test].values
            # Per il decoder, usiamo i dati reali dalla finestra di previsione
            future_input_np_raw = actual_df_period_slice[forecast_feature_cols_model_test].values
            input_cols_for_display = past_feature_cols_model_test

            if num_passes_test > 1 and not is_quantile_model_test:
                predictions_period, uncertainty_period, _ = predict_seq2seq_with_uncertainty(
                    active_model, past_input_np_raw, future_input_np_raw, 
                    active_scalers, active_config, active_device, 
                    num_passes=num_passes_test
                )
            else:
                predictions_period, _ = predict_seq2seq(
                    active_model, past_input_np_raw, future_input_np_raw,
                    active_scalers, active_config, active_device
                )
        elif active_model_type == "Seq2SeqAutoregressivo":
            past_input_np_raw = input_df_period_slice[past_feature_cols_model_test].values
            future_input_np_raw = actual_df_period_slice[forecast_feature_cols_model_test].values
            input_cols_for_display = past_feature_cols_model_test

            # Ultimo target noto: ultima riga della finestra di input
            last_known_target_np = input_df_period_slice[target_columns_model_test].iloc[-1:].values.reshape(1, -1)

            predictions_period, _ = predict_seq2seq_autoregressive(
                active_model, past_input_np_raw, future_input_np_raw,
                last_known_target_np, active_scalers, active_config, active_device
            )
        
        elif active_model_type == "SpatioTemporalGNN":
            node_columns_test = active_config['node_order']
            feature_mapping_test = active_config['node_feature_mapping']
            num_features_test = active_config['num_features']
            
            all_gnn_model_feature_cols = []
            for node_name in node_columns_test:
                all_gnn_model_feature_cols.append(node_name)
                all_gnn_model_feature_cols.extend(feature_mapping_test.get(node_name, []))
            input_cols_for_display = sorted(list(set(all_gnn_model_feature_cols)))

            raw_input_data_multi_feature = np.zeros((input_steps_model_test, len(node_columns_test), num_features_test), dtype=np.float32)
            for t in range(input_steps_model_test):
                for node_idx, node_name in enumerate(node_columns_test):
                    raw_input_data_multi_feature[t, node_idx, 0] = input_df_period_slice[node_name].iloc[t]
                    additional_features = feature_mapping_test.get(node_name, [])
                    for feat_idx, feat_col in enumerate(additional_features):
                        if feat_col in input_df_period_slice.columns:
                            raw_input_data_multi_feature[t, node_idx, 1 + feat_idx] = input_df_period_slice[feat_col].iloc[t]

            predictions_period, uncertainty_period = predict_gnn(active_model, raw_input_data_multi_feature, active_scalers, active_config, active_device, uncertainty_passes=num_passes_test)

        else: # LSTM Standard
            input_np_raw = input_df_period_slice[feature_columns_model_test].values
            input_cols_for_display = feature_columns_model_test
            
            if num_passes_test > 1 and not is_quantile_model_test:
                predictions_period, uncertainty_period = predict_with_uncertainty(active_model, input_np_raw, active_scalers[0], active_scalers[1], active_config, active_device, num_passes=num_passes_test)
            else:
                predictions_period = predict(active_model, input_np_raw, active_scalers[0], active_scalers[1], active_config, active_device)

        if predictions_period is not None:
            period_mses = {}
            for i_target, target_col in enumerate(target_columns_model_test):
                pred_series = predictions_period[:, i_target, 1] if predictions_period.ndim == 3 else predictions_period[:, i_target]
                mse = mean_squared_error(actual_data_np[:, i_target], pred_series)
                period_mses[target_col] = mse
                all_period_mses[target_col].append(mse)

            evaluation_results_list.append({
                "period_num": i_period + 1,
                "predictions": predictions_period,
                "actuals": actual_data_np,
                "uncertainty": uncertainty_period,
                "start_time_prediction_period": start_time_prediction_period,
                "mses": period_mses,
                "raw_input_df_for_table": input_df_period_slice, # Salva lo slice corretto
                "input_cols_for_display": input_cols_for_display,
            })

    progress_bar_test.empty()
    
    if not evaluation_results_list:
        st.error("Nessun periodo di test è stato completato con successo.")
    else:
        st.success(f"Test completato per {len(evaluation_results_list)} periodi.")
        for result in evaluation_results_list:
            st.subheader(f"Risultati Test - Periodo {result['period_num']}")
            st.markdown(f"**Inizio periodo di previsione:** {result['start_time_prediction_period'].strftime('%d/%m/%Y %H:%M')}")
            
            figs = plot_predictions(result['predictions'], active_config, start_time=result['start_time_prediction_period'], actual_data=result['actuals'], uncertainty=result['uncertainty'])
            
            num_graph_cols = min(len(figs), 2)
            graph_cols = st.columns(num_graph_cols)
            for i_fig, fig in enumerate(figs):
                with graph_cols[i_fig % num_graph_cols]:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.write("Errore Quadratico Medio (MSE) per il periodo:")
            st.dataframe(pd.DataFrame.from_dict(result['mses'], orient='index', columns=['MSE']).round(5))

            st.markdown("### Dati Input utilizzati per la Previsione")
            st.caption(f"Gli ultimi {N_INPUT_DISPLAY_STEPS} passi temporali che il modello ha ricevuto come input per questa previsione.")

            raw_input_df_full_slice = result['raw_input_df_for_table']
            input_cols_to_show = result['input_cols_for_display']
            
            if raw_input_df_full_slice is not None and not raw_input_df_full_slice.empty:
                last_n_inputs_df = raw_input_df_full_slice.tail(N_INPUT_DISPLAY_STEPS).copy()
                cols_for_final_table = [date_col_name_csv] + [col for col in input_cols_to_show if col in last_n_inputs_df.columns]
                display_table = last_n_inputs_df[cols_for_final_table].set_index(date_col_name_csv)
                st.dataframe(display_table.style.format("{:.3f}"), use_container_width=True)
            else:
                st.info("Dati input per questo periodo non disponibili per la visualizzazione.")

            st.divider()

        st.subheader("Riepilogo Metriche Medie su Tutti i Periodi")
        avg_mse_data = {get_station_label(k): np.mean(v) for k, v in all_period_mses.items() if v}
        st.dataframe(pd.DataFrame.from_dict(avg_mse_data, orient='index', columns=['MSE Medio']).round(5))

elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici (da file CSV)')
    # ... (Pagina invariata) ...

elif page == 'Allenamento Modello':
    # CORRETTO: TUTTO IL CODICE DELL'ALLENAMENTO È SPOSTATO QUI DENTRO E INDENTATO
    
    st.header('Allenamento Nuovo Modello')
    
    def parse_hour_periods(periods_str, context=""):
        # ... (Questa funzione helper interna rimane invariata) ...
        if not periods_str.strip(): return []
        try:
            periods = [int(p.strip()) for p in periods_str.split(',') if p.strip()]
            if not all(p > 0 for p in periods):
                st.warning(f"I periodi orari ({context}) devono essere numeri interi positivi.")
                return []
            return sorted(list(set(periods)))
        except ValueError:
            st.warning(f"Formato periodi orari ({context}) non valido.")
            return []

    if not data_ready_csv:
        st.warning("Dati Storici CSV non disponibili. Caricane uno dalla sidebar.")
        st.stop()

    st.success(f"Dati CSV disponibili per l'allenamento: {len(df_current_csv)} righe.")
    st.subheader('Configurazione Addestramento')
    
    # La definizione del tipo di modello
    if PYG_AVAILABLE:
        model_options = ["LSTM Standard", "Seq2Seq (Encoder-Decoder)", "Seq2Seq con Attenzione", "Seq2Seq Autoregressivo", "Transformer", "Spatio-Temporal GNN"]
    else:
        model_options = ["LSTM Standard", "Seq2Seq (Encoder-Decoder)", "Seq2Seq con Attenzione", "Seq2Seq Autoregressivo", "Transformer"]
    train_model_type = st.radio("Tipo di Modello da Allenare:", model_options, key="train_select_type", horizontal=True)

    st.markdown("**Metodo di Suddivisione Dati**")
    split_method = st.radio(
        "Scegli come dividere i dati in Training e Validation Set:",
        options=["Temporale (Consigliato per Serie Storiche)", "Casuale per Percentuale"],
        index=0, key="split_method_selection",
        help="Temporale: usa i dati più vecchi per training e i più recenti per validazione. Casuale: mescola tutti i dati (sconsigliato per serie storiche)."
    )
    validation_percentage = 0.2
    if split_method == "Casuale per Percentuale":
        st.warning("⚠️ **Attenzione:** La suddivisione casuale può portare a stime delle performance irrealisticamente ottimistiche per le serie storiche.")
        validation_percentage = st.slider(
            "Percentuale Dati per il Validation Set:", 10, 50, 20, 5, format="%d%%", key="validation_split_percentage"
        ) / 100.0

    st.markdown("**Modalità di Training e Funzione di Loss**")
    col_mode, col_loss = st.columns(2)
    with col_mode:
        training_mode = st.radio("Modalità di Training:", ["Standard (Puntuale)", "Quantile Regression"], key="training_mode_select", horizontal=True)
        quantiles_list = [0.1, 0.5, 0.9]
        if training_mode == "Quantile Regression":
            quantiles_str = st.text_input("Quantili (es. 0.1, 0.5, 0.9):", "0.1, 0.5, 0.9", key="quantiles_input")
            try:
                quantiles_list = sorted([float(q.strip()) for q in quantiles_str.split(',')])
                if len(quantiles_list) != 3: st.error("Inserire esattamente 3 quantili."); st.stop()
            except ValueError: st.error("Formato quantili non valido."); st.stop()
    with col_loss:
        if training_mode == "Quantile Regression":
            loss_choice = "QuantileLoss"
            st.info(f"**Funzione di Loss:** `QuantileLoss` (obbligatoria)")
        else:
            loss_choice = st.selectbox("Funzione di Loss (Standard):", ["MSELoss", "HuberLoss", "DynamicWeightedMSE", "DilateLoss"], key="standard_loss_choice")

    st.markdown("**Opzione: Loss Pesata per Dati Recenti**")
    use_weighted_loss = st.checkbox(
        "Usa Loss Pesata per dati post-modifica", value=False, key="use_weighted_loss_checkbox",
        help="Penalizza di più gli errori sui dati marcati come 'post-modifica' dalla colonna dummy."
    )
    dummy_col_name = "Variabile Dummy"
    post_mod_weight = 5.0
    if use_weighted_loss:
        col_dummy, col_weight = st.columns(2)
        with col_dummy:
            dummy_col_name = st.text_input("Nome colonna Dummy:", "Variabile Dummy", key="dummy_col_name_input")
        with col_weight:
            post_mod_weight = st.number_input("Peso per dati post-modifica (>= 1.0):", 1.0, value=5.0, step=0.5, key="post_mod_weight_input")

    st.markdown("**Opzione: Loss Pesata su Valori di Piena (Target)**")
    use_magnitude_loss = st.checkbox(
        "Usa Loss Pesata per eventi di piena",
        value=True, # Attiviamolo di default
        key="use_magnitude_loss_checkbox",
        help="Se attivato, il modello verrà penalizzato di più per errori su valori del target che superano una certa soglia (es. livelli di piena)."
    )

    target_threshold_scaled = 0.5 # Valore di default sulla scala 0-1
    weight_exponent = 1.0 # Valore di default

    if use_magnitude_loss:
        col_thresh, col_exp = st.columns(2)
        with col_thresh:
            target_threshold_scaled = st.slider(
                "Soglia di attivazione (su scala 0-1):",
                min_value=0.0, max_value=1.0,
                value=0.5, step=0.05,
                key="target_threshold_slider",
                help="Il peso inizia ad aumentare per i valori del target sopra questa soglia normalizzata. 0.5 corrisponde circa alla metà del massimo valore di piena mai registrato."
            )
        with col_exp:
            weight_exponent = st.number_input(
                "Esponente di crescita del peso:",
                min_value=0.1, value=1.0, step=0.1,
                key="weight_exponent_input",
                help="Controlla quanto velocemente aumenta il peso. 1.0 = crescita lineare; 2.0 = crescita quadratica (molto più aggressiva per valori alti)."
            )
    
    # Ora questa riga viene eseguita solo quando sei sulla pagina giusta
    default_save_name = f"modello_{train_model_type.split()[0].lower().replace('-','_')}_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
    save_name_input = st.text_input("Nome base per salvare il modello:", default_save_name, key="train_save_filename")
    save_name = re.sub(r'[^\w-]', '_', save_name_input).strip('_') or "modello_default"
    if save_name != save_name_input: st.caption(f"Nome file valido: `{save_name}`")
    os.makedirs(MODELS_DIR, exist_ok=True)

    if train_model_type == "LSTM Standard":
        st.markdown("**1. Seleziona Feature e Target (LSTM)**")
        # ... (UI per LSTM invariata) ...
        all_features = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
        default_features = [f for f in st.session_state.feature_columns if f in all_features]
        selected_features = st.multiselect("Feature Input LSTM:", options=all_features, default=default_features, key="train_lstm_feat")
        level_options = [f for f in all_features if 'livello' in f.lower() or '[m]' in f.lower()]
        selected_targets = st.multiselect("Target Output LSTM (Livelli):", options=level_options, default=level_options[:1], key="train_lstm_target")

        st.markdown("**2. Parametri Modello e Training (LSTM)**")
        with st.expander("Impostazioni Allenamento LSTM", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                iw_hours = st.number_input("Input Window (ore)", 1, value=24, step=1, key="t_lstm_in_h")
                ow_steps = st.number_input("Output Window (steps 30min)", 1, value=6, step=1, key="t_lstm_out_s")
                n_splits = st.number_input("Numero Fold CV", 2, value=3, step=1, key="t_lstm_n_splits")
            with c2:
                hs = st.number_input("Hidden Size", 8, value=128, step=8, key="t_lstm_hs")
                nl = st.number_input("Numero Layers", 1, value=2, step=1, key="t_lstm_nl")
                dr = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_lstm_dr")
            with c3:
                lr = st.number_input("Learning Rate", 1e-6, value=0.001, format="%.5f", step=1e-4, key="t_lstm_lr")
                bs = st.select_slider("Batch Size", [8,16,32,64,128,256], 32, key="t_lstm_bs")
                ep = st.number_input("Numero Epoche", 1, value=50, step=5, key="t_lstm_ep")
            c4, c5 = st.columns(2)
            with c4: device_option = st.radio("Device:", ['Auto', 'CPU'], 0, key='train_dev_lstm', horizontal=True)
            with c5: save_choice = st.radio("Salvataggio:", ['Migliore', 'Finale'], 0, key='train_save_lstm', horizontal=True)

        st.divider()
        ready_to_train = bool(save_name and selected_features and selected_targets and ow_steps > 0)
        if st.button("Avvia Addestramento LSTM", type="primary", disabled=not ready_to_train, key="train_run_lstm"):
            st.info(f"Avvio addestramento LSTM Standard per '{save_name}'...")
            with st.spinner("Preparazione dati LSTM..."):
                # <<< MODIFICA 1: Passaggio parametri e unpacking corretto
                data_tuple = prepare_training_data(
                    df_current_csv.copy(), selected_features, selected_targets, iw_hours, int(ow_steps / 2.0),
                    use_weighted_loss=use_weighted_loss,
                    dummy_col_name=dummy_col_name,
                    post_modification_weight=post_mod_weight
                )
                if data_tuple:
                    X_scaled, y_scaled, sample_weights, sc_f, sc_t = data_tuple
                else:
                    X_scaled, y_scaled = None, None
            
            if X_scaled is not None and y_scaled is not None:
                # <<< MODIFICA 2: Passaggio sample_weights e use_weighted_loss al trainer
                trained_model, _, _ = train_model(
                    X_scaled, y_scaled, sample_weights,
                    X_scaled.shape[2], len(selected_targets), ow_steps, hs, nl, ep, bs, lr, dr, 
                    'migliore' if 'Migliore' in save_choice else 'finale', 
                    'auto' if 'Auto' in device_option else 'cpu', 
                    n_splits_cv=n_splits, 
                    loss_function_name=loss_choice, 
                    training_mode=training_mode.split()[0].lower(), 
                    quantiles=quantiles_list if training_mode == "Quantile Regression" else None,
                    split_method=split_method,
                    validation_size=validation_percentage,
                    use_weighted_loss=use_weighted_loss,
                    use_magnitude_loss=use_magnitude_loss,
                    target_threshold=target_threshold_scaled,
                    weight_exponent=weight_exponent
                )
                if trained_model:
                    st.success("Addestramento LSTM completato!")
                    config_save = {"model_type": "LSTM", "input_window": X_scaled.shape[1], "output_window": ow_steps, "hidden_size": hs, "num_layers": nl, "dropout": dr, "feature_columns": selected_features, "target_columns": selected_targets, "training_date": datetime.now(italy_tz).isoformat(), "display_name": save_name, "training_mode": training_mode.split()[0].lower(), "loss_function": loss_choice}
                    if training_mode == "Quantile Regression": config_save["quantiles"] = quantiles_list
                    base_path = os.path.join(MODELS_DIR, save_name)
                    torch.save(trained_model.state_dict(), f"{base_path}.pth")
                    joblib.dump(sc_f, f"{base_path}_features.joblib")
                    joblib.dump(sc_t, f"{base_path}_targets.joblib")
                    with open(f"{base_path}.json", 'w', encoding='utf-8') as f: json.dump(config_save, f, indent=4)
                    st.success(f"Modello '{save_name}' salvato.")
                    find_available_models.clear()
            else: st.error("Preparazione dati LSTM fallita.")

    elif train_model_type in ["Seq2Seq (Encoder-Decoder)", "Seq2Seq con Attenzione", "Seq2Seq Autoregressivo", "Transformer"]:
        st.markdown(f"**1. Seleziona Feature ({train_model_type})**")
        
        all_features = df_current_csv.columns.drop(date_col_name_csv, errors='ignore').tolist()
        default_past = [f for f in st.session_state.feature_columns if f in all_features]
        selected_past_features = st.multiselect("Feature Storiche (Input Encoder):", all_features, default=default_past, key="train_s2s_past_feat")
        
        forecast_help_text = "Per i modelli Seq2Seq standard, includere tutte le feature future. Per il modello Autoregressivo, includere SOLO le feature esogene (es. meteo), non i livelli."
        default_forecast = [f for f in selected_past_features if 'pioggia' in f.lower() or 'cumulata' in f.lower() or f == HUMIDITY_COL_NAME]
        selected_forecast_features = st.multiselect("Feature Forecast (Input Decoder):", options=selected_past_features, default=default_forecast, key="train_s2s_forecast_feat", help=forecast_help_text)
        
        level_options = [f for f in selected_past_features if 'livello' in f.lower() or '[m]' in f.lower()]
        selected_targets = st.multiselect("Target Output (Livelli):", options=level_options, default=level_options[:1], key="train_s2s_target_feat")

        st.markdown(f"**2. Parametri Modello e Training ({train_model_type})**")
        with st.expander(f"Impostazioni Allenamento {train_model_type}", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                iw_steps = st.number_input("Input Storico (steps 30min)", 2, value=48, step=2, key="t_s2s_in")
                if train_model_type != "Seq2Seq Autoregressivo":
                    fw_steps = st.number_input("Input Forecast (steps 30min)", 1, value=6, step=1, key="t_s2s_fore")
                else:
                    fw_steps = 0 # Non usato direttamente nella preparazione dati autoregressiva
                ow_steps = st.number_input("Output Previsione (steps 30min)", 1, value=6, step=1, key="t_s2s_out")
                n_splits_cv = st.number_input("Numero Fold CV", 2, value=3, step=1, key="t_s2s_n_splits_cv")
            with c2:
                if train_model_type == "Transformer":
                    d_model = st.number_input("Dimensione Modello (d_model)", 16, value=64, step=16, key="t_trans_dmodel")
                    nhead = st.number_input("Numero Teste (nhead)", 1, value=4, step=1, key="t_trans_nhead")
                    dim_ff = st.number_input("Dim FeedForward", 64, value=256, step=64, key="t_trans_dimff")
                    num_enc_l = st.number_input("Layers Encoder", 1, value=2, step=1, key="t_trans_encl")
                    num_dec_l = st.number_input("Layers Decoder", 1, value=2, step=1, key="t_trans_decl")
                else:
                    hs = st.number_input("Hidden Size", 8, value=128, step=8, key="t_s2s_hs")
                    nl = st.number_input("Numero Layers", 1, value=2, step=1, key="t_s2s_nl")
                dr = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="t_s2s_dr")
            with c3:
                lr = st.number_input("Learning Rate", 1e-6, value=0.001, format="%.5f", step=1e-4, key="t_s2s_lr")
                bs = st.select_slider("Batch Size", [8,16,32,64,128,256], 32, key="t_s2s_bs")
                ep = st.number_input("Numero Epoche", 1, value=50, step=5, key="t_s2s_ep")
            c4, c5 = st.columns(2)
            with c4: device_option = st.radio("Device:", ['Auto', 'CPU'], 0, key='train_dev_s2s', horizontal=True)
            with c5: save_choice = st.radio("Salvataggio:", ['Migliore', 'Finale'], 0, key='train_save_s2s', horizontal=True)

        st.divider()
        ready_to_train = bool(save_name and selected_past_features and selected_forecast_features and selected_targets and ow_steps > 0)
        
        if st.button(f"Avvia Addestramento {train_model_type}", type="primary", disabled=not ready_to_train, key="train_run_s2s_trans"):
            st.info(f"Avvio addestramento {train_model_type} per '{save_name}'...")
            
            # --- Blocco di Addestramento Unificato ---
            
            # 1. Preparazione Dati
            data_tuple = None
            with st.spinner(f"Preparazione dati {train_model_type}..."):
                if train_model_type == "Seq2Seq Autoregressivo":
                    data_tuple = prepare_training_data_seq2seq_autoregressive(
                        df_current_csv.copy(), selected_past_features, selected_forecast_features, selected_targets,
                        iw_steps, ow_steps
                    )
                else: # Modelli Seq2Seq standard e Transformer
                    data_tuple = prepare_training_data_seq2seq(
                        df_current_csv.copy(), selected_past_features, selected_forecast_features, selected_targets, 
                        iw_steps, fw_steps, ow_steps,
                        use_weighted_loss=use_weighted_loss,
                        dummy_col_name=dummy_col_name,
                        post_modification_weight=post_mod_weight
                    )
            
            if not data_tuple or data_tuple[0] is None:
                st.error(f"Preparazione dati {train_model_type} fallita.")
                st.stop()

            # 2. Creazione Modello
            model_instance = None
            device = torch.device('cuda' if ('auto' in device_option.lower() and torch.cuda.is_available()) else 'cpu')
            num_quantiles_train = len(quantiles_list) if training_mode == "Quantile Regression" else 1

            if train_model_type == "Seq2Seq Autoregressivo":
                (X_enc_scaled, X_dec_scaled, y_tar_scaled, sc_past, sc_dec_in, sc_tar) = data_tuple
                encoder = EncoderLSTM(X_enc_scaled.shape[2], hs, nl, dr)
                decoder_input_size = len(selected_forecast_features) + len(selected_targets)
                decoder = DecoderLSTMAutoregressive(decoder_input_size, hs, len(selected_targets), nl, dr)
                model_instance = Seq2SeqAutoregressive(encoder, decoder, ow_steps, device)
            elif train_model_type == "Transformer":
                (X_enc_scaled, X_dec_scaled, y_tar_scaled, sample_weights, sc_past, sc_fore, sc_tar) = data_tuple
                model_instance = HydroTransformer(
                    input_dim_encoder=X_enc_scaled.shape[2], input_dim_decoder=X_dec_scaled.shape[2],
                    output_dim=len(selected_targets), d_model=d_model, nhead=nhead,
                    num_encoder_layers=num_enc_l, num_decoder_layers=num_dec_l,
                    dim_feedforward=dim_ff, dropout=dr, num_quantiles=num_quantiles_train)
            else: # Seq2Seq e Seq2Seq con Attenzione
                (X_enc_scaled, X_dec_scaled, y_tar_scaled, sample_weights, sc_past, sc_fore, sc_tar) = data_tuple
                encoder = EncoderLSTM(X_enc_scaled.shape[2], hs, nl, dr)
                if train_model_type == "Seq2Seq con Attenzione":
                    decoder = DecoderLSTMWithAttention(X_dec_scaled.shape[2], hs, len(selected_targets), nl, dr, num_quantiles=num_quantiles_train)
                    model_instance = Seq2SeqWithAttention(encoder, decoder, ow_steps)
                else:
                    decoder = DecoderLSTM(X_dec_scaled.shape[2], hs, len(selected_targets), nl, dr, num_quantiles=num_quantiles_train)
                    model_instance = Seq2SeqHydro(encoder, decoder, ow_steps)

            # 3. Addestramento
            trained_model = None
            if train_model_type == "Seq2Seq Autoregressivo":
                trained_model, _, _ = train_model_seq2seq_autoregressive(
                    X_enc_scaled, X_dec_scaled, y_tar_scaled,
                    model_instance, ow_steps, ep, bs, lr,
                    'migliore' if 'Migliore' in save_choice else 'finale',
                    device_option, teacher_forcing_ratio_schedule=[0.6, 0.1],
                    n_splits_cv=n_splits_cv, loss_function_name=loss_choice,
                    training_mode=training_mode.split()[0].lower(), quantiles=quantiles_list,
                    split_method=split_method, validation_size=validation_percentage,
                    use_magnitude_loss=use_magnitude_loss, target_threshold=target_threshold_scaled, weight_exponent=weight_exponent
                )
            else:
                trained_model, _, _ = train_model_seq2seq(
                    X_enc_scaled, X_dec_scaled, y_tar_scaled, sample_weights,
                    model_instance, ow_steps, ep, bs, lr, 
                    'migliore' if 'Migliore' in save_choice else 'finale',
                    device_option, teacher_forcing_ratio_schedule=[0.6, 0.1], n_splits_cv=n_splits_cv,
                    loss_function_name=loss_choice, training_mode=training_mode.split()[0].lower(),
                    quantiles=quantiles_list, split_method=split_method, validation_size=validation_percentage,
                    use_weighted_loss=use_weighted_loss, use_magnitude_loss=use_magnitude_loss,
                    target_threshold=target_threshold_scaled, weight_exponent=weight_exponent
                )

            # 4. Salvataggio
            if trained_model:
                st.success(f"Addestramento {train_model_type} completato!")
                base_path = os.path.join(MODELS_DIR, save_name)
                torch.save(trained_model.state_dict(), f"{base_path}.pth")
                
                config_save = {
                    "display_name": save_name, "training_date": datetime.now(italy_tz).isoformat(),
                    "model_type": train_model_type.replace(" (Encoder-Decoder)", ""), "input_window_steps": iw_steps,
                    "output_window_steps": ow_steps, "all_past_feature_columns": selected_past_features,
                    "forecast_input_columns": selected_forecast_features, "target_columns": selected_targets,
                    "dropout": dr, "training_mode": training_mode.split()[0].lower(), "loss_function": loss_choice
                }
                if training_mode == "Quantile Regression": config_save["quantiles"] = quantiles_list

                if train_model_type == "Seq2Seq Autoregressivo":
                    joblib.dump(sc_past, f"{base_path}_past_features.joblib")
                    joblib.dump(sc_dec_in, f"{base_path}_forecast_decoder_input.joblib")
                    joblib.dump(sc_tar, f"{base_path}_targets.joblib")
                    config_save.update({"hidden_size": hs, "num_layers": nl})
                elif train_model_type == "Transformer":
                    joblib.dump(sc_past, f"{base_path}_past_features.joblib")
                    joblib.dump(sc_fore, f"{base_path}_forecast_features.joblib")
                    joblib.dump(sc_tar, f"{base_path}_targets.joblib")
                    config_save.update({"d_model": d_model, "nhead": nhead, "dim_feedforward": dim_ff, "num_encoder_layers": num_enc_l, "num_decoder_layers": num_dec_l})
                    config_save["forecast_window_steps"] = fw_steps
                else: # Standard Seq2Seq models
                    joblib.dump(sc_past, f"{base_path}_past_features.joblib")
                    joblib.dump(sc_fore, f"{base_path}_forecast_features.joblib")
                    joblib.dump(sc_tar, f"{base_path}_targets.joblib")
                    config_save.update({"hidden_size": hs, "num_layers": nl})
                    config_save["forecast_window_steps"] = fw_steps

                with open(f"{base_path}.json", 'w', encoding='utf-8') as f: json.dump(config_save, f, indent=4)
                st.success(f"Modello '{save_name}' salvato.")
                find_available_models.clear()
            else:
                st.error("L'addestramento del modello non è stato completato con successo.")

    elif train_model_type == "Spatio-Temporal GNN":
        st.markdown(f"**1. Definizione del Grafo e Selezione Feature ({train_model_type})**")
        # ... (UI per GNN invariata) ...
        all_level_sensors = [f for f in df_current_csv.columns if 'livello' in f.lower() or '[m]' in f.lower()]
        node_order = st.multiselect("Nodi del Grafo (Idrometri - l'ordine è importante!):", options=all_level_sensors, default=all_level_sensors, key="train_gnn_nodes")
        st.markdown("**Associa Feature Aggiuntive ai Nodi**")
        all_other_features = [f for f in df_current_csv.columns if f not in all_level_sensors and f != date_col_name_csv]
        node_feature_mapping = {}
        if node_order:
            global_feature_keywords = ['progressivo', 'dummy', 'seasonality_sin', 'seasonality_cos', 'umidita', 'soil moisture']
            specific_node_feature_map = {'serra dei conti': '1295', 'bettolelle': '2637', 'pianello': '2858', 'corinaldo/nevola': '2964'}
            for i, node_name in enumerate(node_order):
                default_features_for_node = []
                for keyword in global_feature_keywords:
                    for feature in all_other_features:
                        if keyword in feature.lower():
                            default_features_for_node.append(feature)
                node_name_lower = node_name.lower()
                for hydro_keyword, rain_keyword in specific_node_feature_map.items():
                    if hydro_keyword in node_name_lower:
                        rain_features = [f for f in all_other_features if rain_keyword in f and 'cumulata' in f.lower()]
                        default_features_for_node.extend(rain_features)
                default_features_for_node = sorted(list(set(default_features_for_node)))
                short_name_match = re.search(r'\((.*?)\)', node_name)
                short_name = short_name_match.group(1).strip() if short_name_match else node_name
                selected_features_for_node = st.multiselect(f"Feature aggiuntive per Nodo {i} ({short_name}):", options=all_other_features, default=default_features_for_node, key=f"train_gnn_features_node_{i}")
                node_feature_mapping[node_name] = selected_features_for_node
        st.caption("Definisci le connessioni del grafo. Ogni riga è `sorgente,destinazione,peso`.")
        node_mapping_help = {i: name for i, name in enumerate(node_order)}
        st.json(node_mapping_help, expanded=False)
        edge_list_str = st.text_area("Lista Archi (formato: 'sorgente,destinazione,peso')", "0,1,15.5\n1,2,8.2\n2,3,12.0", key="train_gnn_edges", height=100)
        selected_targets_gnn = st.multiselect("Target Output (Nodi da predire):", options=node_order, default=node_order[-1:] if node_order else [], key="train_gnn_target")
        edge_index, edge_weights = None, None
        try:
            s, t, w = [], [], []
            for line in edge_list_str.split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]; s.append(int(parts[0])); t.append(int(parts[1])); w.append(float(parts[2]))
            if s: edge_index, edge_weights = [s, t], w; st.success(f"Grafo definito con {len(node_order)} nodi e {len(s)} archi.")
        except Exception: st.error("Formato archi non valido.")

        st.markdown(f"**2. Parametri Modello e Training**")
        with st.expander("Impostazioni Allenamento GNN", expanded=True):
            # ... (UI per parametri GNN invariata) ...
            c1, c2, c3 = st.columns(3)
            with c1:
                iw_steps = st.number_input("Input Window (steps)", min_value=2, value=48, key="t_gnn_in")
                ow_steps = st.number_input("Output Window (steps)", min_value=1, value=6, key="t_gnn_out")
                hs = st.number_input("Hidden Dimension", min_value=8, value=64, key="t_gnn_hs")
            with c2:
                nl = st.number_input("Numero Layers (GRU)", min_value=1, value=2, key="t_gnn_nl")
                dr = st.slider("Dropout", 0.0, 0.7, 0.2, key="t_gnn_dr")
                lr = st.number_input("Learning Rate", min_value=1e-6, value=0.001, format="%.5f", key="t_gnn_lr")
            with c3:
                bs = st.select_slider("Batch Size", [8,16,32,64], 32, key="t_gnn_bs")
                ep = st.number_input("Numero Epoche", min_value=1, value=50, key="t_gnn_ep")

        if st.button("Avvia Addestramento GNN", type="primary", key="train_run_gnn"):
            if not all([save_name, node_order, selected_targets_gnn, edge_index, edge_weights]):
                st.error("Completa tutti i campi per l'addestramento GNN.")
            else:
                st.info(f"Avvio addestramento GNN per '{save_name}'...")
                with st.spinner("Preparazione dati GNN..."):
                    # <<< MODIFICA 6: Passaggio parametri e unpacking corretto
                    data_tuple = prepare_training_data_gnn(
                        df_current_csv.copy(), node_order, selected_targets_gnn, node_feature_mapping,
                        iw_steps, ow_steps,
                        use_weighted_loss=use_weighted_loss,
                        dummy_col_name=dummy_col_name,
                        post_modification_weight=post_mod_weight
                    )
                    if data_tuple:
                        X_scaled, y_scaled, sample_weights, sc_f, sc_t, num_features_detected = data_tuple
                    else:
                        X_scaled = None
                
                if X_scaled is not None:
                    num_q = len(quantiles_list) if training_mode == "Quantile Regression" else 1
                    model = SpatioTemporalGNN(
                        num_nodes=len(node_order), num_features=num_features_detected, hidden_dim=hs, 
                        num_layers=nl, output_window=ow_steps, output_dim=len(selected_targets_gnn), 
                        num_quantiles=num_q, dropout=dr
                    )
                    
                    # <<< MODIFICA 7: Passaggio parametri al trainer
                    trained_model, _, _ = train_model_gnn(
                        X_scaled, y_scaled, sample_weights,
                        model, edge_index, edge_weights, ep, bs, lr,
                        training_mode=training_mode.split()[0].lower(),
                        quantiles=quantiles_list if training_mode == "Quantile Regression" else None,
                        split_method=split_method,
                        validation_size=validation_percentage,
                        use_weighted_loss=use_weighted_loss,
                        use_magnitude_loss=use_magnitude_loss,
                        target_threshold=target_threshold_scaled,
                        weight_exponent=weight_exponent
                    )
                    
                    if trained_model:
                        st.success("Addestramento GNN completato!")
                        config = {
                            "model_type": "SpatioTemporalGNN", "display_name": save_name, "node_order": node_order,
                            "edge_index": edge_index, "edge_weights": edge_weights, "target_columns": selected_targets_gnn,
                            "input_window_steps": iw_steps, "output_window_steps": ow_steps, "hidden_dim": hs,
                            "num_layers": nl, "dropout": dr, "training_mode": training_mode.split()[0].lower(),
                            "loss_function": loss_choice, "node_feature_mapping": node_feature_mapping,
                            "num_features": num_features_detected
                        }
                        if config["training_mode"] == "quantile": config["quantiles"] = quantiles_list
                        base_path = os.path.join(MODELS_DIR, save_name)
                        torch.save(trained_model.state_dict(), f"{base_path}.pth")
                        joblib.dump(sc_f, f"{base_path}_features.joblib")
                        joblib.dump(sc_t, f"{base_path}_targets.joblib")
                        with open(f"{base_path}.json", 'w') as f: json.dump(config, f, indent=4)
                        st.success(f"Modello GNN '{save_name}' salvato.")
                        find_available_models.clear()
                else:
                    st.error("Preparazione dati GNN fallita.")

# --- PAGINA POST-TRAINING MODELLO ---
elif page == 'Post-Training Modello':
    st.header('Post-Training Modello Esistente')

    if not model_ready:
        st.warning("Seleziona un Modello esistente dalla sidebar per caricarlo e poi procedere con il post-training.")
        st.stop()
    
    if not data_ready_csv:
        st.warning("Dati Storici CSV (per il post-training) non disponibili. Caricane uno dalla sidebar.")
        st.stop()

    st.info(f"Modello selezionato per post-training: **{st.session_state.active_model_name}** (Tipo: {active_model_type})")
    
    original_config = st.session_state.active_config
    model_to_fine_tune_state_dict = st.session_state.active_model.state_dict()
    original_scalers = st.session_state.active_scalers

    st.subheader("Configurazione Post-Training")
    
    default_save_name_pt = f"{original_config.get('config_name', 'modello')}_posttrained_{datetime.now(italy_tz).strftime('%Y%m%d_%H%M')}"
    save_name_input_pt = st.text_input("Nome base per salvare il modello affinato:", default_save_name_pt, key="pt_save_filename")
    save_name_pt = re.sub(r'[^\w-]', '_', save_name_input_pt).strip('_') or "modello_posttrained_default"
    if save_name_pt != save_name_input_pt:
        st.caption(f"Nome file valido per il salvataggio: `{save_name_pt}`")

    with st.expander("Parametri di Fine-Tuning", expanded=True):
        c1_pt, c2_pt, c3_pt = st.columns(3)
        with c1_pt:
            ep_pt = st.number_input("Numero Epoche Aggiuntive:", min_value=1, value=10, step=1, key="pt_epochs")
            n_splits_cv_pt = st.number_input("Numero di Fold CV (su nuovi dati):", min_value=1, value=max(1, original_config.get('n_splits_cv', 3)), step=1, key="pt_n_splits_cv")
        with c2_pt:
            lr_pt_default = original_config.get('learning_rate', 0.001)
            lr_pt = st.number_input("Learning Rate per Post-Training:", min_value=1e-7, value=lr_pt_default / 10, format="%.6f", step=1e-5, key="pt_lr")
            bs_pt = st.select_slider("Batch Size:", options=[8, 16, 32, 64], value=original_config.get('batch_size', 32), key="pt_batch_size")
        with c3_pt:
            training_mode_pt = original_config.get("training_mode", "standard")
            if training_mode_pt == 'quantile':
                loss_choice_pt = "QuantileLoss"
                st.info("**Loss:** `QuantileLoss` (dal modello originale)")
            else:
                loss_choice_pt = st.selectbox("Funzione di Loss:", ["MSELoss", "HuberLoss"], index=["MSELoss", "HuberLoss"].index(original_config.get('loss_function', 'MSELoss')), key="pt_loss_choice")
            save_choice_pt = st.radio("Strategia Salvataggio:", ['Migliore', 'Finale'], index=0, key='pt_save_choice', horizontal=True)
    
    st.divider()
    
    if st.button("Avvia Post-Training", type="primary", key="run_post_training_button"):
        st.info(f"Avvio post-training per '{st.session_state.active_model_name}'...")
        
        current_device_pt = st.session_state.active_device
        quantiles_pt = original_config.get("quantiles") if training_mode_pt == 'quantile' else None

        # Logica di preparazione dati e training
        if active_model_type == "LSTM":
            # Prepara nuovi dati usando i vecchi scaler
            with st.spinner("Preparazione nuovi dati per post-training LSTM..."):
                X_new_scaled, y_new_scaled = prepare_training_data(df_current_csv.copy(), original_config["feature_columns"], original_config["target_columns"], original_config["input_window"]/2, original_config["output_window"]/2)[0:2] #Prendiamo solo i primi due elementi
            
            if X_new_scaled is not None:
                num_q_pt = len(quantiles_pt) if training_mode_pt == 'quantile' else 1
                model_instance = HydroLSTM(X_new_scaled.shape[2], original_config["hidden_size"], len(original_config["target_columns"]), original_config["output_window"], original_config["num_layers"], original_config["dropout"], num_quantiles=num_q_pt).to(current_device_pt)
                model_instance.load_state_dict(model_to_fine_tune_state_dict)

                fine_tuned_model, _, _ = train_model(X_new_scaled, y_new_scaled, X_new_scaled.shape[2], len(original_config["target_columns"]), original_config["output_window"],_model_to_continue_train=model_instance, epochs=ep_pt, batch_size=bs_pt, learning_rate=lr_pt, save_strategy='migliore' if 'Migliore' in save_choice_pt else 'finale', n_splits_cv=n_splits_cv_pt, loss_function_name=loss_choice_pt, training_mode=training_mode_pt, quantiles=quantiles_pt)

                if fine_tuned_model:
                    st.success("Post-Training LSTM completato!")
                    # Salva il modello affinato
                    # ... (logica di salvataggio)
        
        elif active_model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
            with st.spinner(f"Preparazione nuovi dati per post-training {active_model_type}..."):
                 data_tuple = prepare_training_data_seq2seq(df_current_csv.copy(), original_config["all_past_feature_columns"], original_config["forecast_input_columns"], original_config["target_columns"], original_config["input_window_steps"], original_config["forecast_window_steps"], original_config["output_window_steps"])

            if data_tuple:
                (X_enc_new, X_dec_new, y_tar_new, _, _, _) = data_tuple
                num_q_pt = len(quantiles_pt) if training_mode_pt == 'quantile' else 1
                
                model_instance = None
                # Ricrea l'istanza del modello con la configurazione originale
                if active_model_type == "Transformer":
                    model_instance = HydroTransformer(X_enc_new.shape[2], X_dec_new.shape[2], len(original_config["target_columns"]), **original_config, num_quantiles=num_q_pt)
                else:
                    encoder = EncoderLSTM(X_enc_new.shape[2], original_config['hidden_size'], original_config['num_layers'], original_config['dropout'])
                    if active_model_type == "Seq2Seq con Attenzione":
                        decoder = DecoderLSTMWithAttention(X_dec_new.shape[2], original_config['hidden_size'], len(original_config["target_columns"]), original_config['num_layers'], original_config['dropout'], num_quantiles=num_q_pt)
                        model_instance = Seq2SeqWithAttention(encoder, decoder, original_config["output_window_steps"])
                    else:
                        decoder = DecoderLSTM(X_dec_new.shape[2], original_config['hidden_size'], len(original_config["target_columns"]), original_config['num_layers'], original_config['dropout'], num_quantiles=num_q_pt)
                        model_instance = Seq2SeqHydro(encoder, decoder, original_config["output_window_steps"])

                model_instance.load_state_dict(model_to_fine_tune_state_dict)

                fine_tuned_model, _, _ = train_model_seq2seq(X_enc_new, X_dec_new, y_tar_new, model_instance, original_config["output_window_steps"], epochs=ep_pt, batch_size=bs_pt, learning_rate=lr_pt, save_strategy='migliore' if 'Migliore' in save_choice_pt else 'finale', n_splits_cv=n_splits_cv_pt, loss_function_name=loss_choice_pt, training_mode=training_mode_pt, quantiles=quantiles_pt)

                if fine_tuned_model:
                    st.success(f"Post-Training {active_model_type} completato!")
                    # Salva il modello affinato
                    # ... (logica di salvataggio)
        else:
            st.error(f"Tipo modello '{active_model_type}' non supportato per il post-training.")
            st.stop()

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption(f'Modello Predittivo Idrologico © {datetime.now().year}')
