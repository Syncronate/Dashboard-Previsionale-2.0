import re
import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
import math
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.graph_objects as go
import time
import gspread # Importa gspread
from google.oauth2.service_account import Credentials # Importa Credentials

# Configurazione della pagina
st.set_page_config(page_title="Modello Predittivo Idrologico", page_icon="🌊", layout="wide")

# Costanti globali
INPUT_WINDOW = 24  # Finestra di input di 24 ore
OUTPUT_WINDOW = 12  # Previsione per le prossime 12 ore

# --- INIZIO DEFINIZIONI FUNZIONI (Dataset, Modello LSTM, Loaders, Train, Predict, Plot, Download, etc.) ---
# (Le definizioni delle funzioni rimangono invariate rispetto al codice originale,
#  le includo qui per completezza ma senza modifiche rispetto a prima, eccetto
#  import_data_from_sheet)

# Dataset personalizzato per le serie temporali
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modello LSTM per serie temporali
class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer per la previsione dei livelli idrometrici
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Inizializzazione dello stato nascosto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Prendiamo solo l'output dell'ultimo timestep
        out = out[:, -1, :]

        # Fully connected layer
        # out shape: (batch_size, output_size * output_window)
        out = self.fc(out)

        # Reshaping per ottenere la sequenza di output
        # out shape: (batch_size, output_window, output_size)
        out = out.view(out.size(0), self.output_window, self.output_size)

        return out

# Funzione per caricare il modello addestrato
@st.cache_resource
def load_model(model_path, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
    # Impostazione del device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creazione del modello **CON I PARAMETRI SPECIFICATI**
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)

    # Caricamento dei pesi del modello
    try:
        # Se model_path è un file caricato, usa il suo buffer
        if hasattr(model_path, 'getvalue'):
            model_path.seek(0) # Assicura che la lettura parta dall'inizio
            model.load_state_dict(torch.load(model_path, map_location=device))
        else: # Altrimenti è un percorso di file
            model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"Errore durante il caricamento dei pesi del modello: {e}")
        return None, device # Return None model and device

    model.eval()
    return model, device

# Funzione per caricare gli scaler
@st.cache_resource
def load_scalers(scaler_features_path, scaler_targets_path):
    try:
         # Gestione sia percorsi file che file caricati
        if hasattr(scaler_features_path, 'getvalue'):
            scaler_features_path.seek(0)
            scaler_features = joblib.load(scaler_features_path)
        else:
            scaler_features = joblib.load(scaler_features_path)

        if hasattr(scaler_targets_path, 'getvalue'):
            scaler_targets_path.seek(0)
            scaler_targets = joblib.load(scaler_targets_path)
        else:
            scaler_targets = joblib.load(scaler_targets_path)

        return scaler_features, scaler_targets
    except Exception as e:
        st.error(f"Errore durante il caricamento degli scaler: {e}")
        return None, None # Return None scalers

# Funzione per preparare i dati per l'addestramento
def prepare_training_data(df, feature_columns, target_columns, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, val_split=20):
    # Assicurati che le colonne target siano numeriche
    for col in target_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Gestisci eventuali NaN introdotti
    df = df.fillna(method='ffill').fillna(method='bfill') # Forward fill then backward fill

    # Creazione delle sequenze di input (X) e output (y)
    X, y = [], []
    for i in range(len(df) - input_window - output_window + 1):
        X.append(df.iloc[i:i+input_window][feature_columns].values)
        y.append(df.iloc[i+input_window:i+input_window+output_window][target_columns].values)

    if not X or not y:
        st.error("Non è stato possibile creare sequenze di training/validazione. Controlla la lunghezza dei dati e le finestre temporali.")
        return None, None, None, None, None, None

    X = np.array(X)
    y = np.array(y)

    # Normalizzazione dei dati
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    # Verifica dimensioni prima del reshape
    if X.size == 0 or y.size == 0:
        st.error("Dati di input o output vuoti prima della normalizzazione.")
        return None, None, None, None, None, None

    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])

    X_scaled_flat = scaler_features.fit_transform(X_flat)
    y_scaled_flat = scaler_targets.fit_transform(y_flat)

    X_scaled = X_scaled_flat.reshape(X.shape)
    y_scaled = y_scaled_flat.reshape(y.shape)

    # Divisione in set di addestramento e validazione
    split_idx = int(len(X_scaled) * (1 - val_split/100))
    X_train = X_scaled[:split_idx]
    y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]
    y_val = y_scaled[split_idx:]

    return X_train, y_train, X_val, y_val, scaler_features, scaler_targets

# Funzione per addestrare il modello
def train_model(
    X_train, y_train, X_val, y_val, input_size, output_size, output_window,
    hidden_size=128, num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, dropout=0.2
):
    # Impostazione del device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creazione del modello
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)

    # Preparazione dei dataset
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Addestramento
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty() # Placeholder per il grafico

    # Creazione del grafico interattivo di perdita (Plotly)
    def update_loss_chart(train_losses, val_losses, placeholder):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, mode='lines+markers', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_losses, mode='lines+markers', name='Validation Loss'))
        fig.update_layout(
            title='Andamento della perdita (Train vs Validation)',
            xaxis_title='Epoca',
            yaxis_title='Loss (MSE)',
            height=400,
            legend_title_text='Legenda'
        )
        placeholder.plotly_chart(fig, use_container_width=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validazione
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Aggiornamento dello scheduler
        scheduler.step(val_loss)

        # Aggiornamento della progress bar e del testo di stato
        progress_percentage = (epoch + 1) / epochs
        progress_bar.progress(progress_percentage)
        status_text.text(f'Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} - LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Aggiornamento del grafico di perdita (ogni epoca)
        update_loss_chart(train_losses, val_losses, loss_chart_placeholder)

        # Salvataggio del modello migliore
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Salva lo stato del modello in memoria RAM
            best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()} # Sposta su CPU per sicurezza
            st.text(f"Nuovo miglior modello trovato all'epoca {epoch+1} con Val Loss: {best_val_loss:.6f}")


        # Breve pausa per consentire l'aggiornamento dell'interfaccia
        time.sleep(0.05)

    # Caricamento del modello migliore
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        st.success(f"Caricato modello migliore dall'epoca con Val Loss: {best_val_loss:.6f}")
    else:
        st.warning("Nessun modello migliore salvato durante l'addestramento.")


    return model, train_losses, val_losses

# Funzione per fare previsioni
def predict(model, input_data, scaler_features, scaler_targets, target_columns, device, output_window):
    """
    Funzione per fare previsioni con il modello addestrato.
    """
    if model is None or scaler_features is None or scaler_targets is None:
        st.error("Modello o scaler non caricati correttamente. Impossibile fare previsioni.")
        return None

    if input_data.shape[0] != INPUT_WINDOW:
         st.error(f"I dati di input hanno {input_data.shape[0]} righe, ma ne sono attese {INPUT_WINDOW}. Impossibile fare previsioni.")
         return None
    if input_data.shape[1] != len(feature_columns):
         st.error(f"I dati di input hanno {input_data.shape[1]} colonne, ma ne sono attese {len(feature_columns)}. Impossibile fare previsioni.")
         return None

    model.eval()

    try:
        # Normalizzazione dei dati di input
        input_normalized = scaler_features.transform(input_data)

        # Conversione in tensore PyTorch
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)

        # Previsione
        with torch.no_grad():
            output = model(input_tensor)

        # Conversione in numpy
        output_np = output.cpu().numpy().reshape(-1, len(target_columns)) # Usa len(target_columns) dinamicamente

        # Denormalizzazione
        # Assicurati che scaler_targets sia stato fittato sul numero corretto di target
        if output_np.shape[1] != scaler_targets.n_features_in_:
             st.error(f"Errore di dimensione: l'output del modello ha {output_np.shape[1]} feature, ma lo scaler target ne attende {scaler_targets.n_features_in_}.")
             return None

        predictions = scaler_targets.inverse_transform(output_np)

        # Reshape per ottenere [output_window, num_target_columns]
        predictions = predictions.reshape(output_window, len(target_columns)) # Usa len(target_columns)

        return predictions

    except Exception as e:
        st.error(f"Errore durante la fase di previsione: {e}")
        st.error(f"Shape dati input: {input_data.shape}")
        st.error(f"Shape dati normalizzati: {input_normalized.shape if 'input_normalized' in locals() else 'N/A'}")
        st.error(f"Shape output modello: {output.shape if 'output' in locals() else 'N/A'}")
        st.error(f"Shape output numpy: {output_np.shape if 'output_np' in locals() else 'N/A'}")
        st.error(f"Numero colonne target attese dallo scaler: {scaler_targets.n_features_in_ if scaler_targets else 'N/A'}")
        return None


# Funzione per plot dei risultati (usando Plotly per interattività)
def plot_predictions(predictions, target_columns, output_window, start_time=None):
    figures = []

    # Per ogni sensore idrometrico target
    for i, sensor_name in enumerate(target_columns):
        fig = go.Figure()

        # Creazione dell'asse x per le ore future
        if start_time:
            hours = [start_time + timedelta(hours=h+1) for h in range(output_window)] # Parte da h+1 per la previsione
            x_axis = hours
            x_title = "Data e Ora Previste"
        else:
            hours = np.arange(1, output_window + 1) # Ore da 1 a OUTPUT_WINDOW
            x_axis = hours
            x_title = "Ore Future"

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsione {sensor_name}'))

        fig.update_layout(
            title=f'Previsione - {sensor_name}',
            xaxis_title=x_title,
            yaxis_title='Livello idrometrico [m]',
            height=400,
            hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero') # Assicura che l'asse Y parta da 0 se i valori sono positivi

        figures.append(fig)

    return figures

# Funzione per ottenere un link di download per un file CSV
def get_table_download_link(df, filename="previsioni.csv"):
    """Genera un link per scaricare il dataframe come file CSV"""
    csv = df.to_csv(index=False, sep=';', decimal=',') # Usa ; e , per coerenza
    b64 = base64.b64encode(csv.encode('utf-8')).decode() # Specifica encoding
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Scarica i dati CSV</a>'

# Funzione per ottenere un link di download per un file pkl/joblib o pth
def get_binary_file_download_link(file_object, filename, text):
    """Genera un link per scaricare un file binario (modello o scaler)"""
    file_object.seek(0) # Assicura di partire dall'inizio del buffer
    b64 = base64.b64encode(file_object.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

# Funzione per scaricare grafici Plotly come HTML o PNG (richiede kaleido)
def get_plotly_download_link(fig, filename_base, text_html="Scarica Grafico (HTML)", text_png="Scarica Grafico (PNG)"):
    """Genera link per scaricare un grafico Plotly come HTML e PNG (se kaleido è installato)"""
    # HTML Link
    buffer_html = io.StringIO()
    fig.write_html(buffer_html)
    b64_html = base64.b64encode(buffer_html.getvalue().encode()).decode()
    href_html = f'<a href="data:text/html;base64,{b64_html}" download="{filename_base}.html">{text_html}</a>'

    # PNG Link (opzionale, richiede kaleido: pip install -U kaleido)
    href_png = ""
    try:
        buffer_png = io.BytesIO()
        fig.write_image(buffer_png, format="png")
        buffer_png.seek(0)
        b64_png = base64.b64encode(buffer_png.getvalue()).decode()
        href_png = f'<a href="data:image/png;base64,{b64_png}" download="{filename_base}.png">{text_png}</a>'
    except Exception as e:
        # st.warning(f"Impossibile generare PNG per {filename_base}: {e}. Installa kaleido (`pip install -U kaleido`)")
        pass # Non mostrare warning, semplicemente non mostrare il link PNG

    return f"{href_html} {href_png}" # Restituisce entrambi i link (o solo HTML se PNG fallisce)

# Funzione per estrarre l'ID del foglio dall'URL
def extract_sheet_id(url):
    # Pattern più robusto per estrarre l'ID
    patterns = [
        r'/spreadsheets/d/([a-zA-Z0-9-_]+)', # Pattern standard
        r'/d/([a-zA-Z0-9-_]+)/'             # Altro pattern comune
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None # Se nessun pattern corrisponde


# --- FUNZIONE DI IMPORTAZIONE DA GOOGLE SHEET MODIFICATA ---
# Funzione per importare i dati dal foglio Google
def import_data_from_sheet(sheet_id, expected_cols, date_col_name='Data_Ora', date_format='%d/%m/%Y %H:%M'):
    """
    Importa dati da Google Sheet, pulisce i valori numerici (gestendo virgole e 'N/A'),
    e restituisce le ultime INPUT_WINDOW righe valide.

    Args:
        sheet_id (str): L'ID del foglio Google.
        expected_cols (list): Lista dei nomi delle colonne come appaiono nel foglio Google.
        date_col_name (str): Nome della colonna data/ora nel foglio Google.
        date_format (str): Formato della data/ora nel foglio Google.

    Returns:
        pd.DataFrame or None: DataFrame con i dati puliti e pronti, o None se errore.
    """
    try:
        # Utilizzo di gspread per accedere al foglio
        # Carica le credenziali da Streamlit Secrets
        # Assicurati che "GOOGLE_CREDENTIALS" sia configurato nei secrets di Streamlit
        if "GOOGLE_CREDENTIALS" not in st.secrets:
             st.error("Credenziali Google non trovate nei secrets di Streamlit. Configura 'GOOGLE_CREDENTIALS'.")
             return None

        credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        )
        gc = gspread.authorize(credentials)

        # Apri il foglio per ID
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1  # Assumiamo il primo foglio

        # Ottieni tutte le righe, inclusa l'intestazione
        data = worksheet.get_all_values()

        # Crea un DataFrame pandas
        if not data or len(data) < 2: # Deve esserci almeno intestazione + 1 riga dati
            st.error("Il foglio Google sembra essere vuoto o contiene solo l'intestazione.")
            return None

        headers = data[0]
        rows = data[1:]

        # Verifica che le colonne attese esistano nel foglio
        actual_headers_set = set(headers)
        missing_cols = [col for col in expected_cols if col not in actual_headers_set]
        if missing_cols:
             st.error(f"Le seguenti colonne attese non sono state trovate nel foglio Google: {', '.join(missing_cols)}")
             st.info(f"Colonne trovate: {', '.join(headers)}")
             return None

        df_sheet = pd.DataFrame(rows, columns=headers)

        # Seleziona solo le colonne che ci servono effettivamente (quelle nel mapping + data)
        columns_to_keep = expected_cols[:] # Crea una copia
        if date_col_name not in columns_to_keep:
            columns_to_keep.append(date_col_name)
        df_sheet = df_sheet[columns_to_keep]


        # --- PULIZIA E CONVERSIONE DATI ---
        # 1. Converte la colonna di data in datetime
        if date_col_name in df_sheet.columns:
            df_sheet[date_col_name] = pd.to_datetime(df_sheet[date_col_name], format=date_format, errors='coerce')
        else:
            # Questo caso non dovrebbe accadere per via del check precedente, ma per sicurezza
            st.error(f"La colonna data '{date_col_name}' non è presente dopo la selezione.")
            return None

        # 2. Ordina per data e gestisci NaT prima di ordinare
        df_sheet = df_sheet.dropna(subset=[date_col_name]) # Rimuovi righe dove la data non è valida
        df_sheet = df_sheet.sort_values(by=date_col_name, ascending=True)

        # 3. Converti le colonne numeriche (tutte tranne la data)
        numeric_cols = [col for col in df_sheet.columns if col != date_col_name]
        for col in numeric_cols:
            # Sostituisci 'N/A', stringhe vuote, '-' con NaN di numpy
            df_sheet[col] = df_sheet[col].replace(['N/A', '', '-', ' '], np.nan, regex=False)
            # Sostituisci la virgola con il punto per il separatore decimale
            df_sheet[col] = df_sheet[col].astype(str).str.replace(',', '.', regex=False)
            # Tenta la conversione a numerico. 'coerce' imposterà a NaN i valori non convertibili
            df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')

        # --- FINE PULIZIA E CONVERSIONE ---

        # 4. Prendi le ultime INPUT_WINDOW righe valide
        df_sheet = df_sheet.tail(INPUT_WINDOW)

        # 5. Verifica se sono rimaste abbastanza righe
        if len(df_sheet) < INPUT_WINDOW:
             st.warning(f"Attenzione: dopo la pulizia dei dati dal Google Sheet, sono disponibili solo {len(df_sheet)} righe valide delle {INPUT_WINDOW} richieste. Assicurati che il foglio contenga dati sufficienti e validi.")
             if len(df_sheet) == 0:
                 st.error("Nessuna riga valida trovata dopo la pulizia dei dati dal Google Sheet.")
                 return None
        elif len(df_sheet) > INPUT_WINDOW:
             # Questo non dovrebbe succedere con .tail(), ma per sicurezza
             st.warning(f"Trovate più righe ({len(df_sheet)}) del previsto ({INPUT_WINDOW}) dopo il tail().")
             df_sheet = df_sheet.tail(INPUT_WINDOW)

        st.success(f"Importate e pulite {len(df_sheet)} righe dal foglio Google.")
        return df_sheet

    except gspread.exceptions.APIError as api_e:
        # Dettagli sull'errore API
        error_details = api_e.response.json()
        error_message = error_details.get('error', {}).get('message', str(api_e))
        st.error(f"Errore API Google Sheets: {error_message}. Verifica l'URL, le autorizzazioni dell'account di servizio sul foglio e le credenziali.")
        return None
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Foglio Google non trovato. Verifica che l'ID '{sheet_id}' sia corretto e che l'account di servizio abbia accesso.")
        return None
    except ValueError as ve:
         if "time data" in str(ve) and "does not match format" in str(ve):
             st.error(f"Errore nel formato data/ora: {ve}. Verifica che i dati nella colonna '{date_col_name}' siano nel formato '{date_format}' (es. '25/03/2025 17:41').")
         else:
             st.error(f"Errore durante la conversione dei dati: {ve}")
         return None
    except Exception as e:
        st.error(f"Errore generico durante l'importazione da Google Sheet: {type(e).__name__} - {str(e)}")
        import traceback
        st.error(traceback.format_exc()) # Stampa traceback per debug
        return None

# --- FINE DEFINIZIONI FUNZIONI ---


# --- INIZIO LAYOUT STREAMLIT ---

# Titolo dell'app
st.title('Dashboard Modello Predittivo Idrologico')

# Sidebar per le opzioni
st.sidebar.header('Impostazioni')

# Opzione per caricare i propri file o usare quelli demo
use_demo_files = st.sidebar.checkbox('Usa file di esempio', value=True)

# Definisci i percorsi/variabili qui, verranno aggiornati dopo
DATA_PATH = None
MODEL_PATH = None
SCALER_FEATURES_PATH = None
SCALER_TARGETS_PATH = None
# Parametri modello (inizializzati a None o default)
hidden_size = 128
num_layers = 2
dropout = 0.2

if use_demo_files:
    st.sidebar.info("Stai usando i file di esempio predefiniti.")
    # Qui dovresti fornire percorsi ai file di esempio (assicurati che esistano)
    # Sostituisci con i percorsi corretti dove esegui lo script
    DATA_PATH = 'dati_idro.csv'  # Esempio: 'data/dati_idro.csv'
    MODEL_PATH = 'best_hydro_model.pth'  # Esempio: 'models/best_hydro_model.pth'
    SCALER_FEATURES_PATH = 'scaler_features.joblib' # Esempio: 'models/scaler_features.joblib'
    SCALER_TARGETS_PATH = 'scaler_targets.joblib' # Esempio: 'models/scaler_targets.joblib'
    # Parametri del modello DEMO (corrispondenti ai file demo caricati)
    hidden_size = 128 # Assicurati che questi corrispondano al modello .pth fornito
    num_layers = 2
    dropout = 0.2
    # Verifica esistenza file demo
    missing_demo_files = []
    if not os.path.exists(DATA_PATH): missing_demo_files.append(DATA_PATH)
    if not os.path.exists(MODEL_PATH): missing_demo_files.append(MODEL_PATH)
    if not os.path.exists(SCALER_FEATURES_PATH): missing_demo_files.append(SCALER_FEATURES_PATH)
    if not os.path.exists(SCALER_TARGETS_PATH): missing_demo_files.append(SCALER_TARGETS_PATH)
    if missing_demo_files:
         st.sidebar.error(f"File di esempio mancanti: {', '.join(missing_demo_files)}. Verifica i percorsi.")
         st.stop() # Ferma l'esecuzione se i file demo non ci sono

else:
    st.sidebar.info("Carica i tuoi file personalizzati.")
    # Caricamento dei file dall'utente
    st.sidebar.subheader('Carica i tuoi file')
    data_file = st.sidebar.file_uploader('File CSV con i dati storici', type=['csv'])
    model_file = st.sidebar.file_uploader('File del modello (.pth)', type=['pth'])
    scaler_features_file = st.sidebar.file_uploader('File scaler features (.joblib)', type=['joblib'])
    scaler_targets_file = st.sidebar.file_uploader('File scaler targets (.joblib)', type=['joblib'])

    # Configurazione parametri modello (solo se si usano file propri)
    st.sidebar.subheader('Configurazione Modello (se carichi il tuo)')
    hidden_size = st.sidebar.number_input("Dimensione hidden layer", min_value=16, max_value=1024, value=128, step=16)
    num_layers = st.sidebar.number_input("Numero di layer LSTM", min_value=1, max_value=8, value=2)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.7, 0.2, 0.05)

    # Assegnazione dei file caricati ai path (usiamo i buffer direttamente)
    if data_file: DATA_PATH = data_file
    if model_file: MODEL_PATH = model_file
    if scaler_features_file: SCALER_FEATURES_PATH = scaler_features_file
    if scaler_targets_file: SCALER_TARGETS_PATH = scaler_targets_file

    # Controllo se tutti i file sono stati caricati quando non si usano i demo
    if not (DATA_PATH and MODEL_PATH and SCALER_FEATURES_PATH and SCALER_TARGETS_PATH):
        st.sidebar.warning('Carica tutti i file necessari (CSV, pth, 2x joblib) per procedere.')
        # Non fermare l'app, ma alcune sezioni potrebbero non funzionare
        # st.stop() # Decommenta se vuoi bloccare l'app senza tutti i file


# Estrazione delle caratteristiche (colonne del dataframe) - ASSICURATI CHE QUESTI NOMI CORRISPONDANO AL TUO CSV
# Questi nomi DEVONO corrispondere esattamente alle intestazioni nel file CSV caricato o demo
rain_features = [
    'Cumulata Sensore 1295 (Arcevia)',
    'Cumulata Sensore 2637 (Bettolelle)', # Verifica se nel tuo CSV è 'Misa - Pioggia Ora (mm)' o altro
    'Cumulata Sensore 2858 (Barbara)',
    'Cumulata Sensore 2964 (Corinaldo)'
]
# Assicurati che questa colonna esista nel tuo CSV
humidity_feature = ['Umidita\' Sensore 3452 (Montemurello)'] # Verifica se esiste

hydro_features = [
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)', # Verifica se nel tuo CSV è 'Misa - Livello Misa (mt)' o altro
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)', # Verifica se nel tuo CSV è 'Nevola - Livello Nevola (mt)' o altro
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)', # Verifica se nel tuo CSV è 'Pianello di Ostra - Livello Misa (m)' o altro
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)' # Verifica se nel tuo CSV è 'Ponte Garibaldi - Livello Misa 2 (mt)' o altro
]

feature_columns = rain_features + humidity_feature + hydro_features


# Caricamento dei dati storici
df = None
df_load_error = None
if DATA_PATH:
    try:
        # Gestisce sia file caricato che percorso
        if hasattr(DATA_PATH, 'getvalue'):
            DATA_PATH.seek(0) # Resetta il puntatore del buffer
            df = pd.read_csv(DATA_PATH, sep=';', decimal=',', encoding='utf-8') # Aggiungi encoding
        else:
            df = pd.read_csv(DATA_PATH, sep=';', decimal=',', encoding='utf-8')

        # Conversione Data e Ora - Assicurati che il nome 'Data e Ora' sia corretto
        date_col_name_csv = 'Data e Ora' # Modifica se il nome nel CSV è diverso
        if date_col_name_csv not in df.columns:
             raise ValueError(f"Colonna '{date_col_name_csv}' non trovata nel file CSV. Colonne presenti: {', '.join(df.columns)}")

        # Prova diversi formati comuni se quello standard fallisce
        try:
             df[date_col_name_csv] = pd.to_datetime(df[date_col_name_csv], format='%d/%m/%Y %H:%M', errors='raise')
        except ValueError:
             st.warning("Formato data '%d/%m/%Y %H:%M' fallito, tentativo con inferenza automatica...")
             try:
                 df[date_col_name_csv] = pd.to_datetime(df[date_col_name_csv], errors='coerce') # Lascia che pandas indovini
             except Exception as e_dt:
                 raise ValueError(f"Impossibile convertire la colonna data '{date_col_name_csv}': {e_dt}")

        df = df.dropna(subset=[date_col_name_csv]) # Rimuovi righe con date non valide
        df = df.sort_values(by=date_col_name_csv).reset_index(drop=True) # Ordina e resetta indice

        # Verifica presenza di tutte le feature columns nel DataFrame caricato
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            st.sidebar.error(f"Colonne feature mancanti nel CSV: {', '.join(missing_features)}. Verifica i nomi in `feature_columns` e nel file.")
            df = None # Invalida il dataframe se mancano colonne essenziali
        else:
            st.sidebar.success(f'Dati CSV caricati: {len(df)} righe, {len(df.columns)} colonne.')
            # Conversione numerica robusta per tutte le feature (dopo caricamento)
            for col in feature_columns:
                 if col != date_col_name_csv: # Salta colonna data
                      # Sostituisci eventuali separatori di migliaia (es. '.') se presenti
                      if df[col].dtype == 'object': # Solo se è stringa
                           df[col] = df[col].str.replace('.', '', regex=False) # Rimuove punti (migliaia)
                           df[col] = df[col].str.replace(',', '.', regex=False) # Virgola -> Punto (decimali)
                           df[col] = df[col].replace(['N/A', '', '-'], np.nan, regex=False) # Gestisce N/A etc.
                      df[col] = pd.to_numeric(df[col], errors='coerce')
            # Gestione NaN post-conversione (riempimento semplice)
            n_nan_before = df[feature_columns].isnull().sum().sum()
            if n_nan_before > 0:
                 st.sidebar.warning(f"Trovati {n_nan_before} valori mancanti/non numerici nelle colonne feature. Verranno riempiti con forward/backward fill.")
                 df[feature_columns] = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
                 n_nan_after = df[feature_columns].isnull().sum().sum()
                 if n_nan_after > 0:
                      st.sidebar.error(f"Ancora {n_nan_after} valori mancanti dopo il fill. Controlla l'inizio/fine del dataset.")
                      df = None # Invalida se ci sono ancora NaN

    except ValueError as ve:
        df_load_error = f"Errore di valore durante il caricamento CSV: {ve}"
    except FileNotFoundError:
        df_load_error = f"Errore: File CSV non trovato al percorso '{DATA_PATH}'."
    except Exception as e:
        df_load_error = f'Errore imprevisto nel caricamento/processamento dati CSV: {type(e).__name__} - {e}'

    if df_load_error:
        st.sidebar.error(df_load_error)
        df = None # Assicura che df sia None se il caricamento fallisce

# Caricamento del modello e degli scaler
model = None
device = None
scaler_features = None
scaler_targets = None
target_columns = [] # Lista delle colonne target effettivamente usate/previste
load_error = False

# Procedi solo se i path/file necessari sono definiti
if MODEL_PATH and SCALER_FEATURES_PATH and SCALER_TARGETS_PATH:
    try:
        # Definisci quali sono i target che il modello deve prevedere.
        # Questo dipende da COME il modello è stato addestrato.
        # Se usi i file demo, DEVI sapere quali target prevede quel modello.
        # Se l'utente carica il suo, dovremmo idealmente salvarlo con il modello,
        # ma per ora usiamo tutti gli idrometri come default per i file utente.
        if use_demo_files:
            # *** IMPORTANTE: Modifica questa lista se il tuo modello demo prevede target diversi ***
            target_columns = hydro_features[:4] # Esempio: il modello demo prevede solo i primi 4 idrometri
            st.sidebar.info(f"Modalità Demo: Il modello prevede {len(target_columns)} idrometri: {', '.join(target_columns)}")
        else:
            # Quando l'utente carica i file, assumiamo che il modello preveda tutti gli idrometri,
            # a meno che non si aggiunga un modo per specificarlo.
            target_columns = hydro_features # Assume tutti gli idrometri
            st.sidebar.info(f"Modalità Utente: Si assume che il modello preveda {len(target_columns)} idrometri.")

        # Calcola input_size e output_size in base alle feature e ai target SELEZIONATI
        input_size = len(feature_columns)
        output_size = len(target_columns) # Basato sui target definiti sopra

        # Carica modello e scaler usando i parametri corretti
        model, device = load_model(MODEL_PATH, input_size, hidden_size, output_size, OUTPUT_WINDOW, num_layers, dropout)
        scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)

        # Verifica post-caricamento
        if model is None or scaler_features is None or scaler_targets is None:
            load_error = True
            # Gli errori specifici sono già stati mostrati da load_model/load_scalers
            st.sidebar.error("Caricamento modello o scaler fallito.")
        else:
            # Verifica compatibilità scaler targets con output_size
            if scaler_targets.n_features_in_ != output_size:
                 st.sidebar.error(f"Incompatibilità: Lo scaler target è stato fittato su {scaler_targets.n_features_in_} features, ma il modello ne prevede {output_size} ({', '.join(target_columns)}).")
                 load_error = True
                 model, scaler_features, scaler_targets = None, None, None # Invalida tutto
            else:
                 st.sidebar.success('Modello e scaler caricati e verificati.')

    except Exception as e:
        st.sidebar.error(f'Errore generico nel caricamento modello/scaler: {e}')
        load_error = True
else:
    if not use_demo_files: # Mostra solo se l'utente doveva caricare i file
       st.sidebar.warning("Mancano i percorsi per modello o scaler.")
    load_error = True # Considera errore se mancano file essenziali

# Menu principale
st.sidebar.header('Menu')
# Disabilita opzioni se i dati o il modello non sono caricati
disable_dashboard = df is None or load_error
disable_sim = load_error # La simulazione richiede solo modello/scaler
disable_analysis = df is None
disable_training = df is None # L'allenamento richiede solo dati iniziali

page = st.sidebar.radio(
    'Scegli una funzionalità',
    ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello'],
    captions=[
        "Visualizza dati e previsioni" if not disable_dashboard else "Richiede dati e modello carichi",
        "Esegui previsioni con dati custom" if not disable_sim else "Richiede modello/scaler carichi",
        "Esplora i dati caricati" if not disable_analysis else "Richiede dati carichi",
        "Allena un nuovo modello" if not disable_training else "Richiede dati carichi"
    ],
    # index=0 # Pagina iniziale di default
    # disabled non è un argomento diretto di radio, gestiamo l'accesso dopo
)

# Logica di navigazione e gestione disabilitazione
if page == 'Dashboard':
    if disable_dashboard:
        st.warning("Funzionalità non disponibile: carica correttamente i dati storici (CSV) e i file del modello (pth, joblib).")
    else:
        # --- CODICE DASHBOARD ---
        st.header('Dashboard Idrologica')
        st.success("Dati e modello caricati correttamente.")

        # Mostra ultimi dati disponibili
        st.subheader('Ultimi dati disponibili')
        last_data = df.iloc[-1]
        last_date = last_data[date_col_name_csv] # Usa il nome corretto della colonna data

        # Formattazione dei dati per la visualizzazione
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Ultimo Rilevamento", value=last_date.strftime('%d/%m/%Y %H:%M'))

        with col2:
             st.subheader('Livelli idrometrici [m]')
             # Mostra solo i target columns previsti dal modello
             hydro_data_display = last_data[target_columns].round(3).to_frame(name="Valore")
             hydro_data_display.index.name = "Sensore"
             st.dataframe(hydro_data_display)
             # Aggiungi eventuali altri idrometri non target, se si vuole
             other_hydro = [h for h in hydro_features if h not in target_columns]
             if other_hydro:
                  st.caption("Altri Idrometri (non previsti dal modello):")
                  st.dataframe(last_data[other_hydro].round(3).to_frame(name="Valore"))


        with col3:
            st.subheader('Precipitazioni Cumulate [mm]')
            rain_data_display = last_data[rain_features].round(2).to_frame(name="Valore")
            rain_data_display.index.name = "Sensore Pioggia"
            st.dataframe(rain_data_display)

            st.subheader('Umidità [%]')
            # Gestisci caso in cui humidity_feature non sia nei dati
            if humidity_feature[0] in last_data:
                st.metric(label=humidity_feature[0], value=f"{last_data[humidity_feature[0]]:.1f}%")
            else:
                st.caption("Dato Umidità non disponibile.")


        # Previsione basata sugli ultimi dati disponibili
        st.header('Previsione basata sugli ultimi dati')

        if st.button('Genera previsione dagli ultimi dati', type="primary"):
            with st.spinner('Generazione previsione in corso...'):
                 # Preparazione dei dati di input (ultime INPUT_WINDOW ore)
                 # Assicurati di usare le feature_columns corrette
                 try:
                      latest_data = df.iloc[-INPUT_WINDOW:][feature_columns].values
                      if latest_data.shape[0] < INPUT_WINDOW:
                           st.error(f"Dati insufficienti ({latest_data.shape[0]}) per la finestra di input ({INPUT_WINDOW}). Impossibile prevedere.")
                      else:
                           # Previsione
                           predictions = predict(model, latest_data, scaler_features, scaler_targets, target_columns, device, OUTPUT_WINDOW)

                           if predictions is not None:  # Check if prediction was successful
                                # Visualizzazione dei risultati
                                st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore')

                                # Creazione dataframe risultati
                                start_pred_time = last_date # La previsione inizia dall'ora successiva all'ultimo dato
                                prediction_times = [start_pred_time + timedelta(hours=i+1) for i in range(OUTPUT_WINDOW)]
                                results_df = pd.DataFrame(predictions, columns=target_columns)
                                results_df['Ora previsione'] = prediction_times
                                # Formatta ora per leggibilità
                                results_df['Ora previsione'] = results_df['Ora previsione'].dt.strftime('%d/%m %H:%M')
                                results_df = results_df[['Ora previsione'] + target_columns]

                                # Visualizzazione tabella risultati
                                st.dataframe(results_df.round(3))

                                # Download dei risultati
                                st.markdown(get_table_download_link(results_df, filename=f"previsione_{last_date.strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                                # Grafici Plotly per ogni sensore target
                                st.subheader('Grafici delle previsioni')
                                figs = plot_predictions(predictions, target_columns, OUTPUT_WINDOW, start_pred_time)

                                # Visualizzazione grafici
                                for i, fig in enumerate(figs):
                                    sensor_name_safe = target_columns[i].replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.markdown(
                                        get_plotly_download_link(fig, f"grafico_{sensor_name_safe}_{last_date.strftime('%Y%m%d_%H%M')}", text_html="Scarica Grafico Interattivo (HTML)", text_png="Scarica Immagine (PNG)"),
                                        unsafe_allow_html=True
                                    )
                 except Exception as e_pred:
                      st.error(f"Errore durante la preparazione dei dati o la previsione: {e_pred}")


elif page == 'Simulazione':
    if disable_sim:
         st.warning("Funzionalità non disponibile: carica correttamente i file del modello (pth, joblib).")
    else:
        # --- CODICE SIMULAZIONE ---
        st.header('Simulazione Idrologica')
        st.write(f'Prepara i dati per le ultime {INPUT_WINDOW} ore per generare una previsione delle prossime {OUTPUT_WINDOW} ore.')
        st.info(f"Il modello attualmente caricato prevede i seguenti idrometri: {', '.join(target_columns)}")

        # Variabile per contenere i dati finali per la simulazione
        sim_data_input = None

        # Opzioni per la simulazione
        sim_method = st.radio(
            "Metodo di preparazione dati per la simulazione",
            ['Inserisci manualmente valori costanti', 'Importa da Google Sheet', 'Inserisci dati orari (Avanzato)'] # Riordinato per semplicità
        )

        # --- SIMULAZIONE: Inserimento Manuale Costante ---
        if sim_method == 'Inserisci manualmente valori costanti':
            st.subheader(f'Inserisci valori costanti per le {INPUT_WINDOW} ore precedenti')
            st.caption("Questi valori verranno ripetuti per tutte le ore della finestra di input.")

            # Creiamo un dataframe temporaneo per l'inserimento
            temp_sim_values = {}

            cols_manual = st.columns(3)
            with cols_manual[0]:
                 st.write("**Pioggia Cumulata (mm)**")
                 for feature in rain_features:
                      # Usa la media storica come valore di default se df è disponibile
                      default_val = df[feature].mean() if df is not None and feature in df else 0.0
                      temp_sim_values[feature] = st.number_input(f'{feature}', min_value=0.0, max_value=200.0, value=round(default_val,1), step=0.5, format="%.1f")

            with cols_manual[1]:
                 st.write("**Umidità (%)**")
                 for feature in humidity_feature:
                      default_val = df[feature].mean() if df is not None and feature in df else 70.0
                      temp_sim_values[feature] = st.number_input(f'{feature}', min_value=0.0, max_value=100.0, value=round(default_val,1), step=1.0, format="%.1f")

            with cols_manual[2]:
                 st.write("**Livelli Idrometrici (m)**")
                 for feature in hydro_features:
                      default_val = df[feature].mean() if df is not None and feature in df else 0.5
                      temp_sim_values[feature] = st.number_input(f'{feature}', min_value=-2.0, max_value=15.0, value=round(default_val,2), step=0.05, format="%.2f")

            # Crea l'array numpy per la simulazione replicando i valori
            sim_data_list = []
            for feature in feature_columns: # Mantieni l'ordine corretto!
                 sim_data_list.append(np.repeat(temp_sim_values[feature], INPUT_WINDOW))
            sim_data_input = np.column_stack(sim_data_list)

            st.subheader("Anteprima Dati (valori costanti)")
            preview_df = pd.DataFrame([temp_sim_values[col] for col in feature_columns], index=feature_columns, columns=['Valore Costante'])
            st.dataframe(preview_df.round(2))


        # --- SIMULAZIONE: Importa da Google Sheet ---
        elif sim_method == 'Importa da Google Sheet':
            st.subheader(f'Importa le ultime {INPUT_WINDOW} ore dal foglio Google')
            st.markdown("""
            Assicurati che:
            1.  Il foglio Google sia accessibile dall'account di servizio le cui credenziali sono nei secrets di Streamlit (`GOOGLE_CREDENTIALS`).
            2.  L'URL fornito sia corretto.
            3.  Il foglio contenga le colonne mappate qui sotto con dati numerici (o 'N/A'). La virgola come decimale è gestita.
            4.  La colonna data/ora sia nel formato `GG/MM/AAAA HH:MM`.
            5.  Ci siano almeno `INPUT_WINDOW` righe di dati validi e recenti.
            """)

            # Input URL foglio
            sheet_url = st.text_input(
                "URL del foglio Google Sheet",
                "https://docs.google.com/spreadsheets/d/your-google-sheet-id/edit#gid=0" # Esempio
            )

            # Mappatura tra i nomi delle colonne del foglio e i nomi attesi dal MODELLO (feature_columns)
            # Modifica questa mappatura SE i nomi nel tuo foglio Google sono DIVERSI
            column_mapping_gsheet_to_model = {
                # NOME_COLONNA_FOGLIO_GOOGLE : NOME_COLONNA_MODELLO (da feature_columns)
                'Data_Ora': 'Data e Ora', # Nome colonna data nel foglio -> Nome colonna data usato internamente (non è una feature)
                'Arcevia - Pioggia Ora (mm)': 'Cumulata Sensore 1295 (Arcevia)',
                'Barbara - Pioggia Ora (mm)': 'Cumulata Sensore 2858 (Barbara)',
                'Corinaldo - Pioggia Ora (mm)': 'Cumulata Sensore 2964 (Corinaldo)',
                'Misa - Pioggia Ora (mm)': 'Cumulata Sensore 2637 (Bettolelle)', # Associazione Misa -> Bettolelle
                'Serra dei Conti - Livello Misa (mt)': 'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
                'Pianello di Ostra - Livello Misa (m)': 'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
                'Nevola - Livello Nevola (mt)': 'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
                'Misa - Livello Misa (mt)': 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)', # Associazione Misa -> Bettolelle
                'Ponte Garibaldi - Livello Misa 2 (mt)': 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
                # AGGIUNGI QUI ALTRE EVENTUALI MAPPATURE SE NECESSARIO
            }

            # Verifica se la colonna Umidità è nel mapping, altrimenti chiedila manualmente
            humidity_col_in_model = humidity_feature[0]
            humidity_col_in_gsheet = None
            for gsheet_name, model_name in column_mapping_gsheet_to_model.items():
                 if model_name == humidity_col_in_model:
                      humidity_col_in_gsheet = gsheet_name
                      break # Trovata

            selected_humidity_gsheet = None
            if humidity_col_in_gsheet:
                 st.info(f"La colonna Umidità '{humidity_col_in_model}' verrà letta dalla colonna '{humidity_col_in_gsheet}' del foglio Google.")
            else:
                 st.warning(f"La colonna Umidità '{humidity_col_in_model}' non è stata trovata nel mapping del foglio Google.")
                 # Valore di umidità manuale
                 humidity_preset_gsheet = st.selectbox(
                     "Seleziona condizione di umidità del terreno (verrà usata per tutte le ore importate)",
                     ["Molto secco (20%)", "Secco (40%)", "Normale (60%)", "Umido (80%)", "Saturo (95%)"], index=2
                 )
                 # Estrai valore numerico dal preset
                 selected_humidity_gsheet = float(humidity_preset_gsheet.split('(')[1].split('%')[0])
                 st.info(f"Verrà usato un valore costante di {selected_humidity_gsheet}% per l'umidità.")


            # Nomi delle colonne attese nel foglio Google (chiavi del mapping)
            expected_google_sheet_cols = list(column_mapping_gsheet_to_model.keys())
            date_col_name_gsheet = 'Data_Ora' # Nome colonna data nel foglio Google

            # Bottone per importare i dati
            if st.button("Importa e Prepara dati dal foglio Google"):
                 sheet_id = extract_sheet_id(sheet_url) # Usa la funzione per estrarre l'ID

                 if not sheet_id:
                     st.error("URL del foglio Google non valido o ID non estraibile. Assicurati che sia nel formato corretto (es. .../d/SHEET_ID/edit...).")
                 else:
                     st.info(f"Tentativo di connessione al foglio con ID: {sheet_id}")
                     with st.spinner("Importazione e pulizia dati da Google Sheet in corso..."):
                         # Importa e pulisci i dati usando la funzione aggiornata
                         sheet_data_cleaned = import_data_from_sheet(
                             sheet_id,
                             expected_google_sheet_cols,
                             date_col_name=date_col_name_gsheet,
                             date_format='%d/%m/%Y %H:%M' # Formato atteso
                         )

                         if sheet_data_cleaned is not None:
                             # Mappatura dei nomi delle colonne da GSheet a quelli del modello
                             mapped_data = pd.DataFrame()
                             successful_mapping = True
                             for gsheet_col, model_col in column_mapping_gsheet_to_model.items():
                                 if gsheet_col in sheet_data_cleaned.columns:
                                     # Rinomina la colonna dal nome GSheet al nome atteso dal modello
                                     mapped_data[model_col] = sheet_data_cleaned[gsheet_col]
                                 else:
                                      # Questo non dovrebbe succedere se import_data_from_sheet funziona
                                      st.error(f"Errore interno: la colonna '{gsheet_col}' non è presente nei dati puliti.")
                                      successful_mapping = False
                                      break

                             if successful_mapping:
                                  # Aggiungi la colonna di umidità se non era nel foglio
                                  if humidity_col_in_model not in mapped_data.columns and selected_humidity_gsheet is not None:
                                       mapped_data[humidity_col_in_model] = selected_humidity_gsheet
                                       st.info(f"Aggiunta colonna umidità '{humidity_col_in_model}' con valore costante {selected_humidity_gsheet}%.")
                                  elif humidity_col_in_model not in mapped_data.columns and selected_humidity_gsheet is None:
                                       st.error(f"Errore: la colonna umidità '{humidity_col_in_model}' non è nel foglio e non è stato fornito un valore manuale.")
                                       successful_mapping = False


                             if successful_mapping:
                                  # Verifica finale se tutte le feature_columns sono presenti in mapped_data
                                  missing_model_features = [col for col in feature_columns if col not in mapped_data.columns]
                                  if missing_model_features:
                                       st.error(f"Errore dopo la mappatura: mancano le seguenti colonne richieste dal modello: {', '.join(missing_model_features)}")
                                  else:
                                       # Riordina le colonne secondo feature_columns per sicurezza
                                       sim_data_input = mapped_data[feature_columns].values

                                       # Salva in session state per evitare ricaricamenti
                                       st.session_state.imported_sim_data = sim_data_input
                                       st.session_state.imported_sim_df_preview = mapped_data # Salva anche il df per preview

                                       st.success(f"Dati importati e mappati con successo ({sim_data_input.shape[0]} righe). Premi 'Esegui simulazione'.")

                         # Se sheet_data_cleaned è None, l'errore è già stato mostrato dalla funzione

            # Mostra anteprima se i dati sono stati importati con successo
            if 'imported_sim_data' in st.session_state and 'imported_sim_df_preview' in st.session_state:
                 st.subheader("Anteprima dei dati importati e mappati (ultime righe)")
                 # Mostra solo le colonne feature nell'ordine corretto + data
                 preview_cols = [column_mapping_gsheet_to_model.get(date_col_name_gsheet, date_col_name_gsheet)] + feature_columns
                 # Filtra colonne esistenti nel dataframe di preview
                 preview_cols = [col for col in preview_cols if col in st.session_state.imported_sim_df_preview.columns]
                 st.dataframe(st.session_state.imported_sim_df_preview[preview_cols].tail().round(3))
                 sim_data_input = st.session_state.imported_sim_data # Recupera i dati numpy per la simulazione


        # --- SIMULAZIONE: Inserimento Orario Dettagliato ---
        elif sim_method == 'Inserisci dati orari (Avanzato)':
            st.subheader(f'Inserisci dati specifici per ogni ora ({INPUT_WINDOW} ore precedenti)')
            st.warning("Questo metodo richiede l'inserimento manuale di molti valori.")

            # Inizializza/recupera i dati della simulazione da session_state
            if 'sim_hourly_data' not in st.session_state:
                 # Inizializza con valori medi o zero se df non disponibile
                 init_values = {}
                 for col in feature_columns:
                      init_values[col] = df[col].mean() if df is not None and col in df else 0.0
                 # Crea un dataframe con valori iniziali ripetuti
                 st.session_state.sim_hourly_data = pd.DataFrame(
                     np.repeat([list(init_values.values())], INPUT_WINDOW, axis=0),
                     columns=feature_columns,
                     index=[f"Ora T-{INPUT_WINDOW-i}" for i in range(INPUT_WINDOW)]
                 )
            else:
                 # Assicurati che abbia la dimensione corretta se cambia INPUT_WINDOW
                 if len(st.session_state.sim_hourly_data) != INPUT_WINDOW:
                      del st.session_state.sim_hourly_data # Resetta se la dimensione è cambiata
                      # Riprova a inizializzare
                      init_values = {}
                      for col in feature_columns:
                           init_values[col] = df[col].mean() if df is not None and col in df else 0.0
                      st.session_state.sim_hourly_data = pd.DataFrame(
                          np.repeat([list(init_values.values())], INPUT_WINDOW, axis=0),
                          columns=feature_columns,
                          index=[f"Ora T-{INPUT_WINDOW-i}" for i in range(INPUT_WINDOW)]
                      )


            # Usa st.data_editor per un inserimento tabellare
            st.caption(f"Modifica direttamente la tabella sottostante. Le colonne sono le features richieste dal modello. Le righe rappresentano le {INPUT_WINDOW} ore che precedono la simulazione (T-{INPUT_WINDOW} è la più vecchia).")

            edited_df = st.data_editor(
                st.session_state.sim_hourly_data,
                num_rows="dynamic", # Permette aggiunta/rimozione righe (anche se dovremmo limitarlo a INPUT_WINDOW)
                use_container_width=True,
                # Configura colonne per tipo numerico e limiti (opzionale ma utile)
                column_config={
                    col: st.column_config.NumberColumn(format="%.2f") for col in feature_columns
                    # Aggiungi limiti specifici se necessario, es:
                    # rain_features[0]: st.column_config.NumberColumn(min_value=0, max_value=200, format="%.1f"),
                    # humidity_feature[0]: st.column_config.NumberColumn(min_value=0, max_value=100, format="%.1f"),
                }
            )

            # Verifica se il numero di righe è corretto dopo l'editing
            if len(edited_df) != INPUT_WINDOW:
                 st.warning(f"Attenzione: la tabella contiene {len(edited_df)} righe, ma ne sono richieste {INPUT_WINDOW}. Verranno usate le ultime {INPUT_WINDOW} righe se ce ne sono di più, o verrà generato un errore se sono meno.")
                 if len(edited_df) > INPUT_WINDOW:
                      edited_df = edited_df.iloc[-INPUT_WINDOW:]
                 elif len(edited_df) < INPUT_WINDOW:
                      st.error(f"Numero di righe insufficiente ({len(edited_df)}). Inserisci esattamente {INPUT_WINDOW} righe.")
                      # Non procedere con la simulazione
                      edited_df = None # Invalida

            if edited_df is not None:
                 # Assicurati che le colonne siano nell'ordine corretto
                 try:
                      sim_data_input = edited_df[feature_columns].values
                      st.session_state.sim_hourly_data = edited_df # Salva modifiche
                      st.success("Dati orari pronti per la simulazione.")
                 except KeyError as ke:
                      st.error(f"Errore: colonna mancante dopo l'editing: {ke}. Assicurati che tutte le colonne {feature_columns} siano presenti.")
                      sim_data_input = None # Invalida



        # --- ESECUZIONE DELLA SIMULAZIONE ---
        st.divider()
        run_simulation = st.button('Esegui simulazione', type="primary", disabled=(sim_data_input is None))

        if run_simulation and sim_data_input is not None:
             # Verifica finale dimensioni prima di chiamare predict
             if sim_data_input.shape[0] != INPUT_WINDOW or sim_data_input.shape[1] != len(feature_columns):
                  st.error(f"Errore dimensionale nei dati di input per la simulazione. Atteso: ({INPUT_WINDOW}, {len(feature_columns)}), Ottenuto: {sim_data_input.shape}")
             else:
                  st.info(f"Esecuzione previsione con dati di input shape: {sim_data_input.shape}")
                  with st.spinner('Simulazione in corso...'):
                       # Previsione
                       predictions_sim = predict(model, sim_data_input, scaler_features, scaler_targets, target_columns, device, OUTPUT_WINDOW)

                       if predictions_sim is not None:
                           # Visualizzazione dei risultati
                           st.subheader(f'Risultato Simulazione: Previsione per le prossime {OUTPUT_WINDOW} ore')

                           # Creazione dataframe risultati
                           current_time_sim = datetime.now() # Ora di esecuzione simulazione
                           prediction_times_sim = [current_time_sim + timedelta(hours=i+1) for i in range(OUTPUT_WINDOW)]
                           results_df_sim = pd.DataFrame(predictions_sim, columns=target_columns)
                           results_df_sim['Ora previsione'] = prediction_times_sim
                           # Formatta ora
                           results_df_sim['Ora previsione'] = results_df_sim['Ora previsione'].dt.strftime('%d/%m %H:%M')
                           results_df_sim = results_df_sim[['Ora previsione'] + target_columns]


                           # Visualizzazione tabella risultati
                           st.dataframe(results_df_sim.round(3))

                           # Download dei risultati
                           st.markdown(get_table_download_link(results_df_sim, filename=f"simulazione_{current_time_sim.strftime('%Y%m%d_%H%M')}.csv"), unsafe_allow_html=True)

                           # Grafici Plotly per ogni sensore target
                           st.subheader('Grafici delle previsioni simulate')
                           figs_sim = plot_predictions(predictions_sim, target_columns, OUTPUT_WINDOW, current_time_sim)

                           for i, fig_sim in enumerate(figs_sim):
                               sensor_name_sim_safe = target_columns[i].replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
                               st.plotly_chart(fig_sim, use_container_width=True)
                               st.markdown(
                                   get_plotly_download_link(fig_sim, f"grafico_sim_{sensor_name_sim_safe}_{current_time_sim.strftime('%Y%m%d_%H%M')}", text_html="Scarica Grafico Interattivo (HTML)", text_png="Scarica Immagine (PNG)"),
                                   unsafe_allow_html=True
                               )
        elif run_simulation and sim_data_input is None:
             st.error("Dati di input per la simulazione non pronti o non validi. Completa la preparazione dei dati.")


elif page == 'Analisi Dati Storici':
    if disable_analysis:
         st.warning("Funzionalità non disponibile: carica correttamente i dati storici (CSV).")
    else:
        # --- CODICE ANALISI DATI ---
        st.header('Analisi Dati Storici')
        st.info(f"Dataset caricato: {len(df)} righe, dal {df[date_col_name_csv].min().strftime('%d/%m/%Y')} al {df[date_col_name_csv].max().strftime('%d/%m/%Y')}")

        # Selezione del range temporale
        st.subheader('Seleziona il periodo di analisi')

        # Otteniamo il range di date disponibili
        min_date = df[date_col_name_csv].min().date() # Prendi solo la data
        max_date = df[date_col_name_csv].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Data inizio', min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input('Data fine', max_date, min_value=min_date, max_value=max_date)

        # Validazione range date
        if start_date > end_date:
             st.error("La data di inizio non può essere successiva alla data di fine.")
        else:
             # Filtraggio dei dati in base al range selezionato (considera l'intera giornata)
             mask = (df[date_col_name_csv].dt.date >= start_date) & (df[date_col_name_csv].dt.date <= end_date)
             filtered_df = df.loc[mask]

             if len(filtered_df) == 0:
                 st.warning("Nessun dato trovato nel periodo selezionato.")
             else:
                 st.success(f"Trovati {len(filtered_df)} record nel periodo dal {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.")

                 tab1, tab2, tab3 = st.tabs(["Andamento Temporale", "Statistiche e Distribuzione", "Correlazione"])

                 with tab1:
                     st.subheader('Andamento temporale')
                     # Selezione multipla delle feature da plottare
                     features_to_plot = st.multiselect(
                         'Seleziona una o più features da visualizzare',
                         feature_columns, # Usa tutte le colonne numeriche disponibili
                         default=target_columns[:1] if target_columns else feature_columns[:1] # Default al primo target o prima feature
                     )

                     if features_to_plot:
                          # Crea grafico Plotly interattivo
                          fig_time = go.Figure()
                          for feature in features_to_plot:
                               fig_time.add_trace(go.Scatter(x=filtered_df[date_col_name_csv], y=filtered_df[feature], mode='lines', name=feature))

                          fig_time.update_layout(
                               title='Andamento Temporale Features Selezionate',
                               xaxis_title='Data e Ora',
                               yaxis_title='Valore',
                               height=500,
                               hovermode="x unified"
                          )
                          st.plotly_chart(fig_time, use_container_width=True)
                     else:
                          st.info("Seleziona almeno una feature da visualizzare.")

                 with tab2:
                     st.subheader('Statistiche descrittive e Distribuzione')

                     # Selezione della feature da analizzare
                     feature_to_analyze_stat = st.selectbox(
                         'Seleziona la feature per statistiche e distribuzione',
                         feature_columns,
                         index = feature_columns.index(target_columns[0]) if target_columns else 0 # Default al primo target
                     )

                     # Statistiche descrittive
                     st.write(f"**Statistiche per '{feature_to_analyze_stat}'**")
                     stats = filtered_df[feature_to_analyze_stat].describe()
                     stats_df = pd.DataFrame(stats).transpose()
                     # Rinomina indici per chiarezza
                     stats_df.rename(index={'count': 'Numero Valori', 'mean': 'Media', 'std': 'Dev. Standard', 'min': 'Minimo', '25%': '1° Quartile', '50%': 'Mediana', '75%': '3° Quartile', 'max': 'Massimo'}, inplace=True)
                     st.dataframe(stats_df.round(3))

                     # Istogramma della distribuzione (Plotly)
                     st.write(f"**Distribuzione di '{feature_to_analyze_stat}'**")
                     fig_hist = go.Figure()
                     fig_hist.add_trace(go.Histogram(x=filtered_df[feature_to_analyze_stat], name='Distribuzione'))

                     # Aggiungi linea della media
                     mean_val = filtered_df[feature_to_analyze_stat].mean()
                     fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Media: {mean_val:.2f}")

                     fig_hist.update_layout(
                          title=f'Distribuzione Valori per {feature_to_analyze_stat}',
                          xaxis_title='Valore',
                          yaxis_title='Frequenza (Conteggio)',
                          height=400
                     )
                     st.plotly_chart(fig_hist, use_container_width=True)


                 with tab3:
                     st.subheader('Analisi di correlazione')
                     st.write("Mostra la correlazione lineare (di Pearson) tra le features selezionate.")
                     # Selezione delle features per la correlazione
                     corr_features_select = st.multiselect(
                         'Seleziona le features per la matrice di correlazione',
                         feature_columns,
                         default=target_columns + rain_features[:1] if target_columns else feature_columns[:min(5, len(feature_columns))] # Default: targets + 1 pioggia o prime 5
                     )

                     if len(corr_features_select) > 1:
                         # Calcolo della matrice di correlazione
                         corr_matrix = filtered_df[corr_features_select].corr()

                         # Visualizzazione della heatmap (Plotly)
                         fig_heatmap = go.Figure(data=go.Heatmap(
                             z=corr_matrix.values,
                             x=corr_matrix.columns,
                             y=corr_matrix.columns,
                             colorscale='RdBu', # Scala di colori Rosso-Blu
                             zmin=-1, zmax=1,  # Forza range da -1 a 1
                             text=corr_matrix.round(2).values, # Mostra valori sulla heatmap
                             texttemplate="%{text}",
                             hoverongaps = False))

                         fig_heatmap.update_layout(
                              title='Matrice di Correlazione di Pearson',
                              height=600 if len(corr_features_select) > 5 else 500
                         )
                         st.plotly_chart(fig_heatmap, use_container_width=True)

                         # Scatterplot per coppie selezionate (opzionale)
                         st.write("**Scatter Plot tra due Features**")
                         col_scatter1, col_scatter2 = st.columns(2)
                         with col_scatter1:
                              feat1_scatter = st.selectbox("Feature Asse X", corr_features_select, index=0)
                         with col_scatter2:
                              # Filtra lista per non selezionare la stessa variabile
                              options_y = [f for f in corr_features_select if f != feat1_scatter]
                              if options_y:
                                   feat2_scatter = st.selectbox("Feature Asse Y", options_y, index=0)
                              else:
                                   feat2_scatter = None # Se c'è solo una feature selezionata

                         if feat1_scatter and feat2_scatter:
                              fig_scatter = go.Figure(data=go.Scatter(
                                   x=filtered_df[feat1_scatter],
                                   y=filtered_df[feat2_scatter],
                                   mode='markers',
                                   marker=dict(opacity=0.6) # Punti leggermente trasparenti
                              ))
                              fig_scatter.update_layout(
                                   title=f'Scatter Plot: {feat1_scatter} vs {feat2_scatter}',
                                   xaxis_title=feat1_scatter,
                                   yaxis_title=feat2_scatter,
                                   height=500
                              )
                              st.plotly_chart(fig_scatter, use_container_width=True)

                     else:
                         st.info("Seleziona almeno due features per calcolare la correlazione.")

                 # Download dei dati filtrati
                 st.divider()
                 st.subheader('Download Dati Filtrati')
                 st.markdown(get_table_download_link(filtered_df, f"dati_filtrati_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"), unsafe_allow_html=True)


elif page == 'Allenamento Modello':
    if disable_training:
         st.warning("Funzionalità non disponibile: carica correttamente i dati storici (CSV).")
    else:
        # --- CODICE ALLENAMENTO MODELLO ---
        st.header('Allenamento Nuovo Modello LSTM')
        st.warning("L'allenamento può richiedere tempo e risorse computazionali significative.")
        st.info(f"Dati disponibili per l'addestramento: {len(df)} righe.")

        # Opzioni di addestramento
        st.subheader('Configurazione dell\'addestramento')

        # Selezione target (quali idrometri prevedere)
        st.write("**1. Seleziona gli idrometri target da prevedere:**")
        selected_targets_train = []
        cols_targets = st.columns(len(hydro_features))
        for i, feature in enumerate(hydro_features):
            with cols_targets[i]:
                 # Usa solo il nome breve come etichetta
                 label = feature.split('(')[-1].replace(')', '').strip()
                 if st.checkbox(label, value=(feature in target_columns), key=f"target_{i}"): # Default ai target correnti
                     selected_targets_train.append(feature)

        if not selected_targets_train:
            st.error("Seleziona almeno un idrometro target da prevedere.")
            st.stop() # Ferma se nessun target è selezionato
        else:
            st.write(f"Idrometri selezionati per la previsione: {', '.join(selected_targets_train)}")


        # Parametri del modello e dell'addestramento
        st.write("**2. Imposta i parametri del modello e dell'addestramento:**")
        with st.expander("Parametri Finestre Temporali, Modello e Ottimizzazione", expanded=True):
             col1_train, col2_train, col3_train = st.columns(3)

             with col1_train:
                 st.markdown("##### Finestre Temporali")
                 input_window_train = st.number_input("Input Window (ore)", min_value=6, max_value=168, value=INPUT_WINDOW, step=6, help="Quante ore passate usare per prevedere.")
                 output_window_train = st.number_input("Output Window (ore)", min_value=1, max_value=72, value=OUTPUT_WINDOW, step=1, help="Quante ore future prevedere.")
                 val_split_train = st.slider("% Dati di Validazione", 5, 40, 20, step=1, help="Percentuale di dati più recenti usata per validare il modello durante l'allenamento.")

             with col2_train:
                 st.markdown("##### Architettura LSTM")
                 hidden_size_train_cfg = st.number_input("Dimensione Hidden Layer", min_value=16, max_value=1024, value=hidden_size, step=16, key="hidden_train", help="Numero di neuroni negli strati LSTM.")
                 num_layers_train_cfg = st.number_input("Numero Layer LSTM", min_value=1, max_value=8, value=num_layers, step=1, key="layers_train", help="Numero di strati LSTM impilati.")
                 dropout_train_cfg = st.slider("Dropout", 0.0, 0.7, dropout, 0.05, key="dropout_train_slider", help="Probabilità di 'spegnere' neuroni per prevenire overfitting.")

             with col3_train:
                 st.markdown("##### Ottimizzazione")
                 learning_rate_train = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=0.001, format="%.5f", step=1e-4, help="Tasso di apprendimento dell'ottimizzatore Adam.")
                 batch_size_train = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128, 256], value=32, help="Numero di sequenze processate in parallelo per aggiornamento pesi.")
                 epochs_train = st.number_input("Numero di Epoche", min_value=5, max_value=500, value=50, step=5, help="Numero di volte che l'intero dataset di training viene mostrato al modello.")


        # Bottone per avviare l'addestramento
        st.write("**3. Avvia l'addestramento:**")
        train_button = st.button("Addestra Nuovo Modello", type="primary")

        if train_button:
             st.info("Inizio processo di addestramento...")

             # --- Preparazione Dati Training ---
             with st.spinner('Preparazione dati di training e validazione...'):
                 X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train = prepare_training_data(
                     df.copy(), # Usa una copia del dataframe per sicurezza
                     feature_columns, # Usa sempre tutte le feature come input
                     selected_targets_train, # Usa solo i target selezionati
                     input_window_train,
                     output_window_train,
                     val_split_train
                 )

                 if X_train is None: # Controllo se prepare_training_data ha fallito
                      st.error("Preparazione dati fallita. Controlla i log precedenti.")
                      st.stop()

                 st.success(f"Dati preparati: {len(X_train)} seq. training, {len(X_val)} seq. validazione.")
                 st.write(f"Shape Input (X_train): `{X_train.shape}` -> (Num Sequenze, Input Window, Num Features)")
                 st.write(f"Shape Output (y_train): `{y_train.shape}` -> (Num Sequenze, Output Window, Num Target = {len(selected_targets_train)})")

             # --- Addestramento Modello ---
             st.subheader("Addestramento Modello in corso...")
             st.write(f"Parametri: Hidden={hidden_size_train_cfg}, Layers={num_layers_train_cfg}, Dropout={dropout_train_cfg}, LR={learning_rate_train}, Batch={batch_size_train}")

             # Calcolo dimensioni input/output
             input_size_train = len(feature_columns) # Numero totale di features in input
             output_size_train = len(selected_targets_train) # Numero di target selezionati

             # Chiamata alla funzione di addestramento
             try:
                 trained_model, train_losses, val_losses = train_model(
                     X_train, y_train, X_val, y_val,
                     input_size_train, output_size_train, output_window_train, # Passa le dimensioni corrette
                     hidden_size_train_cfg,
                     num_layers_train_cfg,
                     epochs_train,
                     batch_size_train,
                     learning_rate_train,
                     dropout_train_cfg
                 )
             except Exception as e_train:
                  st.error(f"Errore durante l'addestramento: {type(e_train).__name__} - {e_train}")
                  import traceback
                  st.error(traceback.format_exc())
                  st.stop()


             # --- Risultati e Salvataggio ---
             if trained_model:
                 st.success("Addestramento completato!")

                 # Grafico della loss finale (non interattivo, mostra solo alla fine)
                 # Il grafico interattivo è già stato mostrato durante l'allenamento
                 st.subheader("Grafico Finale della Loss (Statico)")
                 fig_loss_final, ax_loss_final = plt.subplots(figsize=(10, 5))
                 ax_loss_final.plot(range(1, epochs_train + 1), train_losses, label='Train Loss')
                 ax_loss_final.plot(range(1, epochs_train + 1), val_losses, label='Validation Loss')
                 ax_loss_final.set_title('Andamento Loss Finale')
                 ax_loss_final.set_xlabel('Epoca')
                 ax_loss_final.set_ylabel('Loss (MSE)')
                 ax_loss_final.legend()
                 ax_loss_final.grid(True)
                 st.pyplot(fig_loss_final)

                 st.subheader("Download del Modello Addestrato e degli Scaler")
                 st.info("Salva questi 3 file per poter usare il modello in futuro.")

                 # Crea i buffer in memoria per il download
                 model_buffer = io.BytesIO()
                 torch.save(trained_model.state_dict(), model_buffer)

                 scaler_features_buffer = io.BytesIO()
                 joblib.dump(scaler_features_train, scaler_features_buffer)

                 scaler_targets_buffer = io.BytesIO()
                 joblib.dump(scaler_targets_train, scaler_targets_buffer)

                 # Crea i link per il download
                 col_dl1, col_dl2, col_dl3 = st.columns(3)
                 with col_dl1:
                     st.markdown(get_binary_file_download_link(model_buffer, "trained_hydro_model.pth", "Scarica Modello (.pth)"), unsafe_allow_html=True)
                 with col_dl2:
                     st.markdown(get_binary_file_download_link(scaler_features_buffer, "trained_scaler_features.joblib", "Scarica Scaler Features (.joblib)"), unsafe_allow_html=True)
                 with col_dl3:
                    st.markdown(get_binary_file_download_link(scaler_targets_buffer, "trained_scaler_targets.joblib", "Scarica Scaler Target (.joblib)"), unsafe_allow_html=True)


                 # Info sul modello addestrato
                 st.subheader("Informazioni sul Modello Addestrato")
                 model_info = {
                     "Data Addestramento": datetime.now().strftime("%d/%m/%Y %H:%M"),
                     "Basato su Dati": f"{len(df)} righe",
                     "Target Previsti": selected_targets_train,
                     "Input Window (ore)": input_window_train,
                     "Output Window (ore)": output_window_train,
                     "% Validazione": val_split_train,
                     "Architettura": {
                         "Tipo": "LSTM",
                         "Input Size (Features)": input_size_train,
                         "Output Size (Targets)": output_size_train,
                         "Hidden Size": hidden_size_train_cfg,
                         "Num Layers": num_layers_train_cfg,
                         "Dropout": dropout_train_cfg,
                     },
                     "Training Params": {
                         "Epoche": epochs_train,
                         "Batch Size": batch_size_train,
                         "Learning Rate Iniziale": learning_rate_train,
                         "Loss Function": "MSELoss",
                         "Optimizer": "Adam"
                     },
                     "Performance Finale": {
                          "Min Validation Loss": min(val_losses) if val_losses else "N/A"
                     }
                 }
                 st.json(model_info) # Mostra le info come JSON

                 # Test rapido su un campione di validazione (opzionale)
                 if X_val.size > 0:
                     st.subheader("Test Rapido su Dati di Validazione")
                     if st.button("Esegui Test su Campione Casuale"):
                          with st.spinner("Esecuzione test..."):
                              # Prendiamo un esempio casuale dal set di validazione
                              sample_idx = np.random.randint(0, len(X_val))
                              sample_input_scaled = X_val[sample_idx]
                              sample_target_scaled = y_val[sample_idx]

                              # Previsione sul campione
                              sample_input_tensor = torch.FloatTensor(sample_input_scaled).unsqueeze(0).to(device)
                              trained_model.eval()
                              with torch.no_grad():
                                   sample_output_scaled_tensor = trained_model(sample_input_tensor)

                              # Denormalizzazione
                              sample_output_scaled = sample_output_scaled_tensor.cpu().numpy().reshape(output_window_train, output_size_train)
                              sample_target_original = scaler_targets_train.inverse_transform(sample_target_scaled)
                              sample_output_original = scaler_targets_train.inverse_transform(sample_output_scaled)

                              # Crea dataframe per confronto
                              test_results_df = pd.DataFrame()
                              hours = [f"Ora +{h+1}" for h in range(output_window_train)]
                              test_results_df['Ora Futura'] = hours
                              for i, target_name in enumerate(selected_targets_train):
                                   test_results_df[f'Reale: {target_name}'] = sample_target_original[:, i]
                                   test_results_df[f'Previsto: {target_name}'] = sample_output_original[:, i]

                              st.dataframe(test_results_df.round(3))

                              # Grafico confronto Plotly
                              fig_test = go.Figure()
                              for i, target_name in enumerate(selected_targets_train):
                                   fig_test.add_trace(go.Scatter(x=hours, y=sample_target_original[:, i], mode='lines+markers', name=f'Reale: {target_name}'))
                                   fig_test.add_trace(go.Scatter(x=hours, y=sample_output_original[:, i], mode='lines+markers', name=f'Previsto: {target_name}', line=dict(dash='dash')))

                              fig_test.update_layout(
                                   title=f'Confronto Reale vs Previsto (Campione Validazione #{sample_idx})',
                                   xaxis_title='Ore Future',
                                   yaxis_title='Livello Idrometrico [m]',
                                   height=500,
                                   hovermode="x unified"
                              )
                              st.plotly_chart(fig_test, use_container_width=True)

             else:
                  st.error("L'addestramento non ha prodotto un modello valido.")


# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Applicazione Streamlit per Modello Idrologico LSTM © 2024')

# --- FINE LAYOUT STREAMLIT ---
