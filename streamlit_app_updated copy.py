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

# Configurazione della pagina
st.set_page_config(page_title="Modello Predittivo Idrologico", page_icon="🌊", layout="wide")

# Costanti globali
INPUT_WINDOW = 24  # Finestra di input di 24 ore
OUTPUT_WINDOW = 12  # Previsione per le prossime 12 ore

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
        scaler_features = joblib.load(scaler_features_path)
        scaler_targets = joblib.load(scaler_targets_path)
        return scaler_features, scaler_targets
    except Exception as e:
        st.error(f"Errore durante il caricamento degli scaler: {e}")
        return None, None # Return None scalers

# Funzione per preparare i dati per l'addestramento
def prepare_training_data(df, feature_columns, target_columns, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, val_split=20):
    # Creazione delle sequenze di input (X) e output (y)
    X, y = [], []
    for i in range(len(df) - input_window - output_window + 1):
        X.append(df.iloc[i:i+input_window][feature_columns].values)
        y.append(df.iloc[i+input_window:i+input_window+output_window][target_columns].values)
    X = np.array(X)
    y = np.array(y)

    # Normalizzazione dei dati
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()
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
    loss_chart = st.empty()

    # Creazione del grafico interattivo di perdita
    def update_loss_chart(train_losses, val_losses):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
        fig.update_layout(
            title='Andamento della perdita',
            xaxis_title='Epoca',
            yaxis_title='Loss',
            height=400
        )
        loss_chart.plotly_chart(fig, use_container_width=True)

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
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Aggiornamento del grafico di perdita
        update_loss_chart(train_losses, val_losses)

        # Salvataggio del modello migliore
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()

        # Breve pausa per consentire l'aggiornamento dell'interfaccia
        time.sleep(0.1)

    # Caricamento del modello migliore
    model.load_state_dict(best_model)

    return model, train_losses, val_losses

# Funzione per fare previsioni
def predict(model, input_data, scaler_features, scaler_targets, target_columns, device, output_window):
    """
    Funzione per fare previsioni con il modello addestrato.

    Args:
        model: Il modello addestrato
        input_data: Dati di input non normalizzati (array di forma [input_window, num_features])
        scaler_features: Scaler per normalizzare i dati di input
        scaler_targets: Scaler per denormalizzare le previsioni
        target_columns: Nomi dei target (sensori idrometrici)
        device: Dispositivo (CPU/GPU)
        output_window: Numero di timestep futuri da prevedere

    Returns:
        Previsioni denormalizzate
    """
    if model is None or scaler_features is None or scaler_targets is None:
        st.error("Modello o scaler non caricati correttamente. Impossibile fare previsioni.")
        return None

    model.eval()

    # Normalizzazione dei dati di input
    input_normalized = scaler_features.transform(input_data)

    # Conversione in tensore PyTorch
    input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)

    # Previsione
    with torch.no_grad():
        output = model(input_tensor)

    # Conversione in numpy
    output_np = output.cpu().numpy().reshape(-1, len(target_columns))

    # Denormalizzazione
    predictions = scaler_targets.inverse_transform(output_np)

    # Reshape per ottenere [output_window, num_target_columns]
    predictions = predictions.reshape(output_window, len(target_columns))

    return predictions

# Funzione per plot dei risultati
def plot_predictions(predictions, target_columns, output_window, start_time=None):
    figures = []

    # Per ogni sensore idrometrico target
    for i, sensor_name in enumerate(target_columns):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Creazione dell'asse x per le ore future
        if start_time:
            hours = [start_time + timedelta(hours=h) for h in range(output_window)]
            ax.plot(hours, predictions[:, i], marker='o', linestyle='-', label=f'Previsione {sensor_name}')
            plt.gcf().autofmt_xdate()
        else:
            hours = np.arange(output_window)
            ax.plot(hours, predictions[:, i], marker='o', linestyle='-', label=f'Previsione {sensor_name}')
            ax.set_xlabel('Ore future')

        ax.set_title(f'Previsione - {sensor_name}')
        ax.set_ylabel('Livello idrometrico [m]')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        figures.append(fig)

    return figures

# Funzione per ottenere un link di download per un file
def get_table_download_link(df, filename="previsioni.csv"):
    """Genera un link per scaricare il dataframe come file CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Scarica i dati CSV</a>'

# Funzione per ottenere un link di download per un file pkl/joblib
def get_binary_file_download_link(file_object, filename, text):
    """Genera un link per scaricare un file binario"""
    b64 = base64.b64encode(file_object.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

# Funzione per scaricare grafici
def get_image_download_link(fig, filename, text):
    """Genera un link per scaricare il grafico come immagine"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

# Titolo dell'app
st.title('Dashboard Modello Predittivo Idrologico')

# Sidebar per le opzioni
st.sidebar.header('Impostazioni')

# Opzione per caricare i propri file o usare quelli demo
use_demo_files = st.sidebar.checkbox('Usa file di esempio', value=True)

if use_demo_files:
    # Qui dovresti fornire percorsi ai file di esempio
    DATA_PATH = 'dati_idro.csv'  # Sostituisci con il percorso corretto
    MODEL_PATH = 'best_hydro_model.pth'  # Sostituisci con il percorso corretto
    SCALER_FEATURES_PATH = 'scaler_features.joblib'  # Sostituisci con il percorso corretto
    SCALER_TARGETS_PATH = 'scaler_targets.joblib'  # Sostituisci con il percorso corretto
    # Parametri del modello DEMO
    DEMO_HIDDEN_SIZE = 128
    DEMO_NUM_LAYERS = 2
    DEMO_DROPOUT = 0.2
else:
    # Caricamento dei file dall'utente
    st.sidebar.subheader('Carica i tuoi file')
    data_file = st.sidebar.file_uploader('File CSV con i dati storici', type=['csv'])
    model_file = st.sidebar.file_uploader('File del modello (.pth)', type=['pth'])
    scaler_features_file = st.sidebar.file_uploader('File scaler features (.joblib)', type=['joblib'])
    scaler_targets_file = st.sidebar.file_uploader('File scaler targets (.joblib)', type=['joblib'])

    # Configurazione parametri modello
    st.sidebar.subheader('Configurazione Modello')
    hidden_size = st.sidebar.number_input("Dimensione hidden layer", min_value=16, max_value=512, value=128, step=16)
    num_layers = st.sidebar.number_input("Numero di layer LSTM", min_value=1, max_value=5, value=2)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, 0.05)


    # Controllo se tutti i file sono stati caricati
    if not use_demo_files and not (data_file and model_file and scaler_features_file and scaler_targets_file):
        st.sidebar.warning('Carica tutti i file necessari per procedere')
    elif not use_demo_files:
        # Salvataggio temporaneo dei file caricati
        DATA_PATH = data_file
        MODEL_PATH = model_file
        SCALER_FEATURES_PATH = scaler_features_file
        SCALER_TARGETS_PATH = scaler_targets_file

# Estrazione delle caratteristiche (colonne del dataframe)
rain_features = [
    'Cumulata Sensore 1295 (Arcevia)',
    'Cumulata Sensore 2637 (Bettolelle)',
    'Cumulata Sensore 2858 (Barbara)',
    'Cumulata Sensore 2964 (Corinaldo)'
]

humidity_feature = ['Umidita\' Sensore 3452 (Montemurello)']

hydro_features = [
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
]

feature_columns = rain_features + humidity_feature + hydro_features


# Caricamento dei dati storici
df = None  # Initialize df to None
try:
    if use_demo_files:
        df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
    elif data_file is not None:  # Check if data_file is loaded
        df = pd.read_csv(data_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
    if df is not None:
        st.sidebar.success(f'Dati caricati: {len(df)} righe')
except Exception as e:
    st.sidebar.error(f'Errore nel caricamento dei dati: {e}')

# Caricamento del modello e degli scaler
model = None
device = None
scaler_features = None
scaler_targets = None

if use_demo_files or (data_file and model_file and scaler_features_file and scaler_targets_file):
    try:
        # Calcola input_size correttamente in base alle feature columns
        input_size = len(feature_columns)

        if use_demo_files:
            demo_target_columns = hydro_features[:4] # Use only first 4 for demo model
            output_size = len(demo_target_columns) # output_size = 4 for demo model
            target_columns = demo_target_columns # Update target_columns to be used in the app for demo mode
            # Usa parametri DEMO
            model, device = load_model(MODEL_PATH, input_size, DEMO_HIDDEN_SIZE, output_size, OUTPUT_WINDOW, DEMO_NUM_LAYERS, DEMO_DROPOUT)
            scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
        else:
            target_columns = hydro_features # Use all 5 hydro features for user uploaded model
            output_size = len(target_columns) # output_size = 5 for user-uploaded model
            # Usa parametri configurati dall'utente
            model, device = load_model(model_file, input_size, hidden_size, output_size, OUTPUT_WINDOW, num_layers, dropout)
            scaler_features, scaler_targets = load_scalers(scaler_features_file, scaler_targets_file)

        if model is not None and scaler_features is not None and scaler_targets is not None:
            st.sidebar.success('Modello e scaler caricati con successo')
    except Exception as e:
        st.sidebar.error(f'Errore nel caricamento del modello o degli scaler: {e}')

# Menu principale
st.sidebar.header('Menu')
page = st.sidebar.radio('Scegli una funzionalità',
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello'])

if page == 'Dashboard':
    st.header('Dashboard Idrologica')

    if df is None or model is None or scaler_features is None or scaler_targets is None:
        st.warning("Attenzione: Alcuni file necessari non sono stati caricati correttamente. Alcune funzionalità potrebbero non essere disponibili.")
    else:
        # Mostra ultimi dati disponibili
        st.subheader('Ultimi dati disponibili')
        last_data = df.iloc[-1]
        last_date = last_data['Data e Ora']

        # Formattazione dei dati per la visualizzazione
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Data e ora ultimo rilevamento:** {last_date}")

            # Ultimi dati idrologici
            st.subheader('Livelli idrometrici attuali')
            hydro_data = []
            for feature in hydro_features:
                if feature in target_columns: # Only display target hydro features
                    hydro_data.append({'Sensore': feature, 'Valore [m]': last_data[feature]})
            st.dataframe(pd.DataFrame(hydro_data).round(3))

        with col2:
            # Ultimi dati di pioggia
            st.subheader('Precipitazioni cumulate attuali')
            rain_data = []
            for feature in rain_features:
                rain_data.append({'Sensore': feature, 'Valore [mm]': last_data[feature]})
            st.dataframe(pd.DataFrame(rain_data).round(2))

            # Umidità
            st.subheader('Umidità attuale')
            st.write(f"{humidity_feature[0]}: {last_data[humidity_feature[0]]:.1f}%")

        # Previsione basata sugli ultimi dati disponibili
        st.header('Previsione in base agli ultimi dati')

        if st.button('Genera previsione'):
            if model is None or scaler_features is None or scaler_targets is None:
                st.error("Modello o scaler non caricati correttamente. Impossibile generare la previsione.")
            else:
                with st.spinner('Generazione previsione in corso...'):
                    # Preparazione dei dati di input (ultime INPUT_WINDOW ore)
                    latest_data = df.iloc[-INPUT_WINDOW:][feature_columns].values

                    # Previsione
                    predictions = predict(model, latest_data, scaler_features, scaler_targets, target_columns, device, OUTPUT_WINDOW)

                    if predictions is not None:  # Check if prediction was successful
                        # Visualizzazione dei risultati
                        st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore')

                        # Creazione dataframe risultati
                        start_time = last_date
                        prediction_times = [start_time + timedelta(hours=i) for i in range(OUTPUT_WINDOW)]
                        results_df = pd.DataFrame(predictions, columns=target_columns)
                        results_df['Ora previsione'] = prediction_times
                        results_df = results_df[['Ora previsione'] + target_columns]

                        # Visualizzazione tabella risultati
                        st.dataframe(results_df.round(3))

                        # Download dei risultati
                        st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)

                        # Grafici per ogni sensore
                        st.subheader('Grafici delle previsioni')
                        figs = plot_predictions(predictions, target_columns, OUTPUT_WINDOW, start_time)

                        # Visualizzazione grafici
                        for i, fig in enumerate(figs):
                            st.pyplot(fig)
                            sensor_name = target_columns[i].replace(' ', '_').replace('/', '_')
                            st.markdown(get_image_download_link(fig, f"{sensor_name}.png", f"Scarica il grafico di {target_columns[i]}"), unsafe_allow_html=True)

elif page == 'Simulazione':
    st.header('Simulazione Idrologica')
    st.write('Inserisci i valori per simulare uno scenario idrologico')

    if df is None or model is None or scaler_features is None or scaler_targets is None:
        st.warning("Attenzione: Alcuni file necessari non sono stati caricati correttamente. La simulazione potrebbe non funzionare.")
    else:
        # Opzioni per la simulazione
        sim_method = st.radio(
            "Metodo di simulazione",
            ['Inserisci dati orari', 'Modifica dati recenti', 'Inserisci manualmente tutti i valori']
        )

        if sim_method == 'Inserisci dati orari':
            st.subheader(f'Inserisci dati per ogni ora ({INPUT_WINDOW} ore precedenti)')

            # Creiamo un dataframe vuoto per i dati della simulazione
            sim_data = np.zeros((INPUT_WINDOW, len(feature_columns)))

            # Opzioni per la compilazione rapida (pioggia e umidità)
            st.subheader("Strumenti di compilazione rapida")
            quick_fill_col1, quick_fill_col2 = st.columns(2)

            with quick_fill_col1:
                # Opzioni per compilare rapidamente pioggia
                st.write("Scenario di pioggia")
                rain_scenario = st.selectbox(
                    "Seleziona uno scenario predefinito",
                    ["Nessuna pioggia", "Pioggia leggera", "Pioggia moderata", "Pioggia intensa", "Evento estremo"]
                )

                rain_values = {
                    "Nessuna pioggia": 0.0,
                    "Pioggia leggera": 2.0,
                    "Pioggia moderata": 5.0,
                    "Pioggia intensa": 10.0,
                    "Evento estremo": 20.0
                }

                rain_duration = st.slider("Durata pioggia (ore)", 0, INPUT_WINDOW, 3)
                rain_start = st.slider("Ora di inizio pioggia", 0, INPUT_WINDOW-1, 0)

                apply_rain = st.button("Applica scenario di pioggia")

            with quick_fill_col2:
                # Opzioni per compilare rapidamente umidità
                st.write("Umidità del terreno")
                humidity_preset = st.selectbox(
                    "Seleziona condizione di umidità",
                    ["Molto secco", "Secco", "Normale", "Umido", "Saturo"]
                )

                humidity_values = {
                    "Molto secco": 20.0,
                    "Secco": 40.0,
                    "Normale": 60.0,
                    "Umido": 80.0,
                    "Saturo": 95.0
                }

                apply_humidity = st.button("Applica umidità")

            # Creiamo tabs per separare i diversi tipi di dati
            data_tabs = st.tabs(["Cumulate Pioggia", "Umidità", "Idrometri"])

            # Utilizziamo session_state per mantenere i valori tra le interazioni
            if 'rain_data' not in st.session_state:
                st.session_state.rain_data = np.zeros((INPUT_WINDOW, len(rain_features)))
            if 'humidity_data' not in st.session_state:
                st.session_state.humidity_data = np.zeros((INPUT_WINDOW, len(humidity_feature)))
            if 'hydro_data' not in st.session_state:
                st.session_state.hydro_data = np.zeros((INPUT_WINDOW, len(hydro_features)))

            # Riferimenti più corti per maggiore leggibilità
            rain_data = st.session_state.rain_data
            humidity_data = st.session_state.humidity_data
            hydro_data = st.session_state.hydro_data

            # Se l'utente ha cliccato su applica scenario di pioggia
            if apply_rain:
                for h in range(rain_duration):
                    hour_idx = (rain_start + h) % INPUT_WINDOW
                    if hour_idx < INPUT_WINDOW:
                        for i in range(len(rain_features)):
                            rain_data[hour_idx, i] = rain_values[rain_scenario]

            # Se l'utente ha cliccato su applica umidità
            if apply_humidity:
                for h in range(INPUT_WINDOW):
                    humidity_data[h, 0] = humidity_values[humidity_preset]

            # Tab per la pioggia
            with data_tabs[0]:
                st.write("Inserisci i valori di pioggia cumulata per ogni ora (mm)")

                # Creiamo un layout a griglia per l'inserimento dei dati orari
                num_cols = 4  # Adatta se necessario
                num_rows = math.ceil(INPUT_WINDOW / num_cols)

                for feature_idx, feature in enumerate(rain_features):
                    st.write(f"### {feature}")

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"rain_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = rain_data[hour_idx, feature_idx]

                                    # Aggiorniamo il valore nel session_state se è stato modificato dallo scenario
                                    if apply_rain and hour_idx >= rain_start and hour_idx < rain_start + rain_duration:
                                        st.session_state[input_key] = rain_values[rain_scenario]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        0.0, 100.0,
                                        st.session_state[input_key], 0.5,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    rain_data[hour_idx, feature_idx] = value

            # Tab per l'umidità
            with data_tabs[1]:
                st.write("Inserisci i valori di umidità per ogni ora (%)")

                for feature_idx, feature in enumerate(humidity_feature):
                    st.write(f"### {feature}")

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"humidity_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = humidity_data[hour_idx, feature_idx]

                                    # Aggiorniamo il valore nel session_state se è stato modificato dallo scenario
                                    if apply_humidity:
                                        st.session_state[input_key] = humidity_values[humidity_preset]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        0.0, 100.0,
                                        st.session_state[input_key], 0.5,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    humidity_data[hour_idx, feature_idx] = value

            # Tab per gli idrometri
            with data_tabs[2]:
                st.write("Inserisci i livelli idrometrici per ogni ora (m)")

                for feature_idx, feature in enumerate(hydro_features):
                    st.write(f"### {feature}")

                    # Aggiungiamo un modo per impostare un valore costante
                    const_col1, const_col2 = st.columns([3, 1])

                    # Chiave univoca per ogni valore costante
                    const_key = f"const_hydro_{feature_idx}"
                    if const_key not in st.session_state:
                        st.session_state[const_key] = 0.0

                    with const_col1:
                        const_value = st.number_input(
                            f"Valore costante per {feature}",
                            -1.0, 10.0,
                            st.session_state[const_key],
                            0.01,
                            key=const_key
                        )

                    # Creiamo una funzione di callback per applicare il valore costante
                    def apply_constant_value(feature_idx=feature_idx, value=const_value):
                        for h in range(INPUT_WINDOW):
                            st.session_state.hydro_data[h, feature_idx] = value
                            input_key = f"hydro_{feature_idx}_{h}" # Generate input key
                            st.session_state[input_key] = value # Update input widget session state

                    with const_col2:
                        if st.button(f"Applica a tutte le ore", key=f"apply_const_hydro_{feature_idx}", on_click=apply_constant_value):
                            pass # Callback già gestisce l'aggiornamento

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"hydro_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = hydro_data[hour_idx, feature_idx]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        -1.0, 10.0,
                                        st.session_state[input_key], 0.01,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    hydro_data[hour_idx, feature_idx] = value

            # Componiamo i dati per la simulazione
            rain_offset = 0
            humidity_offset = len(rain_features)
            hydro_offset = humidity_offset + len(humidity_feature)

            for h in range(INPUT_WINDOW):
                # Copiamo i dati di pioggia
                for i in range(len(rain_features)):
                    sim_data[h, rain_offset + i] = rain_data[h, i]

                # Copiamo i dati di umidità
                for i in range(len(humidity_feature)):
                    sim_data[h, humidity_offset + i] = humidity_data[h, i]

                # Copiamo i dati degli idrometri
                for i in range(len(hydro_features)):
                    sim_data[h, hydro_offset + i] = hydro_data[h, i]

            # Visualizziamo un'anteprima dei dati
            st.subheader("Anteprima dei dati inseriti")
            preview_df = pd.DataFrame(sim_data, columns=feature_columns)
            preview_df.index = [f"Ora {i}" for i in range(INPUT_WINDOW)]
            st.dataframe(preview_df.round(2))

        elif sim_method == 'Modifica dati recenti':
            st.subheader('Modifica i dati recenti per la simulazione')

            # Controlla se ci sono dati recenti
            if df is None or len(df) < INPUT_WINDOW:
                st.error(f"Non ci sono abbastanza dati recenti. Servono almeno {INPUT_WINDOW} ore di dati.")
            else:
                # Ottieni gli ultimi INPUT_WINDOW dati
                last_data = df.iloc[-INPUT_WINDOW:].copy()

                # Creiamo un dataframe modificabile con i dati recenti
                st.write(f"Ultimi {INPUT_WINDOW} dati (modifica i valori se necessario)")

                # Estraiamo le date per riferimento
                dates = last_data['Data e Ora'].tolist()

                # Creiamo tabs per separare i diversi tipi di dati
                data_tabs = st.tabs(["Cumulate Pioggia", "Umidità", "Idrometri"])

                # Estraiamo i dati attuali
                rain_data_recent = last_data[rain_features].values
                humidity_data_recent = last_data[humidity_feature].values
                hydro_data_recent = last_data[hydro_features].values

                # Inizializziamo i dati modificati nel session_state se non esistono
                if 'rain_data_recent' not in st.session_state:
                    st.session_state.rain_data_recent = rain_data_recent.copy()
                if 'humidity_data_recent' not in st.session_state:
                    st.session_state.humidity_data_recent = humidity_data_recent.copy()
                if 'hydro_data_recent' not in st.session_state:
                    st.session_state.hydro_data_recent = hydro_data_recent.copy()

                # Tab per la pioggia
                with data_tabs[0]:
                    st.write("Modifica i valori di pioggia cumulata (mm)")

                    for feature_idx, feature in enumerate(rain_features):
                        st.write(f"### {feature}")

                        # Creiamo una tabella modificabile
                        for hour_idx in range(INPUT_WINDOW):
                            date_str = dates[hour_idx].strftime("%d/%m/%Y %H:%M")

                            # Chiave univoca per ogni input
                            input_key = f"rain_recent_{feature_idx}_{hour_idx}"

                            # Recupera il valore corrente o inizializza
                            if input_key not in st.session_state:
                                st.session_state[input_key] = st.session_state.rain_data_recent[hour_idx, feature_idx]

                            # Input numerico per modificare il valore
                            value = st.number_input(
                                f"{date_str}",
                                0.0, 100.0,
                                st.session_state[input_key], 0.5,
                                key=input_key
                            )

                            # Aggiorna il valore
                            st.session_state.rain_data_recent[hour_idx, feature_idx] = value

                # Tab per l'umidità
                with data_tabs[1]:
                    st.write("Modifica i valori di umidità (%)")

                    for feature_idx, feature in enumerate(humidity_feature):
                        st.write(f"### {feature}")

                        # Creiamo una tabella modificabile
                        for hour_idx in range(INPUT_WINDOW):
                            date_str = dates[hour_idx].strftime("%d/%m/%Y %H:%M")

                            # Chiave univoca per ogni input
                            input_key = f"humidity_recent_{feature_idx}_{hour_idx}"

                            # Recupera il valore corrente o inizializza
                            if input_key not in st.session_state:
                                st.session_state[input_key] = st.session_state.humidity_data_recent[hour_idx, feature_idx]

                            # Input numerico per modificare il valore
                            value = st.number_input(
                                f"{date_str}",
                                0.0, 100.0,
                                st.session_state[input_key], 0.5,
                                key=input_key
                            )

                            # Aggiorna il valore
                            st.session_state.humidity_data_recent[hour_idx, feature_idx] = value

                # Tab per gli idrometri
                with data_tabs[2]:
                    st.write("Modifica i livelli idrometrici (m)")

                    for feature_idx, feature in enumerate(hydro_features):
                        st.write(f"### {feature}")

                        # Creiamo una tabella modificabile
                        for hour_idx in range(INPUT_WINDOW):
                            date_str = dates[hour_idx].strftime("%d/%m/%Y %H:%M")

                            # Chiave univoca per ogni input
                            input_key = f"hydro_recent_{feature_idx}_{hour_idx}"

                            # Recupera il valore corrente o inizializza
                            if input_key not in st.session_state:
                                st.session_state[input_key] = st.session_state.hydro_data_recent[hour_idx, feature_idx]

                            # Input numerico per modificare il valore
                            value = st.number_input(
                                f"{date_str}",
                                -1.0, 10.0,
                                st.session_state[input_key], 0.01,
                                key=input_key
                            )

                            # Aggiorna il valore
                            st.session_state.hydro_data_recent[hour_idx, feature_idx] = value

                # Creiamo un dataframe modificato con i dati aggiornati
                modified_df = last_data.copy()
                modified_df[rain_features] = st.session_state.rain_data_recent
                modified_df[humidity_feature] = st.session_state.humidity_data_recent
                modified_df[hydro_features] = st.session_state.hydro_data_recent

                # Prepariamo i dati per la simulazione
                sim_data = modified_df[feature_columns].values

                # Visualizziamo un'anteprima dei dati modificati
                st.subheader("Anteprima dei dati modificati")
                preview_df = pd.DataFrame(sim_data, columns=feature_columns)
                preview_df.index = [date.strftime("%d/%m/%Y %H:%M") for date in dates]
                st.dataframe(preview_df.round(2))

        else:  # Inserimento manuale completo
            st.subheader('Inserisci valori per ogni parametro')

            # Creiamo un dataframe vuoto per i dati della simulazione
            sim_data = np.zeros((INPUT_WINDOW, len(feature_columns)))

            # Raggruppiamo i controlli per tipo di sensore
            with st.expander("Imposta valori di pioggia cumulata"):
                for i, feature in enumerate(rain_features):
                    value = st.number_input(f'{feature} (mm)', 0.0, 100.0, 0.0, 0.5)
                    sim_data[:, i] = value

            with st.expander("Imposta valore di umidità"):
                value = st.number_input(f'{humidity_feature[0]} (%)', 0.0, 100.0, 50.0, 0.5)
                sim_data[:, len(rain_features)] = value

            with st.expander("Imposta livelli idrometrici"):
                offset = len(rain_features) + len(humidity_feature)
                for i, feature in enumerate(hydro_features):
                    value = st.number_input(f'{feature} (m)', -1.0, 10.0, 0.0, 0.01)
                    sim_data[:, offset + i] = value

        # Bottone per eseguire la simulazione
        if st.button('Esegui simulazione', type="primary"):
            if model is None or scaler_features is None or scaler_targets is None:
                st.error("Modello o scaler non caricati correttamente. Impossibile eseguire la simulazione.")
            else:
                with st.spinner('Simulazione in corso...'):
                    # Previsione
                    predictions = predict(model, sim_data, scaler_features, scaler_targets, target_columns, device, OUTPUT_WINDOW)

                    if predictions is not None:  # Check if prediction was successful
                        # Visualizzazione dei risultati
                        st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore')

                        # Creazione dataframe risultati
                        current_time = datetime.now()
                        prediction_times = [current_time + timedelta(hours=i) for i in range(OUTPUT_WINDOW)]
                        results_df = pd.DataFrame(predictions, columns=target_columns)
                        results_df['Ora previsione'] = prediction_times
                        results_df = results_df[['Ora previsione'] + target_columns]

                        # Visualizzazione tabella risultati
                        st.dataframe(results_df.round(3))

                        # Download dei risultati
                        st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)

                        # Grafici per ogni sensore
                        st.subheader('Grafici delle previsioni')

                        # Mostra sia i dati inseriti che le previsioni
                        for i, feature in enumerate(target_columns):
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Indice per i dati degli idrometri nel sim_data
                            hydro_idx = len(rain_features) + len(humidity_feature) + i

                            # Simulazione di datestamp per i dati di input
                            input_times = [current_time - timedelta(hours=INPUT_WINDOW-j) for j in range(INPUT_WINDOW)]

                            # Dati storici (input)
                            ax.plot(input_times, sim_data[:, hydro_idx], 'b-', label='Dati inseriti')

                            # Dati previsti (output)
                            ax.plot(prediction_times, predictions[:, i], 'r-', label='Previsione')

                            # Linea verticale per separare dati storici e previsione
                            ax.axvline(x=current_time, color='black', linestyle='--')
                            ax.annotate('Ora attuale', (current_time, ax.get_ylim()[0]),
                                        xytext=(10, 10), textcoords='offset points')

                            ax.set_title(f'Idrometro: {feature}')
                            ax.set_xlabel('Data/Ora')
                            ax.set_ylabel('Livello (m)')
                            ax.legend()
                            ax.grid(True)

                            # Formattazione delle date sull'asse x
                            plt.xticks(rotation=45)
                            plt.tight_layout()

                            st.pyplot(fig)
                            sensor_name = feature.replace(' ', '_').replace('/', '_')
                            st.markdown(get_image_download_link(fig, f"sim_{sensor_name}.png", f"Scarica il grafico di {feature}"), unsafe_allow_html=True)

elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici')

    if df is None:
        st.warning("Non sono stati caricati dati storici. Carica un file CSV per procedere.")
    else:
        # Selezione del range temporale
        st.subheader('Seleziona il periodo di analisi')

        # Otteniamo il range di date disponibili
        min_date = df['Data e Ora'].min()
        max_date = df['Data e Ora'].max()

        # Conversione in formato datetime per il selezione
        min_date_input = min_date.to_pydatetime()
        max_date_input = max_date.to_pydatetime()

        # Selezione del range di date
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Data inizio', min_date_input, min_value=min_date_input, max_value=max_date_input)
        with col2:
            end_date = st.date_input('Data fine', max_date_input, min_value=min_date_input, max_value=max_date_input)

        # Filtraggio dei dati in base al range selezionato
        mask = (df['Data e Ora'].dt.date >= start_date) & (df['Data e Ora'].dt.date <= end_date)
        filtered_df = df.loc[mask]

        if len(filtered_df) == 0:
            st.warning("Nessun dato trovato nel periodo selezionato.")
        else:
            st.success(f"Trovati {len(filtered_df)} dati nel periodo selezionato.")

            # Analisi statistica
            st.subheader('Analisi statistica')

            # Selezione della feature da analizzare
            feature_to_analyze = st.selectbox(
                'Seleziona la feature da analizzare',
                feature_columns
            )

            # Statistiche descrittive
            stats = filtered_df[feature_to_analyze].describe()
            stats_df = pd.DataFrame(stats).transpose()
            st.dataframe(stats_df.round(3))

            # Visualizzazione dell'andamento temporale
            st.subheader('Andamento temporale')

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_df['Data e Ora'], filtered_df[feature_to_analyze])
            ax.set_title(f'Andamento temporale di {feature_to_analyze}')
            ax.set_xlabel('Data e Ora')
            ax.set_ylabel(feature_to_analyze)
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Analisi di correlazione
            st.subheader('Analisi di correlazione')

            # Selezione delle features per la correlazione
            corr_features = st.multiselect(
                'Seleziona le features per la correlazione',
                feature_columns,
                default=[hydro_features[0], rain_features[0], humidity_feature[0]]
            )

            if len(corr_features) > 1:
                # Calcolo della matrice di correlazione
                corr_matrix = filtered_df[corr_features].corr()

                # Visualizzazione della heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Matrice di correlazione')
                plt.tight_layout()
                st.pyplot(fig)

                # Scatterplot per coppie di features
                if len(corr_features) == 2:
                    st.subheader('Scatterplot')

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(filtered_df[corr_features[0]], filtered_df[corr_features[1]], alpha=0.5)
                    ax.set_title(f'Scatterplot: {corr_features[0]} vs {corr_features[1]}')
                    ax.set_xlabel(corr_features[0])
                    ax.set_ylabel(corr_features[1])
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Seleziona almeno due features per l'analisi di correlazione.")

            # Distribuzione dei valori
            st.subheader('Distribuzione dei valori')

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_df[feature_to_analyze], kde=True, ax=ax)
            ax.set_title(f'Distribuzione di {feature_to_analyze}')
            ax.set_xlabel(feature_to_analyze)
            ax.set_ylabel('Frequenza')
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

            # Download dei dati filtrati
            st.subheader('Download dei dati')
            st.markdown(get_table_download_link(filtered_df, "dati_filtrati.csv"), unsafe_allow_html=True)

elif page == 'Allenamento Modello':
    st.header('Allenamento Modello')

    if df is None:
        st.warning("Non sono stati caricati dati per l'addestramento. Carica un file CSV per procedere.")
    else:
        st.write(f"Dati disponibili: {len(df)} righe.")

        # Opzioni di addestramento
        st.subheader('Configurazione dell\'addestramento')

        # Selezione target (quali idrometri prevedere)
        st.write("Seleziona gli idrometri da prevedere:")
        selected_targets = []

        # Creiamo una colonna per ogni idrometro
        columns = st.columns(len(hydro_features))
        for i, feature in enumerate(hydro_features):
            with columns[i]:
                if st.checkbox(feature.split("[m]")[-1].strip(), value=True):
                    selected_targets.append(feature)

        if len(selected_targets) == 0:
            st.warning("Seleziona almeno un idrometro da prevedere.")
        else:
            # Parametri del modello e dell'addestramento
            with st.expander("Parametri avanzati", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    # Finestre temporali
                    input_window = st.number_input("Finestra di input (ore)", min_value=1, max_value=72, value=INPUT_WINDOW)
                    output_window = st.number_input("Finestra di output (ore)", min_value=1, max_value=72, value=OUTPUT_WINDOW)

                    # Percentuale di validazione
                    val_split = st.slider("% dati di validazione", 5, 30, 20)

                with col2:
                    # Parametri del modello
                    hidden_size_train = st.number_input("Dimensione hidden layer", min_value=16, max_value=512, value=128, step=16)
                    num_layers_train = st.number_input("Numero di layer", min_value=1, max_value=5, value=2)
                    dropout_train = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05, key="dropout_train_slider") # chiave univoca per slider dropout in training

                    # Parametri dell'addestramento
                    learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
                    batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)

            # Numero di epoche
            epochs = st.slider("Numero di epoche", 10, 200, 50)

            # Bottone per avviare l'addestramento
            train_button = st.button("Addestra modello", type="primary")

            if train_button:
                with st.spinner('Preparazione dei dati in corso...'):
                    # Preparazione dei dati
                    X_train, y_train, X_val, y_val, scaler_features, scaler_targets = prepare_training_data(
                        df, feature_columns, selected_targets, input_window, output_window, val_split
                    )

                    st.success(f"Dati preparati: {len(X_train)} esempi di training, {len(X_val)} esempi di validazione")

                    # Visualizzazione della forma dei dati
                    st.write(f"Forma dei dati di input: {X_train.shape}")
                    st.write(f"Forma dei dati di output: {y_train.shape}")

                # Addestramento del modello
                st.subheader("Addestramento in corso")

                # Calcolo dell'input_size basato sulle feature
                input_size = len(feature_columns)
                output_size = len(selected_targets)

                st.write(f"Dimensione input: {input_size}, Dimensione output: {output_size}")

                # Addestramento
                model, train_losses, val_losses = train_model(
                    X_train, y_train, X_val, y_val,
                    input_size, output_size, output_window,
                    hidden_size_train, num_layers_train, epochs, batch_size, learning_rate, dropout_train
                )

                # Visualizzazione dei risultati
                st.success("Addestramento completato!")

                # Grafico della loss finale
                st.subheader("Andamento della loss durante l'addestramento")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses, label='Train Loss')
                ax.plot(val_losses, label='Validation Loss')
                ax.set_title('Andamento della loss')
                ax.set_xlabel('Epoca')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Salvataggio del modello e degli scaler
                st.subheader("Download del modello e degli scaler")

                # Salvataggio del modello
                model_buffer = io.BytesIO()
                torch.save(model.state_dict(), model_buffer)
                model_buffer.seek(0)

                # Salvataggio degli scaler
                scaler_features_buffer = io.BytesIO()
                joblib.dump(scaler_features, scaler_features_buffer)
                scaler_features_buffer.seek(0)

                scaler_targets_buffer = io.BytesIO()
                joblib.dump(scaler_targets, scaler_targets_buffer)
                scaler_targets_buffer.seek(0)

                # Link per il download
                st.markdown(get_binary_file_download_link(model_buffer, "best_hydro_model.pth", "Scarica il modello addestrato"), unsafe_allow_html=True)
                st.markdown(get_binary_file_download_link(scaler_features_buffer, "scaler_features.joblib", "Scarica lo scaler features"), unsafe_allow_html=True)
                st.markdown(get_binary_file_download_link(scaler_targets_buffer, "scaler_targets.joblib", "Scarica lo scaler targets"), unsafe_allow_html=True)

                # Info sul modello
                st.subheader("Informazioni sul modello")
                model_info = {
                    "Tipo di modello": "LSTM",
                    "Input window": input_window,
                    "Output window": output_window,
                    "Hidden size": hidden_size_train,
                    "Num layers": num_layers_train,
                    "Dropout": dropout_train,
                    "Sensori previsti": ", ".join(selected_targets),
                    "Data di addestramento": datetime.now().strftime("%d/%m/%Y %H:%M")
                }

                # Visualizzazione delle info sul modello
                st.json(model_info)

                # Esempio di utilizzo del modello per previsione
                st.subheader("Test del modello")

                if st.button("Esegui test su dati di validazione"):
                    with st.spinner("Esecuzione test in corso..."):
                        # Prendiamo un esempio dal set di validazione
                        sample_idx = np.random.randint(0, len(X_val))
                        sample_input = X_val[sample_idx]
                        sample_target = y_val[sample_idx]

                        # Convertiamo in tensor
                        sample_input_tensor = torch.FloatTensor(sample_input).unsqueeze(0).to(device)

                        # Previsione
                        model.eval()
                        with torch.no_grad():
                            sample_output = model(sample_input_tensor)

                        # Conversione in numpy
                        sample_output_np = sample_output.cpu().numpy().reshape(-1, len(selected_targets))

                        # Denormalizzazione
                        sample_input_original = scaler_features.inverse_transform(sample_input.reshape(-1, input_size)).reshape(input_window, input_size)
                        sample_target_original = scaler_targets.inverse_transform(sample_target.reshape(-1, len(selected_targets))).reshape(output_window, len(selected_targets))
                        sample_output_original = scaler_targets.inverse_transform(sample_output_np)

                        # Creazione di un dataframe per visualizzazione
                        test_results = pd.DataFrame()

                        for i, target in enumerate(selected_targets):
                            target_name = target.split("[m]")[-1].strip()
                            test_results[f'Target {target_name}'] = sample_target_original[:, i]
                            test_results[f'Previsione {target_name}'] = sample_output_original[:, i]

                        # Visualizzazione dei risultati
                        st.dataframe(test_results.round(3))

                        # Grafico dei risultati
                        for i, target in enumerate(selected_targets):
                            target_name = target.split("[m]")[-1].strip()

                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(sample_target_original[:, i], 'b-', label=f'Target {target_name}')
                            ax.plot(sample_output_original[:, i], 'r--', label=f'Previsione {target_name}')
                            ax.set_title(f'Test previsione - {target_name}')
                            ax.set_xlabel('Ore future')
                            ax.set_ylabel('Livello idrometrico [m]')
                            ax.legend()
                            ax.grid(True)
                            plt.tight_layout()
                            st.pyplot(fig)

# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico © 2023')
