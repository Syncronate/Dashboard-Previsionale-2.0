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
import gspread
from google.oauth2.service_account import Credentials
import re
import json # Importa json per leggere/scrivere file di configurazione
import glob # Per cercare i file dei modelli

# --- Costanti ---
MODELS_DIR = "models" # Cartella dove risiedono i modelli pre-addestrati
# Rimuovi le costanti globali INPUT_WINDOW, OUTPUT_WINDOW

# --- Definizioni Funzioni (Dataset, LSTM) ---
# ... (TimeSeriesDataset, HydroLSTM rimangono quasi invariate) ...
# HydroLSTM init rimane uguale, riceve i parametri dinamicamente

# --- Funzioni Utilità Modificate ---

@st.cache_data # Cache basata sugli argomenti (percorsi file)
def load_model_config(config_path):
    """Carica la configurazione JSON di un modello."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Validazione base della config (aggiungere più controlli se necessario)
        required_keys = ["input_window", "output_window", "hidden_size", "num_layers", "dropout", "target_columns"]
        if not all(key in config for key in required_keys):
            st.error(f"File di configurazione '{config_path}' incompleto. Mancano chiavi.")
            return None
        return config
    except FileNotFoundError:
        st.error(f"File di configurazione '{config_path}' non trovato.")
        return None
    except json.JSONDecodeError:
        st.error(f"Errore nel parsing del file JSON '{config_path}'.")
        return None
    except Exception as e:
        st.error(f"Errore nel caricamento della configurazione '{config_path}': {e}")
        return None

# Cache per caricare il modello specifico, dipende da path e parametri letti dal JSON
@st.cache_resource(show_spinner="Caricamento modello Pytorch...")
def load_specific_model(model_path, config):
    """Carica un modello .pth dati il percorso e la sua configurazione."""
    if not config:
        st.error("Configurazione non valida per caricare il modello.")
        return None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Crea il modello CON I PARAMETRI DALLA CONFIGURAZIONE SPECIFICA
        model = HydroLSTM(
            input_size=len(config.get("feature_columns", [])), # Assumi feature_columns sia nella config o usa un default
            hidden_size=config["hidden_size"],
            output_size=len(config["target_columns"]), # Determina output_size dai target nella config
            output_window=config["output_window"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        ).to(device)

        # Caricamento pesi
        if isinstance(model_path, str): # Path file
             if not os.path.exists(model_path):
                  raise FileNotFoundError(f"File modello '{model_path}' non trovato.")
             model.load_state_dict(torch.load(model_path, map_location=device))
        elif hasattr(model_path, 'getvalue'): # File caricato
             model_path.seek(0)
             model.load_state_dict(torch.load(model_path, map_location=device))
        else:
             raise TypeError("Percorso modello non valido.")

        model.eval()
        st.success(f"Modello '{os.path.basename(model_path)}' caricato su {device}.")
        return model, device
    except FileNotFoundError as e:
         st.error(f"Errore caricamento modello: {e}")
         return None, None
    except Exception as e:
        # Più dettagli sull'errore (es. discrepanza chiavi stato)
        st.error(f"Errore durante il caricamento dei pesi del modello '{os.path.basename(model_path)}': {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Cache per caricare gli scaler specifici
@st.cache_resource(show_spinner="Caricamento scaler...")
def load_specific_scalers(scaler_features_path, scaler_targets_path):
    """Carica gli scaler .joblib dati i percorsi."""
    scaler_features = None
    scaler_targets = None
    try:
        # Gestione sia percorsi file che file caricati
        def _load_joblib(path):
             if isinstance(path, str):
                  if not os.path.exists(path):
                       raise FileNotFoundError(f"File scaler '{path}' non trovato.")
                  return joblib.load(path)
             elif hasattr(path, 'getvalue'):
                  path.seek(0)
                  return joblib.load(path)
             else:
                  raise TypeError("Percorso scaler non valido.")

        scaler_features = _load_joblib(scaler_features_path)
        scaler_targets = _load_joblib(scaler_targets_path)

        st.success(f"Scaler '{os.path.basename(scaler_features_path)}' e '{os.path.basename(scaler_targets_path)}' caricati.")
        return scaler_features, scaler_targets
    except FileNotFoundError as e:
         st.error(f"Errore caricamento scaler: {e}")
         return None, None
    except Exception as e:
        st.error(f"Errore durante il caricamento degli scaler: {e}")
        return None, None

def find_available_models(models_dir=MODELS_DIR):
    """Scansiona la cartella dei modelli e restituisce un dizionario di modelli validi."""
    available = {}
    if not os.path.isdir(models_dir):
        st.warning(f"Cartella modelli '{models_dir}' non trovata.")
        return available

    # Cerca tutti i file .pth
    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))

    for pth_path in pth_files:
        base_name = os.path.splitext(os.path.basename(pth_path))[0]
        config_path = os.path.join(models_dir, f"{base_name}.json")
        scaler_f_path = os.path.join(models_dir, f"{base_name}_features.joblib")
        scaler_t_path = os.path.join(models_dir, f"{base_name}_targets.joblib")

        # Verifica esistenza di tutti i file necessari
        if os.path.exists(config_path) and os.path.exists(scaler_f_path) and os.path.exists(scaler_t_path):
            available[base_name] = {
                "name": base_name,
                "pth_path": pth_path,
                "config_path": config_path,
                "scaler_features_path": scaler_f_path,
                "scaler_targets_path": scaler_t_path
            }
        else:
            st.warning(f"Modello '{base_name}' ignorato: file mancanti (config JSON o scaler joblib).")

    return available

# --- Funzione Predict Modificata ---
def predict(model, input_data, scaler_features, scaler_targets, config, device):
    """
    Funzione per fare previsioni con il modello addestrato.
    Utilizza la configurazione per le finestre e i target.
    """
    if model is None or scaler_features is None or scaler_targets is None or config is None:
        st.error("Modello, scaler o configurazione non disponibili. Impossibile fare previsioni.")
        return None

    # Leggi parametri dalla configurazione specifica del modello
    input_window = config["input_window"]
    output_window = config["output_window"]
    target_columns = config["target_columns"]
    feature_columns = config.get("feature_columns", []) # Leggi anche le feature columns se presenti

    # Validazione input_data
    if input_data.shape[0] != input_window:
         st.error(f"Dati input hanno {input_data.shape[0]} righe, ma modello '{config.get('name', 'N/A')}' richiede {input_window}. Impossibile prevedere.")
         return None
    # Verifica numero colonne input (se feature_columns è definito nella config)
    if feature_columns and input_data.shape[1] != len(feature_columns):
         st.error(f"Dati input hanno {input_data.shape[1]} colonne, ma modello '{config.get('name', 'N/A')}' richiede {len(feature_columns)} features. Impossibile prevedere.")
         return None
    elif not feature_columns and scaler_features.n_features_in_ != input_data.shape[1]:
          st.warning(f"feature_columns non specificate nella config, controllo vs scaler: input {input_data.shape[1]}, scaler {scaler_features.n_features_in_}")
          if scaler_features.n_features_in_ != input_data.shape[1]:
               st.error("Incompatibilità colonne input e scaler features.")
               return None


    model.eval()
    try:
        # Normalizzazione
        input_normalized = scaler_features.transform(input_data)
        # Conversione
        input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
        # Previsione
        with torch.no_grad():
            output = model(input_tensor) # output shape: (1, output_window, num_targets)
        # Numpy
        output_np = output.cpu().numpy().reshape(output_window, len(target_columns)) # Già nella forma corretta

        # Denormalizzazione
        # Verifica compatibilità scaler target
        if scaler_targets.n_features_in_ != len(target_columns):
             st.error(f"Errore dimensione: output modello ha {len(target_columns)} target, ma scaler target ne attende {scaler_targets.n_features_in_}.")
             return None

        predictions = scaler_targets.inverse_transform(output_np)
        # Reshape non necessario se il modello ritorna già (batch, seq_len, features)

        return predictions

    except Exception as e:
        st.error(f"Errore durante la fase di previsione: {e}")
        # ... (log di debug come prima) ...
        return None


# --- Funzione Plot Modificata ---
def plot_predictions(predictions, config, start_time=None):
    """ Plotta le previsioni usando output_window e target_columns dalla config."""
    if config is None or predictions is None:
        return []

    output_window = config["output_window"]
    target_columns = config["target_columns"]
    figures = []
    # ... (logica plotting con Plotly rimane uguale, ma usa output_window e target_columns da config) ...
    # Per ogni sensore idrometrico target
    for i, sensor_name in enumerate(target_columns):
        fig = go.Figure()
        # Creazione dell'asse x per le ore future
        if start_time:
            hours = [start_time + timedelta(hours=h+1) for h in range(output_window)]
            x_axis = hours
            x_title = "Data e Ora Previste"
        else:
            hours = np.arange(1, output_window + 1)
            x_axis = hours
            x_title = "Ore Future"

        fig.add_trace(go.Scatter(x=x_axis, y=predictions[:, i], mode='lines+markers', name=f'Previsione {sensor_name}'))
        fig.update_layout(
            title=f'Previsione - {sensor_name}',
            xaxis_title=x_title, yaxis_title='Livello idrometrico [m]', height=400, hovermode="x unified"
        )
        fig.update_yaxes(rangemode='tozero')
        figures.append(fig)
    return figures

# --- Import Data from Sheet (Modificata per Input Window) ---
def import_data_from_sheet(sheet_id, expected_cols, input_window, date_col_name='Data_Ora', date_format='%d/%m/%Y %H:%M'):
    """
    Importa dati da Google Sheet, pulisce e restituisce le ultime `input_window` righe valide.
    """
    # ... (logica interna di gspread, pulizia virgole, N/A, ecc. rimane invariata) ...
    # La modifica principale è usare l'argomento `input_window` invece della costante globale

    try:
        # ... (codice connessione e lettura dati) ...
        gc = gspread.authorize(credentials)
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.sheet1
        data = worksheet.get_all_values()
        # ... (controlli foglio vuoto, colonne mancanti) ...
        df_sheet = pd.DataFrame(rows, columns=headers)
        df_sheet = df_sheet[columns_to_keep] # Seleziona colonne necessarie

        # --- PULIZIA E CONVERSIONE DATI ---
        if date_col_name in df_sheet.columns:
            df_sheet[date_col_name] = pd.to_datetime(df_sheet[date_col_name], format=date_format, errors='coerce')
        else:
             st.error(f"Colonna data '{date_col_name}' non presente.")
             return None

        df_sheet = df_sheet.dropna(subset=[date_col_name])
        df_sheet = df_sheet.sort_values(by=date_col_name, ascending=True)

        numeric_cols = [col for col in df_sheet.columns if col != date_col_name]
        for col in numeric_cols:
            df_sheet[col] = df_sheet[col].replace(['N/A', '', '-', ' '], np.nan, regex=False)
            df_sheet[col] = df_sheet[col].astype(str).str.replace(',', '.', regex=False)
            df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')

        # --- FINE PULIZIA ---

        # 4. Prendi le ultime `input_window` righe valide (USA LA VARIABILE)
        df_sheet = df_sheet.tail(input_window) # <-- MODIFICA CHIAVE

        # 5. Verifica se sono rimaste abbastanza righe
        if len(df_sheet) < input_window:
             st.warning(f"Attenzione: dopo pulizia, disponibili solo {len(df_sheet)} righe valide delle {input_window} richieste.")
             if len(df_sheet) == 0:
                 st.error("Nessuna riga valida trovata dopo pulizia.")
                 return None
        elif len(df_sheet) > input_window:
             st.warning(f"Trovate più righe ({len(df_sheet)}) del previsto ({input_window}) dopo tail().")
             df_sheet = df_sheet.tail(input_window)

        st.success(f"Importate e pulite {len(df_sheet)} righe dal foglio Google.")
        return df_sheet

    # ... (gestione eccezioni invariata) ...
    except Exception as e:
        st.error(f"Errore generico importazione GSheet: {type(e).__name__} - {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# --- Funzione Allenamento Modificata (Salvataggio Config) ---
def train_model(...): # Argomenti come prima
    # ... (logica allenamento come prima) ...
    # Dopo aver trovato il modello migliore (best_model_state_dict)

    # Carica lo stato migliore nel modello finale
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        st.success(f"Caricato modello migliore...")
    else:
        st.warning("Nessun modello migliore salvato.")
        # Potresti voler ritornare None o il modello dell'ultima epoca

    return model, train_losses, val_losses # Ritorna il modello addestrato

# --- Inizializzazione Session State ---
if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = None
if 'active_config' not in st.session_state:
    st.session_state.active_config = None
if 'active_model' not in st.session_state:
    st.session_state.active_model = None
if 'active_device' not in st.session_state:
    st.session_state.active_device = None
if 'active_scaler_features' not in st.session_state:
    st.session_state.active_scaler_features = None
if 'active_scaler_targets' not in st.session_state:
    st.session_state.active_scaler_targets = None
if 'feature_columns' not in st.session_state: # Definisci un default o leggi dal primo modello
     # IMPORTANTE: Decidi come gestire le feature_columns. Sono sempre le stesse
     # o cambiano per modello? Assumiamo siano le stesse per ora.
     # Definisci qui le feature columns globali se sono fisse.
     st.session_state.feature_columns = [
         'Cumulata Sensore 1295 (Arcevia)', 'Cumulata Sensore 2637 (Bettolelle)',
         'Cumulata Sensore 2858 (Barbara)', 'Cumulata Sensore 2964 (Corinaldo)',
         'Umidita\' Sensore 3452 (Montemurello)',
         'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
         'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
         'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
         'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
         'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
     ]


# --- LAYOUT STREAMLIT ---
st.title('Dashboard Modello Predittivo Idrologico')

# --- Sidebar ---
st.sidebar.header('Impostazioni Dati e Modello')

# Caricamento Dati Storici (come prima, ma senza caricare modello/scaler qui)
uploaded_data_file = st.sidebar.file_uploader('File CSV Dati Storici (per Analisi/Training)', type=['csv'])
df = None
# ... (logica caricamento df da uploaded_data_file o path di default) ...
# Assicurati che la logica di caricamento e pulizia di df sia robusta
# e che definisca st.session_state.feature_columns se non già fatto.

st.sidebar.divider()
st.sidebar.header("Selezione Modello Predittivo")

# Trova modelli disponibili
available_models_dict = find_available_models(MODELS_DIR)
model_options = list(available_models_dict.keys())

# Opzioni Selezione Modello
MODEL_CHOICE_UPLOAD = "Usa File Caricati Manualmente"
MODEL_CHOICE_NONE = "Nessun Modello Selezionato"

selection_options = [MODEL_CHOICE_NONE] + model_options + [MODEL_CHOICE_UPLOAD]

# Gestione selezione precedente
current_selection = st.session_state.get('active_model_name', MODEL_CHOICE_NONE)
if current_selection not in selection_options:
     current_selection = MODEL_CHOICE_NONE # Default se il modello salvato non esiste più

selected_model_option = st.sidebar.selectbox(
    "Scegli il modello da usare per Previsione/Simulazione:",
    selection_options,
    index=selection_options.index(current_selection) # Imposta selezione corrente
)

# --- Caricamento Modello/Scaler in base alla selezione ---
config_to_load = None
model_to_load = None
device_to_load = None
scaler_f_to_load = None
scaler_t_to_load = None
load_error_sidebar = False

if selected_model_option == MODEL_CHOICE_NONE:
    st.sidebar.info("Nessun modello selezionato. Funzionalità di previsione/simulazione disabilitate.")
    # Resetta stato attivo
    st.session_state.active_model_name = MODEL_CHOICE_NONE
    st.session_state.active_config = None
    st.session_state.active_model = None
    st.session_state.active_device = None
    st.session_state.active_scaler_features = None
    st.session_state.active_scaler_targets = None

elif selected_model_option == MODEL_CHOICE_UPLOAD:
    st.sidebar.subheader("Carica File Modello Manualmente")
    model_file_up = st.sidebar.file_uploader('File Modello (.pth)', type=['pth'])
    scaler_features_file_up = st.sidebar.file_uploader('File Scaler Features (.joblib)', type=['joblib'])
    scaler_targets_file_up = st.sidebar.file_uploader('File Scaler Target (.joblib)', type=['joblib'])

    st.sidebar.subheader("Configurazione Modello Caricato")
    # Input MANUALE per la configurazione quando si caricano i file
    input_window_up = st.sidebar.number_input("Input Window (ore)", min_value=1, value=24, key="up_in")
    output_window_up = st.sidebar.number_input("Output Window (ore)", min_value=1, value=12, key="up_out")
    hidden_size_up = st.sidebar.number_input("Dimensione Hidden Layer", min_value=16, value=128, step=16, key="up_hidden")
    num_layers_up = st.sidebar.number_input("Numero Layer LSTM", min_value=1, value=2, key="up_layers")
    dropout_up = st.sidebar.slider("Dropout", 0.0, 0.7, 0.2, 0.05, key="up_dropout")
    # Target columns per modello caricato: difficile da sapere, potremmo chiedere all'utente
    # o assumere tutti gli idrometri. Per semplicità, assumiamo tutti gli idrometri definiti globalmente.
    # IMPORTANTE: Questa è un'assunzione forte!
    hydro_features_global = [col for col in st.session_state.feature_columns if 'Livello' in col] # Trova idrometri
    target_columns_up = st.sidebar.multiselect(
         "Seleziona Target Columns previste da questo modello",
         hydro_features_global,
         default=hydro_features_global, # Assumi tutti di default
         key="up_targets"
    )


    if model_file_up and scaler_features_file_up and scaler_targets_file_up and target_columns_up:
        st.sidebar.info("Tentativo caricamento file...")
        # Crea una config "al volo"
        temp_config = {
            "input_window": input_window_up,
            "output_window": output_window_up,
            "hidden_size": hidden_size_up,
            "num_layers": num_layers_up,
            "dropout": dropout_up,
            "target_columns": target_columns_up,
             # Assumi le feature columns globali definite in session state
            "feature_columns": st.session_state.feature_columns,
            "name": "uploaded_model"
        }
        # Carica modello e scaler dai file buffer
        model_to_load, device_to_load = load_specific_model(model_file_up, temp_config)
        scaler_f_to_load, scaler_t_to_load = load_specific_scalers(scaler_features_file_up, scaler_targets_file_up)

        if model_to_load and scaler_f_to_load and scaler_t_to_load:
             config_to_load = temp_config # Configurazione temporanea valida
             st.session_state.active_model_name = MODEL_CHOICE_UPLOAD
        else:
             load_error_sidebar = True

    elif not (model_file_up and scaler_features_file_up and scaler_targets_file_up):
         st.sidebar.warning("Carica tutti e tre i file (.pth, 2x .joblib) e definisci i parametri.")
         load_error_sidebar = True # Non un errore fatale, ma blocca l'uso

else: # Un modello pre-addestrato dalla cartella è stato selezionato
    model_info = available_models_dict[selected_model_option]
    st.sidebar.info(f"Modello selezionato: {selected_model_option}")

    # Carica configurazione
    config_to_load = load_model_config(model_info["config_path"])

    if config_to_load:
        # Aggiungi info percorso/nome alla config caricata per riferimento
        config_to_load["pth_path"] = model_info["pth_path"]
        config_to_load["scaler_features_path"] = model_info["scaler_features_path"]
        config_to_load["scaler_targets_path"] = model_info["scaler_targets_path"]
        config_to_load["name"] = selected_model_option
         # Assicura che le feature columns siano nella config, altrimenti usa quelle globali
        if "feature_columns" not in config_to_load:
             config_to_load["feature_columns"] = st.session_state.feature_columns

        # Carica modello e scaler specifici
        model_to_load, device_to_load = load_specific_model(model_info["pth_path"], config_to_load)
        scaler_f_to_load, scaler_t_to_load = load_specific_scalers(model_info["scaler_features_path"], model_info["scaler_targets_path"])

        if not (model_to_load and scaler_f_to_load and scaler_t_to_load):
            load_error_sidebar = True # Errore durante caricamento modello/scaler
            config_to_load = None # Invalida config se caricamento fallisce
        else:
             st.session_state.active_model_name = selected_model_option


    else:
        load_error_sidebar = True # Errore caricamento config JSON

# Aggiorna session state DOPO tutta la logica di caricamento
st.session_state.active_config = config_to_load
st.session_state.active_model = model_to_load
st.session_state.active_device = device_to_load
st.session_state.active_scaler_features = scaler_f_to_load
st.session_state.active_scaler_targets = scaler_t_to_load


# Messaggio di stato caricamento
if st.session_state.active_model and st.session_state.active_config:
    st.sidebar.success(f"Modello '{st.session_state.active_model_name}' attivo (In: {st.session_state.active_config['input_window']}h, Out: {st.session_state.active_config['output_window']}h)")
elif load_error_sidebar and selected_model_option != MODEL_CHOICE_NONE:
     st.sidebar.error("Caricamento modello/config/scaler fallito.")


# --- Menu Principale (come prima) ---
st.sidebar.divider()
st.sidebar.header('Menu Navigazione')
page = st.sidebar.radio('Scegli una funzionalità',
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello'])


# --- Logica Pagine Modificata ---

# Leggi lo stato attivo all'inizio di ogni sezione che lo usa
active_config = st.session_state.active_config
active_model = st.session_state.active_model
active_device = st.session_state.active_device
active_scaler_features = st.session_state.active_scaler_features
active_scaler_targets = st.session_state.active_scaler_targets
feature_columns = st.session_state.feature_columns # Feature columns globali/lette dalla config

if page == 'Dashboard':
    st.header('Dashboard Idrologica')
    if not active_model or not active_config:
        st.warning("Nessun modello attivo selezionato o caricato correttamente. Selezionane uno dalla sidebar.")
    elif df is None:
         st.warning("Dati storici non caricati. Carica un file CSV.")
    else:
        # Usa active_config per i parametri
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns = active_config["target_columns"]
        # Assicurati che le feature columns usate qui siano corrette
        # Potrebbero essere in active_config["feature_columns"] o globali
        current_feature_columns = active_config.get("feature_columns", feature_columns)


        st.info(f"Utilizzo del modello: **{st.session_state.active_model_name}** (Input: {input_window}h, Output: {output_window}h)")
        st.write(f"Target previsti: {', '.join(target_columns)}")

        # Mostra ultimi dati (assicurati che df e colonne esistano)
        st.subheader('Ultimi dati disponibili')
        last_data = df.iloc[-1]
        last_date = last_data[date_col_name_csv] # Usa nome colonna data corretto

        col1, col2, col3 = st.columns(3)
        # ... (visualizzazione dati come prima, ma usa target_columns da active_config) ...
        with col2:
             st.subheader('Livelli idrometrici [m]')
             # Mostra solo i target columns previsti dal modello attivo
             hydro_data_display = last_data[target_columns].round(3).to_frame(name="Valore")
             st.dataframe(hydro_data_display)

        # Previsione basata sugli ultimi dati
        st.header('Previsione basata sugli ultimi dati')
        if st.button('Genera previsione dagli ultimi dati', type="primary"):
            with st.spinner('Generazione previsione...'):
                # Prepara dati input usando input_window attivo
                latest_data = df.iloc[-input_window:][current_feature_columns].values
                if latest_data.shape[0] < input_window:
                    st.error(f"Dati insufficienti ({latest_data.shape[0]}) per finestra input ({input_window}).")
                else:
                    # Usa la funzione predict aggiornata
                    predictions = predict(active_model, latest_data, active_scaler_features, active_scaler_targets, active_config, active_device)

                    if predictions is not None:
                        # Visualizza risultati (usa plot_predictions aggiornata)
                        st.subheader(f'Previsione per le prossime {output_window} ore')
                        # ... (crea dataframe risultati) ...
                        figs = plot_predictions(predictions, active_config, last_date)
                        # ... (visualizza tabella e grafici) ...

elif page == 'Simulazione':
    st.header('Simulazione Idrologica')
    if not active_model or not active_config:
        st.warning("Nessun modello attivo selezionato o caricato. Selezionane uno dalla sidebar.")
    else:
        # Usa parametri da active_config
        input_window = active_config["input_window"]
        output_window = active_config["output_window"]
        target_columns = active_config["target_columns"]
        current_feature_columns = active_config.get("feature_columns", feature_columns)

        st.info(f"Simulazione con modello: **{st.session_state.active_model_name}** (Input Richiesto: {input_window}h, Previsione: {output_window}h)")
        st.write(f"Target previsti: {', '.join(target_columns)}")

        sim_data_input = None
        sim_method = st.radio(...) # ['Manuale Costante', 'Google Sheet', 'Orario Dettagliato']

        if sim_method == 'Inserisci manualmente valori costanti':
            st.subheader(f'Inserisci valori costanti per le {input_window} ore precedenti') # Usa input_window attivo
            # ... (logica input manuale come prima) ...
            # Crea array numpy replicando per input_window attivo
            sim_data_list = []
            for feature in current_feature_columns:
                 sim_data_list.append(np.repeat(temp_sim_values[feature], input_window)) # Usa input_window
            sim_data_input = np.column_stack(sim_data_list)
            # ... (anteprima) ...

        elif sim_method == 'Importa da Google Sheet':
            st.subheader(f'Importa le ultime {input_window} ore dal foglio Google') # Usa input_window attivo
            # ... (Input URL, mapping colonne come prima) ...
            # Modifica la chiamata a import_data_from_sheet per passare input_window
            if st.button("Importa e Prepara dati dal foglio Google"):
                # ... (logica estrazione sheet_id) ...
                with st.spinner("Importazione e pulizia dati..."):
                    sheet_data_cleaned = import_data_from_sheet(
                        sheet_id,
                        expected_google_sheet_cols,
                        input_window, # Passa input_window attivo
                        date_col_name=date_col_name_gsheet,
                        date_format='%d/%m/%Y %H:%M'
                    )
                    # ... (logica mappatura e gestione umidità come prima) ...
                    if successful_mapping:
                         # Verifica colonne e crea sim_data_input
                         missing_model_features = [col for col in current_feature_columns if col not in mapped_data.columns]
                         if missing_model_features:
                              st.error(f"Errore mappatura GSheet: mancano colonne: {missing_model_features}")
                         else:
                              sim_data_input = mapped_data[current_feature_columns].values
                              # ... (salva in session state, mostra anteprima) ...

        elif sim_method == 'Inserisci dati orari (Avanzato)':
            st.subheader(f'Inserisci dati specifici per ogni ora ({input_window} ore precedenti)') # Usa input_window attivo
            # ... (logica data_editor come prima, ma assicurati che gestisca input_window dinamicamente) ...
            # Potrebbe essere necessario resettare st.session_state.sim_hourly_data se input_window cambia
            # Verifica numero righe contro input_window attivo
            if len(edited_df) != input_window:
                 st.warning(f"Attenzione: la tabella contiene {len(edited_df)} righe, ma ne sono richieste {input_window}.")
                 # ... (gestione righe in eccesso/difetto) ...

            if edited_df is not None:
                 try:
                    sim_data_input = edited_df[current_feature_columns].values # Assicurati ordine colonne
                    st.session_state.sim_hourly_data = edited_df # Salva modifiche
                    # ...
                 except KeyError as ke:
                    st.error(f"Errore colonne data editor: {ke}")
                    sim_data_input = None

        # --- ESECUZIONE SIMULAZIONE ---
        st.divider()
        run_simulation = st.button('Esegui simulazione', type="primary", disabled=(sim_data_input is None))
        if run_simulation and sim_data_input is not None:
             # Verifica dimensioni input
             if sim_data_input.shape[0] != input_window or sim_data_input.shape[1] != len(current_feature_columns):
                  st.error(f"Errore dimensioni input simulazione. Atteso: ({input_window}, {len(current_feature_columns)}), Ottenuto: {sim_data_input.shape}")
             else:
                  with st.spinner('Simulazione...'):
                       # Usa predict aggiornata
                       predictions_sim = predict(active_model, sim_data_input, active_scaler_features, active_scaler_targets, active_config, active_device)
                       if predictions_sim is not None:
                           # Visualizza risultati (usa plot_predictions aggiornata)
                           st.subheader(f'Risultato Simulazione: Previsione per {output_window} ore')
                           # ... (crea dataframe risultati) ...
                           figs_sim = plot_predictions(predictions_sim, active_config, datetime.now())
                           # ... (visualizza tabella e grafici) ...


elif page == 'Analisi Dati Storici':
     # Questa sezione dipende principalmente da df, non dal modello attivo
     # Rimane sostanzialmente invariata, ma assicurati usi feature_columns corrette
     st.header('Analisi Dati Storici')
     if df is None:
          st.warning("Dati storici non caricati.")
     else:
          # Usa feature_columns globali o quelle dell'ultimo config caricato se vuoi
          cols_to_analyze = feature_columns
          # ... (resto della logica di analisi come prima, usando cols_to_analyze) ...

elif page == 'Allenamento Modello':
    st.header('Allenamento Nuovo Modello LSTM')
    if df is None:
        st.warning("Dati storici non caricati.")
    else:
        st.info(f"Dati disponibili: {len(df)} righe.")
        st.subheader('Configurazione Addestramento')

        # Input per nome modello da salvare
        save_model_name = st.text_input("Nome base per salvare il modello (es. 'modello_prova_36_18')", value=f"modello_{datetime.now().strftime('%Y%m%d_%H%M')}")

        # Selezione target (come prima)
        selected_targets_train = []
        # ... (checkbox per selezionare target) ...

        # Parametri (come prima)
        input_window_train = st.number_input("Input Window (ore)", ..., value=24)
        output_window_train = st.number_input("Output Window (ore)", ..., value=12)
        # ... (altri parametri: hidden, layers, dropout, lr, batch, epochs) ...

        train_button = st.button("Addestra Nuovo Modello", type="primary")

        if train_button and selected_targets_train and save_model_name:
             # Verifica nome valido
             if not re.match(r'^[a-zA-Z0-9_-]+$', save_model_name):
                  st.error("Nome modello non valido. Usa solo lettere, numeri, trattini, underscore.")
             else:
                  # --- Preparazione Dati ---
                  with st.spinner('Preparazione dati...'):
                       # Assumi feature_columns globali per l'input
                       X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train = prepare_training_data(
                           df.copy(), feature_columns, selected_targets_train,
                           input_window_train, output_window_train, val_split_train
                       )
                       # ... (controllo errore preparazione) ...

                  # --- Addestramento ---
                  st.subheader("Addestramento in corso...")
                  input_size_train = len(feature_columns)
                  output_size_train = len(selected_targets_train)
                  # ... (chiama train_model) ...
                  trained_model, train_losses, val_losses = train_model(...)

                  # --- Salvataggio Risultati ---
                  if trained_model:
                      st.success("Addestramento completato!")
                      # ... (mostra grafico loss) ...

                      st.subheader("Salvataggio Modello, Configurazione e Scaler")
                      # Crea cartella se non esiste
                      os.makedirs(MODELS_DIR, exist_ok=True)

                      # Definisci percorsi salvataggio
                      base_path = os.path.join(MODELS_DIR, save_model_name)
                      model_save_path = f"{base_path}.pth"
                      config_save_path = f"{base_path}.json"
                      scaler_f_save_path = f"{base_path}_features.joblib"
                      scaler_t_save_path = f"{base_path}_targets.joblib"

                      # Crea dizionario configurazione
                      config_to_save = {
                          "input_window": input_window_train,
                          "output_window": output_window_train,
                          "hidden_size": hidden_size_train_cfg, # Usa i valori reali usati
                          "num_layers": num_layers_train_cfg,
                          "dropout": dropout_train_cfg,
                          "target_columns": selected_targets_train,
                          "feature_columns": feature_columns, # Salva anche le feature usate
                          "training_date": datetime.now().isoformat()
                          # Aggiungi altre info utili se vuoi (es. loss finale)
                      }

                      try:
                          # Salva modello .pth
                          torch.save(trained_model.state_dict(), model_save_path)
                          # Salva config .json
                          with open(config_save_path, 'w') as f:
                              json.dump(config_to_save, f, indent=4)
                          # Salva scaler .joblib
                          joblib.dump(scaler_features_train, scaler_f_save_path)
                          joblib.dump(scaler_targets_train, scaler_t_save_path)

                          st.success(f"Modello '{save_model_name}' salvato in '{MODELS_DIR}/'")
                          # Fornisci link download (opzionale se già salvato in repo)
                          # Potresti creare un zip dei 4 file per comodità

                      except Exception as e_save:
                           st.error(f"Errore durante il salvataggio dei file: {e_save}")

                      # ... (eventuale test rapido come prima) ...


# --- Footer ---
# ... (come prima) ...
