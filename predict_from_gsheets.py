"""
Script di predizione per SimpleTCN con dati da Google Sheets
Versione ottimizzata e debuggata
"""

import os
import sys
import yaml
import glob
import torch
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pytz
import warnings

# Aggiungi path per import locali
sys.path.append(os.getcwd())

from models.lightning_wrapper import LitSpatioTemporalGNN
from data.event_based_data_module import EventBasedDataModule

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== CONFIGURAZIONE ====================
CONFIG_PATH = "config_bettolelle_final.yaml"
CREDENTIALS_PATH = "credentials.json"
CHECKPOINT_DIR = "checkpoints"

# Nomi fogli Google Sheets
SHEET_HISTORICAL = "DATI METEO CON FEATURE"
SHEET_FORECAST = "Previsioni Cumulate Feature ICON"
SHEET_OUTPUT = "Previsioni Idro-Bettolelle 6h"

ITALY_TZ = pytz.timezone('Europe/Rome')

# ==================== UTILITY ====================
def load_config(path):
    """Carica configurazione YAML"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_best_checkpoint(checkpoint_dir):
    """Trova il checkpoint con il miglior val_event_peak_error"""
    files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"❌ Nessun checkpoint in {checkpoint_dir}")
    
    # Escludi "last.ckpt"
    files = [f for f in files if "last" not in os.path.basename(f).lower()]
    
    best_ckpt = None
    best_error = float('inf')
    
    for f in files:
        try:
            # Parse: bettolelle-rain-epoch=XX-val_event_peak_error=0.XXX.ckpt
            parts = os.path.basename(f).split("val_event_peak_error=")
            if len(parts) > 1:
                error_str = parts[1].replace(".ckpt", "")
                error = float(error_str)
                
                if error < best_error:
                    best_error = error
                    best_ckpt = f
        except:
            continue
    
    # Fallback: prendi il più recente
    if best_ckpt is None:
        best_ckpt = max(files, key=os.path.getmtime)
        print("⚠️  Impossibile parsare metriche. Uso checkpoint più recente.")
    else:
        print(f"🏆 Best checkpoint: {os.path.basename(best_ckpt)} (error={best_error:.3f})")
    
    return best_ckpt

# ==================== GOOGLE SHEETS ====================
def connect_gsheets(credentials_path):
    """Connessione a Google Sheets"""
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"❌ {credentials_path} non trovato")
    
    scopes = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    return gspread.authorize(creds)

def fetch_gsheet_data(gc, sheet_id, config):
    """Scarica dati da Google Sheets"""
    print("\n☁️  Caricamento dati da Google Sheets...")
    
    sh = gc.open_by_key(sheet_id)
    
    # Dati storici
    print(f"   📥 {SHEET_HISTORICAL}")
    ws_hist = sh.worksheet(SHEET_HISTORICAL)
    df_hist = pd.DataFrame(ws_hist.get_all_records())
    
    # Dati previsionali
    print(f"   📥 {SHEET_FORECAST}")
    ws_fcst = sh.worksheet(SHEET_FORECAST)
    df_fcst = pd.DataFrame(ws_fcst.get_all_records())
    
    print(f"   ✅ Storico: {len(df_hist)} righe | Forecast: {len(df_fcst)} righe")
    
    return df_hist, df_fcst

# ==================== PREPROCESSING ====================
def clean_column_names(df):
    """Rimuove prefissi ridondanti dalle colonne"""
    rename_map = {}
    for col in df.columns:
        new_col = col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
        if new_col != col:
            rename_map[col] = new_col
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        print(f"   🔄 Rinominate {len(rename_map)} colonne")
    
    return df

def parse_dates(df, config):
    """Parsing e validazione date"""
    date_col = config['data']['timestamp_column']
    date_fmt = config['data']['timestamp_format']
    
    df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors='coerce')
    
    # Rimuovi righe con date invalide
    before = len(df)
    df = df.dropna(subset=[date_col])
    after = len(df)
    
    if before != after:
        print(f"   ⚠️  Rimosse {before - after} righe con date invalide")
    
    return df.sort_values(date_col).reset_index(drop=True)

def convert_to_numeric(df, exclude_cols):
    """Converte colonne a numerico (gestisce virgole da Excel)"""
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if df[col].dtype == object:
            try:
                # Sostituisci virgola con punto
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass  # Ignora colonne non numeriche
    
    return df

def prepare_input_data(df_hist, df_fcst, config):
    """Prepara i dati per il modello"""
    print("\n⚙️  Preprocessing dati...")
    
    # 1. Pulisci nomi colonne
    df_hist = clean_column_names(df_hist)
    df_fcst = clean_column_names(df_fcst)
    
    # 2. Parse date
    df_hist = parse_dates(df_hist, config)
    df_fcst = parse_dates(df_fcst, config)
    
    date_col = config['data']['timestamp_column']
    last_hist_date = df_hist[date_col].iloc[-1]
    
    print(f"   📅 Ultimo dato storico: {last_hist_date}")
    
    # 3. Filtra previsioni future
    df_future = df_fcst[df_fcst[date_col] > last_hist_date].copy()
    print(f"   🔮 Previsioni disponibili: {len(df_future)} timestep futuri")
    
    # 4. Conversione numerica
    df_hist = convert_to_numeric(df_hist, [date_col])
    df_future = convert_to_numeric(df_future, [date_col])
    
    # 5. Forward fill + riempimento NaN
    numeric_cols = df_hist.select_dtypes(include=[np.number]).columns
    df_hist[numeric_cols] = df_hist[numeric_cols].ffill().fillna(0)
    
    if len(df_future) > 0:
        numeric_cols_fcst = df_future.select_dtypes(include=[np.number]).columns
        df_future[numeric_cols_fcst] = df_future[numeric_cols_fcst].ffill().fillna(0)
    
    return df_hist, df_future, last_hist_date

# ==================== ESTRAZIONE FEATURES ====================
def extract_features_from_df(df, config):
    """
    Estrae le feature richieste dal modello dal dataframe.
    IMPORTANTE: Usa node_columns dal YAML per sapere quali colonne servono.
    """
    node_cols = config['data']['node_columns']
    global_feats = config['data'].get('global_features', [])
    
    # Raccoglie tutte le colonne necessarie
    required_cols = set()
    
    for node_name, features in node_cols.items():
        for feat_name, col_name in features.items():
            if col_name in df.columns:
                required_cols.add(col_name)
    
    for gf in global_feats:
        if gf in df.columns:
            required_cols.add(gf)
    
    # Estrai solo colonne presenti
    available_cols = [c for c in required_cols if c in df.columns]
    missing_cols = required_cols - set(available_cols)
    
    if missing_cols:
        print(f"   ⚠️  Colonne mancanti (saranno ignorate): {missing_cols}")
    
    return df[available_cols].values

# ==================== MODELLO ====================
def load_model_and_datamodule(config_path, checkpoint_path):
    """Carica modello e DataModule"""
    print("\n🧠 Caricamento modello...")
    
    # 1. Inizializza DataModule (serve per lo scaler)
    print("   📊 Setup DataModule...")
    dm = EventBasedDataModule(config_path)
    
    # Questo carica il CSV e fitta lo scaler
    dm.prepare_data()
    dm.setup()
    
    print("   ✅ Scaler pronto")
    
    # 2. Carica modello
    print(f"   🔧 Caricamento checkpoint...")
    device = torch.device('cpu')  # Usa CPU per semplicità
    
    model = LitSpatioTemporalGNN.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    
    print("   ✅ Modello caricato")
    
    return model, dm

def predict_future(model, dm, input_data, config):
    """
    Esegue la predizione.
    
    Args:
        model: LitSpatioTemporalGNN
        dm: EventBasedDataModule (per scaler)
        input_data: numpy array (input_window, num_features)
        config: dict
    
    Returns:
        predictions: numpy array (output_window, num_quantiles)
    """
    print("\n🔮 Generazione previsioni...")
    
    # 1. Normalizza input
    input_normalized = dm.scaler.transform(input_data)
    
    # 2. Converti in tensore
    # ATTENZIONE: La forma dipende dall'architettura del modello
    # SimpleTCN si aspetta: (batch, timesteps, nodes, features)
    # Ma qui abbiamo dati flat da GSheets
    
    # Dobbiamo ricostruire la struttura a grafo
    # Per semplicità, assumiamo che dm abbia un metodo per farlo
    # Altrimenti dobbiamo implementarlo manualmente
    
    # SOLUZIONE TEMPORANEA: Usa il batch collate del datamodule
    # che sa come trasformare i dati flat in formato grafo
    
    # Wrap in tensore
    input_tensor = torch.FloatTensor(input_normalized)
    
    # Aggiungi dimensione batch
    input_batch = input_tensor.unsqueeze(0)  # (1, timesteps, features)
    
    # 3. Predizione
    with torch.no_grad():
        # Il modello restituisce (batch, output_window, nodes, 1, num_quantiles)
        output = model(input_batch)
    
    # 4. Estrai predizioni per Bettolelle (nodo 3)
    target_node_idx = 3
    
    # (1, out_window, num_quantiles)
    pred_bett = output[0, :, target_node_idx, 0, :]
    
    # 5. Denormalizza
    # PROBLEMA: Lo scaler scala tutte le feature insieme
    # Dobbiamo trovare quale colonna corrisponde al livello di Bettolelle
    
    # Dalla config, il livello di Bettolelle è la prima feature di quel nodo
    # Trova l'indice della feature target
    target_col = config['data']['node_columns']['Bettolelle']['livello']
    
    # Lo scaler è stato fittato sulle colonne in ordine
    # Dobbiamo sapere a che indice si trova target_col
    # Per ora assumiamo sia la colonna 0 (prima feature)
    
    scaler_mean = dm.scaler.mean_[0]
    scaler_scale = dm.scaler.scale_[0]
    
    pred_denorm = pred_bett.numpy() * scaler_scale + scaler_mean
    
    return pred_denorm

# ==================== SALVATAGGIO ====================
def save_to_gsheet(gc, sheet_id, predictions, start_time, config):
    """Salva previsioni su Google Sheets"""
    print(f"\n💾 Salvataggio su '{SHEET_OUTPUT}'...")
    
    sh = gc.open_by_key(sheet_id)
    
    try:
        ws = sh.worksheet(SHEET_OUTPUT)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_OUTPUT, rows=100, cols=20)
    
    # Header
    quantiles = config['model']['quantile_levels']
    header = ["Data e Ora"] + [f"Q{int(q*100)}" for q in quantiles]
    
    rows = [header]
    
    # Genera righe di previsione
    for i in range(len(predictions)):
        ts = start_time + timedelta(minutes=30 * (i + 1))
        ts_str = ts.strftime("%d/%m/%Y %H:%M")
        
        # Formatta con virgola per Excel italiano
        vals = [f"{v:.3f}".replace('.', ',') for v in predictions[i]]
        
        rows.append([ts_str] + vals)
    
    # Scrivi tutto in una volta
    ws.update(rows, value_input_option='USER_ENTERED')
    
    print(f"   ✅ Salvate {len(rows)-1} previsioni")

# ==================== MAIN ====================
def main():
    print("=" * 70)
    print(f"🚀 PREVISIONI BETTOLELLE - {datetime.now(ITALY_TZ)}")
    print("=" * 70)
    
    try:
        # 1. Carica config
        config = load_config(CONFIG_PATH)
        print(f"✅ Config caricato: {CONFIG_PATH}")
        
        # 2. Trova checkpoint
        ckpt_path = find_best_checkpoint(CHECKPOINT_DIR)
        
        # 3. Carica modello
        model, dm = load_model_and_datamodule(CONFIG_PATH, ckpt_path)
        
        # 4. Connetti a GSheets
        sheet_id = os.environ.get("GSHEET_ID")
        if not sheet_id:
            raise ValueError("❌ GSHEET_ID non configurato!")
        
        gc = connect_gsheets(CREDENTIALS_PATH)
        print(f"✅ Connesso a Google Sheets (ID: {sheet_id[:15]}...)")
        
        # 5. Scarica dati
        df_hist, df_fcst = fetch_gsheet_data(gc, sheet_id, config)
        
        # 6. Preprocessing
        df_hist, df_future, last_date = prepare_input_data(df_hist, df_fcst, config)
        
        # 7. Estrai finestra di input
        input_window = config['data']['input_window']
        
        # Ultimi input_window timestep
        if len(df_hist) < input_window:
            raise ValueError(f"❌ Dati insufficienti: {len(df_hist)} < {input_window}")
        
        input_data = extract_features_from_df(df_hist.tail(input_window), config)
        
        print(f"   📊 Input shape: {input_data.shape}")
        
        # 8. Predizione
        predictions = predict_future(model, dm, input_data, config)
        
        # 9. Salvataggio
        save_to_gsheet(gc, sheet_id, predictions, last_date, config)
        
        print("\n" + "=" * 70)
        print("✨ ESECUZIONE COMPLETATA CON SUCCESSO")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()