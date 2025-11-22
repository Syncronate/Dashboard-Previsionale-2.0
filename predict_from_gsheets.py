"""
Script di predizione SimpleTCN - Solo da Google Sheets
Fix: Lista HARDCODED delle 20 features per garantire compatibilità col checkpoint
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
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())

from models.lightning_wrapper import LitSpatioTemporalGNN

warnings.filterwarnings("ignore")

# ==================== CONFIGURAZIONE ====================
CONFIG_PATH = "config_bettolelle_final.yaml"
CREDENTIALS_PATH = "credentials.json"
CHECKPOINT_DIR = "checkpoints"

SHEET_HISTORICAL = "DATI METEO CON FEATURE"
SHEET_FORECAST = "Previsioni Cumulate Feature ICON"
SHEET_OUTPUT = "Previsioni Idro-Bettolelle 6h"

ITALY_TZ = pytz.timezone('Europe/Rome')

# ==================== UTILITY ====================
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_best_checkpoint(checkpoint_dir):
    files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"❌ Nessun checkpoint in {checkpoint_dir}")
    
    files = [f for f in files if "last" not in os.path.basename(f).lower()]
    
    best_ckpt = None
    best_error = float('inf')
    
    for f in files:
        try:
            parts = os.path.basename(f).split("val_event_peak_error=")
            if len(parts) > 1:
                error_str = parts[1].replace(".ckpt", "")
                error = float(error_str)
                
                if error < best_error:
                    best_error = error
                    best_ckpt = f
        except:
            continue
    
    if best_ckpt is None:
        best_ckpt = max(files, key=os.path.getmtime)
        print("⚠️  Uso checkpoint più recente")
    else:
        print(f"🏆 Best checkpoint: {os.path.basename(best_ckpt)} (error={best_error:.3f})")
    
    return best_ckpt

# ==================== GOOGLE SHEETS ====================
def connect_gsheets(credentials_path):
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"❌ {credentials_path} non trovato")
    
    scopes = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    return gspread.authorize(creds)

def fetch_gsheet_data(gc, sheet_id):
    print("\n☁️  Caricamento dati da Google Sheets...")
    
    sh = gc.open_by_key(sheet_id)
    
    print(f"   📥 {SHEET_HISTORICAL}")
    ws_hist = sh.worksheet(SHEET_HISTORICAL)
    df_hist = pd.DataFrame(ws_hist.get_all_records())
    
    print(f"   📥 {SHEET_FORECAST}")
    ws_fcst = sh.worksheet(SHEET_FORECAST)
    df_fcst = pd.DataFrame(ws_fcst.get_all_records())
    
    return df_hist, df_fcst

# ==================== PREPROCESSING ====================
def clean_column_names(df):
    rename_map = {}
    for col in df.columns:
        new_col = col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
        if new_col != col:
            rename_map[col] = new_col
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    return df

def parse_dates(df, date_col, date_fmt):
    df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors='coerce')
    df = df.dropna(subset=[date_col])
    return df.sort_values(date_col).reset_index(drop=True)

def convert_to_numeric(df, exclude_cols):
    for col in df.columns:
        if col in exclude_cols: continue
        try:
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame): continue
            if pd.api.types.is_numeric_dtype(col_data): continue
            if col_data.dtype == object:
                df[col] = col_data.astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except: pass
    return df

def prepare_data(df_hist, df_fcst, config):
    print("\n⚙️  Preprocessing dati...")
    df_hist = clean_column_names(df_hist)
    df_fcst = clean_column_names(df_fcst)
    date_col = config['data']['timestamp_column']
    date_fmt = config['data']['timestamp_format']
    df_hist = parse_dates(df_hist, date_col, date_fmt)
    df_fcst = parse_dates(df_fcst, date_col, date_fmt)
    last_hist_date = df_hist[date_col].iloc[-1]
    df_hist = convert_to_numeric(df_hist, [date_col])
    df_fcst = convert_to_numeric(df_fcst, [date_col])
    numeric_cols = df_hist.select_dtypes(include=[np.number]).columns
    df_hist[numeric_cols] = df_hist[numeric_cols].ffill().fillna(0)
    return df_hist, df_fcst, last_hist_date

# ==================== ESTRAZIONE FEATURES ====================
def extract_features(df, config):
    # --- LISTA ESPLICITA DELLE 20 FEATURES ---
    # Questa lista deve corrispondere esattamente a quella usata nel training del checkpoint
    target_features = [
        # 1. LIVELLI (5)
        "Livello Idrometrico Sensore 1112 [m] (Bettolelle)",
        "Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)",
        "Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)",
        "Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)",
        "Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)",
        
        # 2. PIOGGE 1h (4)
        "Cumulata Sensore 2637 (Bettolelle)_cumulata_1h",
        "Cumulata Sensore 1295 (Arcevia)_cumulata_1h",
        "Cumulata Sensore 2858 (Barbara)_cumulata_1h",
        "Cumulata Sensore 2964 (Corinaldo)_cumulata_1h",
        
        # 3. PIOGGE 3h (4)
        "Cumulata Sensore 2637 (Bettolelle)_cumulata_3h",
        "Cumulata Sensore 1295 (Arcevia)_cumulata_3h",
        "Cumulata Sensore 2858 (Barbara)_cumulata_3h",
        "Cumulata Sensore 2964 (Corinaldo)_cumulata_3h",
        
        # 4. VELOCITÀ (5)
        "Livello Idrometrico Sensore 1112 [m] (Bettolelle)_velocity",
        "Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)_velocity",
        "Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)_velocity",
        "Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)_velocity",
        "Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)_velocity",
        
        # 5. STAGIONALITÀ (2)
        "Seasonality_Sin",
        "Seasonality_Cos"
    ]
    
    # Ordiniamo per coerenza
    target_features.sort()
    
    print(f"   🎯 Target features: {len(target_features)}")
    
    # Verifica presenza
    missing = [c for c in target_features if c not in df.columns]
    if missing:
        print(f"   ⚠️  MANCANO {len(missing)} COLONNE NEL DATAFRAME:")
        for m in missing: print(f"      - {m}")
        raise ValueError("Colonne mancanti nel DataFrame. Impossibile procedere.")
        
    return df[target_features].values, target_features

# ==================== MODELLO ====================
def load_model_and_fit_scaler(config_path, checkpoint_path, historical_data, feature_names):
    print("\n🧠 Caricamento modello...")
    config = load_config(config_path)
    
    scaler = StandardScaler()
    scaler.fit(historical_data)
    
    print(f"   🔧 Checkpoint: {checkpoint_path}")
    
    model = LitSpatioTemporalGNN(
        num_nodes=5,  
        num_features=len(feature_names), 
        hidden_dim=config['model']['hidden_dim'],
        rnn_layers=config['model'].get('rnn_layers', 3),
        output_window=config['data']['output_window'],
        output_dim=1,
        num_quantiles=config['model']['num_quantiles'],
        config=config
    )
    
    device = torch.device('cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if "edge_index" in k or "edge_weight" in k: continue
        if k.startswith("model."): new_key = k.replace("model.", "gnn_model.", 1)
        else: new_key = k
        new_state_dict[new_key] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("   ✅ Pesi caricati con successo!")
    except Exception as e:
        print(f"   ❌ Errore caricamento pesi: {e}")
        sys.exit(1)

    model.eval()
    return model, scaler, config

def predict(model, scaler, input_data, config):
    print("\n🔮 Generazione previsioni...")
    input_normalized = scaler.transform(input_data)
    num_nodes = 5
    input_reshaped = np.stack([input_normalized] * num_nodes, axis=1)
    input_reshaped = input_reshaped[np.newaxis, :, :, :]
    input_tensor = torch.FloatTensor(input_reshaped)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    target_node_idx = 3
    pred_bett = output[0, :, target_node_idx, 0, :]
    scaler_mean = scaler.mean_[0]
    scaler_scale = scaler.scale_[0]
    pred_denorm = pred_bett.numpy() * scaler_scale + scaler_mean
    return pred_denorm

def save_to_gsheet(gc, sheet_id, predictions, start_time, config):
    print(f"\n💾 Salvataggio su '{SHEET_OUTPUT}'...")
    sh = gc.open_by_key(sheet_id)
    try: ws = sh.worksheet(SHEET_OUTPUT)
    except: ws = sh.add_worksheet(title=SHEET_OUTPUT, rows=100, cols=20)
    ws.clear()
    quantiles = config['model']['quantile_levels']
    header = ["Data e Ora"] + [f"Q{int(q*100)}" for q in quantiles]
    rows = [header]
    for i in range(len(predictions)):
        ts = start_time + timedelta(minutes=30 * (i + 1))
        ts_str = ts.strftime("%d/%m/%Y %H:%M")
        vals = [f"{v:.3f}".replace('.', ',') for v in predictions[i]]
        rows.append([ts_str] + vals)
    ws.update(rows, value_input_option='USER_ENTERED')
    print(f"   ✅ Salvate {len(rows)-1} previsioni")

def main():
    print("=" * 70)
    print(f"🚀 PREVISIONI BETTOLELLE - {datetime.now(ITALY_TZ)}")
    print("=" * 70)
    
    try:
        config = load_config(CONFIG_PATH)
        ckpt_path = find_best_checkpoint(CHECKPOINT_DIR)
        sheet_id = os.environ.get("GSHEET_ID")
        gc = connect_gsheets(CREDENTIALS_PATH)
        df_hist, df_fcst = fetch_gsheet_data(gc, sheet_id)
        df_hist, df_fcst, last_date = prepare_data(df_hist, df_fcst, config)
        
        historical_features, feature_names = extract_features(df_hist, config)
        
        model, scaler, config = load_model_and_fit_scaler(
            CONFIG_PATH, 
            ckpt_path, 
            historical_features,
            feature_names
        )
        
        input_window = config['data']['input_window']
        if len(historical_features) < input_window:
            raise ValueError(f"Dati insufficienti: {len(historical_features)} < {input_window}")
            
        input_data = historical_features[-input_window:]
        predictions = predict(model, scaler, input_data, config)
        save_to_gsheet(gc, sheet_id, predictions, last_date, config)
        
        print("\n" + "=" * 70)
        print("✨ ESECUZIONE COMPLETATA")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
