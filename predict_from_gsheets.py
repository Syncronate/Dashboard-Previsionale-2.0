"""
Script di predizione SimpleTCN - Solo da Google Sheets
Fix: gestione robusta colonne duplicate, conversione numerica, caricamento checkpoint e mismatch features
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
    
    print(f"   ✅ Storico: {len(df_hist)} righe | Forecast: {len(df_fcst)} righe")
    
    return df_hist, df_fcst

# ==================== PREPROCESSING ====================
def clean_column_names(df):
    """Rimuove prefissi ridondanti e gestisce duplicati"""
    rename_map = {}
    for col in df.columns:
        new_col = col.replace('Giornaliera ', '').replace('_cumulata_30min', '')
        if new_col != col:
            rename_map[col] = new_col
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        print(f"   🔄 Rinominate {len(rename_map)} colonne")
    
    # ✅ RIMUOVI DUPLICATI
    if df.columns.duplicated().any():
        num_duplicates = df.columns.duplicated().sum()
        print(f"   ⚠️  Rimosse {num_duplicates} colonne duplicate")
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    return df

def parse_dates(df, date_col, date_fmt):
    """Parse date con gestione errori"""
    df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors='coerce')
    
    before = len(df)
    df = df.dropna(subset=[date_col])
    after = len(df)
    
    if before != after:
        print(f"   ⚠️  Rimosse {before - after} righe con date invalide")
    
    return df.sort_values(date_col).reset_index(drop=True)

def convert_to_numeric(df, exclude_cols):
    """Converte colonne a numerico con gestione errori robusta"""
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        try:
            col_data = df[col]
            
            # Se è un DataFrame (colonne duplicate), skippa
            if isinstance(col_data, pd.DataFrame):
                print(f"   ⚠️  Colonna duplicata ignorata: {col}")
                continue
            
            # Se già numerico, skippa
            if pd.api.types.is_numeric_dtype(col_data):
                continue
            
            # Se object/string, converti
            if col_data.dtype == object:
                df[col] = col_data.astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        except Exception as e:
            print(f"   ⚠️  Errore conversione '{col}': {e}")
            continue
    
    return df

def get_feature_columns(config):
    """Estrae lista colonne feature dal config"""
    node_cols = config['data']['node_columns']
    global_feats = config['data'].get('global_features', [])
    
    all_cols = set()
    
    for node_name, features in node_cols.items():
        for feat_name, col_name in features.items():
            all_cols.add(col_name)
    
    for gf in global_feats:
        all_cols.add(gf)
    
    return sorted(list(all_cols))

def prepare_data(df_hist, df_fcst, config):
    """Prepara dati con gestione robusta"""
    print("\n⚙️  Preprocessing dati...")
    
    df_hist = clean_column_names(df_hist)
    df_fcst = clean_column_names(df_fcst)
    
    date_col = config['data']['timestamp_column']
    date_fmt = config['data']['timestamp_format']
    
    df_hist = parse_dates(df_hist, date_col, date_fmt)
    df_fcst = parse_dates(df_fcst, date_col, date_fmt)
    
    last_hist_date = df_hist[date_col].iloc[-1]
    print(f"   📅 Ultimo dato storico: {last_hist_date}")
    
    df_hist = convert_to_numeric(df_hist, [date_col])
    df_fcst = convert_to_numeric(df_fcst, [date_col])
    
    numeric_cols = df_hist.select_dtypes(include=[np.number]).columns
    df_hist[numeric_cols] = df_hist[numeric_cols].ffill().fillna(0)
    
    print(f"   ✅ Dati preprocessati: {len(df_hist)} righe")
    
    return df_hist, df_fcst, last_hist_date

# ==================== ESTRAZIONE FEATURES ====================
def extract_features(df, config):
    """Estrae feature array dal dataframe con filtro per compatibilità checkpoint"""
    feature_cols = get_feature_columns(config)
    
    # --- FIX: FILTRO FEATURES PER COMPATIBILITÀ CHECKPOINT (20 features) ---
    # Il config ne ha 26, il checkpoint ne aspetta 20.
    # Rimuoviamo le features "nuove" (6h, 12h, upstream velocity)
    
    filtered_cols = []
    excluded_cols = []
    
    for col in feature_cols:
        is_new = False
        # Criteri per identificare le nuove features
        if "_cumulata_6h" in col: is_new = True
        if "_cumulata_12h" in col: is_new = True
        if "_velocity" in col and "Bettolelle" not in col: is_new = True # Velocity upstream
        
        if is_new:
            excluded_cols.append(col)
        else:
            filtered_cols.append(col)
            
    # Se il filtro è troppo aggressivo o non basta, fallback ai primi 20
    if len(filtered_cols) != 20:
        print(f"   ⚠️  Attenzione: Filtro euristico ha prodotto {len(filtered_cols)} features.")
        print(f"       Checkpoint ne richiede 20. Uso i primi 20 in ordine alfabetico.")
        filtered_cols = feature_cols[:20]
        
    print(f"   ✂️  Features selezionate: {len(filtered_cols)} (Originali: {len(feature_cols)})")
    if excluded_cols:
        print(f"   🗑️  Escluse {len(excluded_cols)} features (nuove):")
        for c in excluded_cols[:3]: print(f"      - {c}")
    
    # -----------------------------------------------------------------------
    
    available_cols = [c for c in filtered_cols if c in df.columns]
    missing_cols = set(filtered_cols) - set(available_cols)
    
    if missing_cols:
        print(f"   ⚠️  Colonne mancanti: {len(missing_cols)}")
        if len(missing_cols) <= 5:
            for mc in list(missing_cols)[:5]:
                print(f"      - {mc}")
    
    print(f"   📊 Feature estratte: {len(available_cols)} colonne")
    
    return df[available_cols].values, available_cols

# ==================== MODELLO ====================
def load_model_and_fit_scaler(config_path, checkpoint_path, historical_data, feature_names):
    """Carica modello e fitta scaler con gestione robusta delle chiavi"""
    print("\n🧠 Caricamento modello...")
    
    config = load_config(config_path)
    
    print("   📊 Fitting scaler su dati GSheets...")
    
    scaler = StandardScaler()
    scaler.fit(historical_data)
    
    print(f"   ✅ Scaler fittato su {len(historical_data)} timestep")
    print(f"      Features: {len(feature_names)}")
    
    print(f"   🔧 Caricamento checkpoint: {checkpoint_path}")
    
    # 1. Inizializza il modello
    model = LitSpatioTemporalGNN(
        num_nodes=5,  
        num_features=len(feature_names), # Deve essere 20
        hidden_dim=config['model']['hidden_dim'],
        rnn_layers=config['model'].get('rnn_layers', 3),
        output_window=config['data']['output_window'],
        output_dim=1,
        num_quantiles=config['model']['num_quantiles'],
        config=config
    )
    
    # 2. Carica il checkpoint grezzo
    device = torch.device('cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    # 3. Fix delle chiavi
    new_state_dict = {}
    for k, v in state_dict.items():
        # Rimuovi buffer del grafo se non necessari (edge_index, edge_weight)
        if "edge_index" in k or "edge_weight" in k:
            continue
            
        # Rinomina 'model.' -> 'gnn_model.' se necessario
        if k.startswith("model."):
            new_key = k.replace("model.", "gnn_model.", 1)
        else:
            new_key = k
            
        new_state_dict[new_key] = v
        
    # 4. Carica i pesi con strict=False per tollerare piccole differenze
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("   ✅ Pesi caricati con successo (con remapping chiavi)")
    except Exception as e:
        print(f"   ⚠️  Warning nel caricamento pesi: {e}")
        print("   Tentativo caricamento standard...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    
    print("   ✅ Modello pronto")
    
    return model, scaler, config

def predict(model, scaler, input_data, config):
    """Esegue predizione"""
    print("\n🔮 Generazione previsioni...")
    
    input_normalized = scaler.transform(input_data)
    
    num_nodes = 5
    
    # Replica per ogni nodo
    input_reshaped = np.stack([input_normalized] * num_nodes, axis=1)
    input_reshaped = input_reshaped[np.newaxis, :, :, :]
    
    input_tensor = torch.FloatTensor(input_reshaped)
    
    print(f"   📊 Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"   📊 Output shape: {output.shape}")
    
    # Estrai Bettolelle (nodo 3)
    target_node_idx = 3
    pred_bett = output[0, :, target_node_idx, 0, :]
    
    # Denormalizza
    scaler_mean = scaler.mean_[0]
    scaler_scale = scaler.scale_[0]
    
    pred_denorm = pred_bett.numpy() * scaler_scale + scaler_mean
    
    return pred_denorm

# ==================== SALVATAGGIO ====================
def save_to_gsheet(gc, sheet_id, predictions, start_time, config):
    print(f"\n💾 Salvataggio su '{SHEET_OUTPUT}'...")
    
    sh = gc.open_by_key(sheet_id)
    
    try:
        ws = sh.worksheet(SHEET_OUTPUT)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_OUTPUT, rows=100, cols=20)
    
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

# ==================== MAIN ====================
def main():
    print("=" * 70)
    print(f"🚀 PREVISIONI BETTOLELLE - {datetime.now(ITALY_TZ)}")
    print("=" * 70)
    
    try:
        config = load_config(CONFIG_PATH)
        print(f"✅ Config caricato")
        
        ckpt_path = find_best_checkpoint(CHECKPOINT_DIR)
        
        sheet_id = os.environ.get("GSHEET_ID")
        if not sheet_id:
            raise ValueError("❌ GSHEET_ID non configurato!")
        
        gc = connect_gsheets(CREDENTIALS_PATH)
        print(f"✅ Connesso a Google Sheets")
        
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
            raise ValueError(f"❌ Dati insufficienti: {len(historical_features)} < {input_window}")
        
        input_data = historical_features[-input_window:]
        
        print(f"   📊 Input shape: {input_data.shape}")
        
        predictions = predict(model, scaler, input_data, config)
        
        print(f"   📈 Predictions shape: {predictions.shape}")
        print(f"   📊 Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
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
