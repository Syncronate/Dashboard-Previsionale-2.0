"""
Feature engineering per flood forecasting - Bacino Bettolelle
Versione ROBUSTA: si adatta automaticamente al config
"""

import pandas as pd
import numpy as np


def add_derived_features(df, config):
    """
    Aggiunge features derivate al dataframe.
    
    Args:
        df: DataFrame con dati raw
        config: Config YAML
        
    Returns:
        df con features aggiuntive
    """
    
    print("\n🔧 Feature Engineering...")
    
    node_cols = config['data']['node_columns']
    
    # ===================================================================
    # 1. TEMPORAL DERIVATIVES (Velocity & Acceleration)
    # ===================================================================
    
    print("   📈 Calcolo derivate temporali...")
    
    for node_name, features in node_cols.items():
        if 'livello' in features:
            livello_col = features['livello']
            
            if livello_col not in df.columns:
                print(f"      ⚠️  {node_name}: colonna livello non trovata: {livello_col}")
                continue
            
            # Velocity (prima derivata)
            velocity_col = f"{livello_col}_velocity"
            df[velocity_col] = df[livello_col].diff().fillna(0)
            
            # Acceleration (seconda derivata)
            accel_col = f"{livello_col}_accel"
            df[accel_col] = df[velocity_col].diff().fillna(0)
            
            print(f"      ✅ {node_name}: velocity + acceleration")
    
    # ===================================================================
    # 2. LAG FEATURES (ai tempi di corrivazione)
    # ===================================================================
    
    print("   ⏱️  Calcolo lag features...")
    
    # Trova features Bettolelle in modo robusto
    bett_features = node_cols.get('Bettolelle', {})
    
    # Cerca livello Bettolelle
    bett_livello = bett_features.get('livello')
    
    # Cerca pluvio Bettolelle (QUALSIASI chiave che contiene 'pluvio')
    bett_pluvio = None
    
    # Ordine di preferenza
    search_keys = ['pluvio_bett', 'pluvio_cumul', 'pluvio_bett_3h', 'pluvio_3h', 'pluvio_cumul_3h']
    
    for key in search_keys:
        if key in bett_features:
            bett_pluvio = bett_features[key]
            print(f"      🔍 Trovata chiave pluvio: '{key}' → {bett_pluvio[:50]}...")
            break
    
    # Fallback: cerca qualsiasi chiave con 'pluvio'
    if bett_pluvio is None:
        for key, value in bett_features.items():
            if 'pluvio' in key.lower():
                bett_pluvio = value
                print(f"      🔍 Trovata chiave pluvio (fallback): '{key}' → {value[:50]}...")
                break
    
    if bett_pluvio is None:
        print(f"      ⚠️  WARNING: Nessuna colonna pluvio per Bettolelle!")
        print(f"         Chiavi disponibili: {list(bett_features.keys())}")
    
    # Lag per Bettolelle livello
    if bett_livello and bett_livello in df.columns:
        df[f"{bett_livello}_lag2"] = df[bett_livello].shift(2).fillna(method='bfill')
        df[f"{bett_livello}_lag4"] = df[bett_livello].shift(4).fillna(method='bfill')
        print(f"      ✅ Bettolelle livello: lag2, lag4")
    
    # Lag per Bettolelle pluvio
    if bett_pluvio and bett_pluvio in df.columns:
        df[f"{bett_pluvio}_lag4"] = df[bett_pluvio].shift(4).fillna(0)
        print(f"      ✅ Bettolelle pluvio: lag4")
    
    # Lag per Serra (tempo corrivazione = 2h = 4 timestep)
    serra_features = node_cols.get('Serra_dei_Conti', {})
    serra_livello = serra_features.get('livello')
    
    if serra_livello and serra_livello in df.columns:
        df[f"{serra_livello}_lag4"] = df[serra_livello].shift(4).fillna(method='bfill')
        print(f"      ✅ Serra livello: lag4")
    
    # ===================================================================
    # 3. PRECIPITATION MULTI-SCALE (opzionale - skippa se troppo complesso)
    # ===================================================================
    
    print("   🌧️  Calcolo pioggia multi-scala...")
    
    # Per ora skippiamo (troppo complesso senza raw data)
    print("      ⚠️  Skipped (non implementato in questa versione)")
    
    # ===================================================================
    # 4. INTERACTION FEATURES
    # ===================================================================
    
    print("   🔗 Calcolo features interazione...")
    
    # Pioggia × Soil Moisture
    soil_col = 'soil_moisture_28_to_100cm (m³/m³)'
    if soil_col in df.columns and bett_pluvio and bett_pluvio in df.columns:
        df['pluvio_x_soil_moisture'] = df[bett_pluvio] * df[soil_col]
        print(f"      ✅ pluvio × soil_moisture")
    
    # Pioggia × Regime Dummy
    dummy_col = 'Variabile Dummy'
    if dummy_col in df.columns and bett_pluvio and bett_pluvio in df.columns:
        df['pluvio_x_regime_dummy'] = df[bett_pluvio] * df[dummy_col]
        print(f"      ✅ pluvio × regime_dummy")
    
    # Livello × Velocity (Momentum)
    if bett_livello and bett_livello in df.columns:
        velocity_col = f"{bett_livello}_velocity"
        if velocity_col in df.columns:
            df['livello_x_velocity'] = df[bett_livello] * df[velocity_col]
            print(f"      ✅ livello × velocity")
    
    print(f"\n✅ Feature engineering completato!")
    print(f"   Colonne totali: {len(df.columns)}")
    
    # Conta nuove features
    original_cols = config['data'].get('_original_columns', set())
    if not original_cols:
        # Stima basata su pattern nomi
        new_cols = [c for c in df.columns if '_velocity' in c or '_accel' in c or 
                    '_lag' in c or '_x_' in c or 'pluvio_x' in c or 'livello_x' in c]
    else:
        new_cols = set(df.columns) - original_cols
    
    print(f"   Nuove features create: {len(new_cols)}")
    
    return df
