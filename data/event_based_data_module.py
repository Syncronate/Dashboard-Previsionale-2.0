"""
DataModule con event-based validation e formato WIDE.
Versione con Feature Engineering integrato.
⚡ OTTIMIZZATO: Parallel data loading per GPU training
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path


class TimeSeriesGraphDataset(Dataset):
    """Dataset per serie temporali su grafo"""
    
    def __init__(self, data, timestamps, input_window, output_window):
        """
        Args:
            data: Tensor (timesteps, num_nodes, features)
            timestamps: Array di timestamp pandas/numpy
            input_window: Lunghezza input
            output_window: Lunghezza output
        """
        self.data = data
        self.timestamps = timestamps
        self.input_window = input_window
        self.output_window = output_window
        
        total_timesteps = len(data)
        self.num_samples = total_timesteps - input_window - output_window + 1
        
        if self.num_samples <= 0:
            raise ValueError(
                f"Dataset troppo corto! Serve almeno {input_window + output_window} timesteps, "
                f"ma hai solo {total_timesteps}"
            )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx
        end_input = start + self.input_window
        end_output = end_input + self.output_window
        
        # Input: (input_window, num_nodes, features)
        x = self.data[start:end_input]
        
        # Target: (output_window, num_nodes, 1) - solo livello (prima feature)
        y = self.data[end_input:end_output, :, 0:1]
        
        # Timestamp convertito a tensor long (unix timestamp in secondi)
        if self.timestamps is not None and end_input < len(self.timestamps):
            try:
                ts_value = self.timestamps[end_input]
                
                if hasattr(ts_value, 'timestamp'):
                    ts = int(ts_value.timestamp())
                elif isinstance(ts_value, np.datetime64):
                    ts = int(ts_value.astype('datetime64[s]').astype('int64'))
                else:
                    ts = end_input
            except Exception:
                ts = end_input
        else:
            ts = end_input
        
        return x, y, torch.tensor(ts, dtype=torch.long)


class EventBasedDataModule(pl.LightningDataModule):
    """
    DataModule con event-based validation per formato WIDE.
    
    Features:
    - Carica CSV formato italiano (sep=';', decimal=',')
    - Feature Engineering automatico
    - Trasforma da WIDE (colonna per nodo) a GRAFO (timesteps, nodes, features)
    - Esclude evento specifico per validation
    - Esclude dati corrotti
    - Normalizzazione Z-score
    - ⚡ Parallel data loading (num_workers + persistent_workers)
    """
    
    def __init__(self, config_path="config_bettolelle_final.yaml"):
        super().__init__()
        
        # Carica config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Parametri dati
        self.csv_path = self.config['data']['csv_path']
        self.input_window = self.config['data']['input_window']
        self.output_window = self.config['data']['output_window']
        self.batch_size = self.config['training']['batch_size']
        
        # Event validation
        self.val_event = self.config['data']['validation_event']
        self.exclude_corrupted = self.config['data']['exclude_corrupted_data']
        
        # Scaler per normalizzazione
        self.scaler = StandardScaler()
        
        # Datasets (creati in setup)
        self.train_dataset = None
        self.val_dataset = None
        
        # Metadata
        self.num_nodes = 5  # Fisso: 5 idrometri
        self.num_features = None  # Calcolato dopo trasformazione
        self.node_names = [
            'Serra_dei_Conti', 
            'Pianello_di_Ostra', 
            'Corinaldo', 
            'Bettolelle', 
            'Ponte_Garibaldi'
        ]
        
        # ====================================================================
        # ⚡ DATALOADER OPTIMIZATION - Legge da config['data']
        # ====================================================================
        self.num_workers = self.config['data'].get('num_workers', 0)
        self.pin_memory = self.config['data'].get('pin_memory', True)
        self.prefetch_factor = self.config['data'].get('prefetch_factor', 2)
        
        # ✅ FIX: persistent_workers SOLO se num_workers > 0
        if self.num_workers > 0:
            self.persistent_workers = self.config['data'].get('persistent_workers', False)
        else:
            self.persistent_workers = False
        
        # Log configurazione
        if self.num_workers > 0:
            print(f"⚡ DataLoader ottimizzato:")
            print(f"   - num_workers: {self.num_workers}")
            print(f"   - persistent_workers: {self.persistent_workers}")
            print(f"   - pin_memory: {self.pin_memory}")
            print(f"   - prefetch_factor: {self.prefetch_factor}")
        else:
            print(f"⚠️  DataLoader sequenziale (num_workers=0)")
    
    def prepare_data(self):
        """Verifica che il CSV esista (chiamato una volta)"""
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"❌ CSV non trovato: {self.csv_path}")
    
    def setup(self, stage=None):
        """Carica e prepara dati (chiamato per ogni processo)"""
        
        print("\n" + "="*80)
        print("📊 CARICAMENTO DATI - EVENT-BASED VALIDATION")
        print("="*80)
        
        # === CARICA CSV ===
        print(f"\n📁 Lettura: {self.csv_path}")
        
        df = pd.read_csv(
            self.csv_path,
            sep=self.config['data']['separator'],
            decimal=self.config['data']['decimal'],
            encoding=self.config['data']['encoding']
        )
        
        print(f"✅ Caricato: {df.shape}")
        
        # === CONVERTI TIMESTAMP ===
        timestamp_col = self.config['data']['timestamp_column']
        timestamp_format = self.config['data']['timestamp_format']
        
        df['datetime'] = pd.to_datetime(
            df[timestamp_col], 
            format=timestamp_format,
            errors='coerce'
        )
        
        # Rimuovi righe con timestamp invalidi
        df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
        
        print(f"✅ Timestamp convertiti: {len(df):,} righe valide")
        print(f"   Range: {df['datetime'].min()} → {df['datetime'].max()}")
        
        # ✅ FEATURE ENGINEERING
        if self.config.get('derived_features', {}).get('enable', True):
            print(f"\n🔧 Feature Engineering...")
            try:
                from preprocessing.feature_engineering import add_derived_features
                df = add_derived_features(df, self.config)
                print(f"✅ Feature derivate aggiunte: {df.shape}")
            except ImportError:
                print("⚠️  Modulo feature_engineering non trovato, skip")
            except Exception as e:
                print(f"⚠️  Errore feature engineering: {e}")
        
        # === TRASFORMA WIDE → GRAFO ===
        print(f"\n🔄 Trasformazione WIDE → GRAFO...")
        
        data_tensor, timestamps = self._wide_to_graph(df)
        
        print(f"✅ Shape finale: {data_tensor.shape}")
        print(f"   (timesteps={data_tensor.shape[0]}, nodes={data_tensor.shape[1]}, features={data_tensor.shape[2]})")
        
        self.num_features = data_tensor.shape[2]
        
        # === IDENTIFICA INDICI DA ESCLUDERE ===
        print(f"\n🚫 Identificazione dati da escludere...")
        
        exclude_indices = set()
        
        # 1. Evento validation
        event_start = self.val_event['start_row']
        event_end = self.val_event['end_row']
        margin_before = self.val_event.get('margin_before', 0)
        margin_after = self.val_event.get('margin_after', 0)
        
        # ✅ Gestione margini
        if margin_before == 0 and margin_after == 0:
            exclude_start = event_start
            exclude_end = event_end
            stats_start = event_start
            stats_end = event_end
            print(f"   Evento validation: righe {event_start}-{event_end}")
            print(f"   ⚠️  Margini = 0: assume range già comprensivo")
        else:
            stats_start = event_start
            stats_end = event_end
            
            exclude_start = max(0, event_start - margin_before)
            exclude_end = min(len(data_tensor), event_end + margin_after)
            
            print(f"   Evento validation CORE: righe {event_start}-{event_end}")
            print(f"   Con margini per exclude: {exclude_start}-{exclude_end}")
        
        val_indices = set(range(exclude_start, exclude_end + 1))
        
        # 2. Dati corrotti (sedimento/erba)
        if self.exclude_corrupted.get('enabled', False):
            corrupt_start = self.exclude_corrupted.get('start_row', len(data_tensor))
            corrupt_end = self.exclude_corrupted.get('end_row', len(data_tensor))
            
            if corrupt_end is None:
                corrupt_end = len(data_tensor)
            
            corrupt_indices = set(range(corrupt_start, corrupt_end))
            exclude_indices.update(corrupt_indices)
            
            print(f"   Dati corrotti: righe {corrupt_start}-{corrupt_end} ({len(corrupt_indices)} righe)")
            print(f"      Motivo: {self.exclude_corrupted['reason']}")
        
        # === CREA DATASETS ===
        print(f"\n📊 Creazione datasets...")
        
        # Full dataset
        full_dataset = TimeSeriesGraphDataset(
            data_tensor, timestamps, self.input_window, self.output_window
        )
        
        total_samples = len(full_dataset)
        print(f"   Samples totali potenziali: {total_samples:,}")
        
        # Indici training e validation
        train_indices = []
        val_sample_indices = []
        
        for i in range(total_samples):
            sample_rows = set(range(i, i + self.input_window + self.output_window))
            
            if sample_rows & val_indices:
                val_sample_indices.append(i)
            elif sample_rows & exclude_indices:
                continue
            else:
                train_indices.append(i)
        
        print(f"\n   ✅ Training samples:   {len(train_indices):,}")
        print(f"   ✅ Validation samples: {len(val_sample_indices):,}")
        print(f"   🚫 Esclusi (corrotti): {total_samples - len(train_indices) - len(val_sample_indices):,}")
        
        # Crea Subset
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_sample_indices)
        
        # === STATISTICHE EVENTO VALIDATION (denormalizzate) ===
        if len(val_sample_indices) > 0:
            val_data_rows = list(range(stats_start, stats_end + 1))
            
            val_levels_norm = data_tensor[val_data_rows, 3, 0].numpy()
            
            # Denormalizza
            if self.config['data']['normalize'] and hasattr(self.scaler, 'mean_'):
                feature_idx = 0
                mean = self.scaler.mean_[feature_idx]
                scale = self.scaler.scale_[feature_idx]
                val_levels = val_levels_norm * scale + mean
            else:
                val_levels = val_levels_norm
            
            print(f"\n🌊 EVENTO VALIDATION:")
            print(f"   📍 Range dati: {stats_start} → {stats_end} ({len(val_data_rows)} righe)")
            print(f"   📊 Livello min:  {val_levels.min():.2f}m")
            print(f"   📊 Livello max:  {val_levels.max():.2f}m")
            print(f"   📊 Livello medio: {val_levels.mean():.2f}m")
            print(f"   ⏱️  Durata:       {len(val_data_rows) * 0.5:.1f} ore")
            
            if timestamps is not None and len(timestamps) > stats_start:
                print(f"   📅 Inizio:       {timestamps[stats_start]}")
                
                max_idx_relative = val_levels.argmax()
                max_idx_absolute = stats_start + max_idx_relative
                if max_idx_absolute < len(timestamps):
                    print(f"   🔝 Picco:        {timestamps[max_idx_absolute]} (row {max_idx_absolute})")
        
        print("="*80 + "\n")
    
    def _wide_to_graph(self, df):
        """
        Trasforma da formato WIDE (1 colonna per nodo) a formato GRAFO.
        
        Returns:
            data: (timesteps, num_nodes, features) FloatTensor
            timestamps: Array datetime
        """
        
        num_timesteps = len(df)
        num_nodes = 5
        
        # Mapping nodi
        node_mapping = self.config['data']['node_columns']
        
        # Lista features per nodo
        features_per_node = []
        
        for node_name in self.node_names:
            node_config = node_mapping[node_name]
            
            node_features = []
            
            # Features specifiche nodo
            for feat_name, col_name in node_config.items():
                if col_name in df.columns:
                    node_features.append(df[col_name].values)
                else:
                    print(f"⚠️  Colonna non trovata: {col_name} (feature {feat_name} per {node_name})")
            
            # Stack features per questo nodo
            if len(node_features) > 0:
                features_per_node.append(np.stack(node_features, axis=1))
            else:
                features_per_node.append(np.zeros((num_timesteps, 1)))
        
        # Trova numero max features
        max_features_node = max(f.shape[1] for f in features_per_node)
        
        # Padda nodi con meno features
        for i in range(len(features_per_node)):
            if features_per_node[i].shape[1] < max_features_node:
                pad_width = max_features_node - features_per_node[i].shape[1]
                features_per_node[i] = np.pad(
                    features_per_node[i], 
                    ((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=0
                )
        
        # Stack tutti i nodi: (num_nodes, timesteps, features_per_node)
        node_data = np.stack(features_per_node, axis=0)
        
        # Transpose a (timesteps, num_nodes, features_per_node)
        node_data = node_data.transpose(1, 0, 2)
        
        # Aggiungi features GLOBALI
        global_features = []
        for feat_col in self.config['data']['global_features']:
            if feat_col in df.columns:
                global_features.append(df[feat_col].values)
        
        if len(global_features) > 0:
            global_data = np.stack(global_features, axis=1)
            global_broadcast = np.tile(
                global_data[:, np.newaxis, :], 
                (1, num_nodes, 1)
            )
            data = np.concatenate([node_data, global_broadcast], axis=2)
        else:
            data = node_data
        
        # Normalizza
        if self.config['data']['normalize']:
            original_shape = data.shape
            data_flat = data.reshape(-1, data.shape[2])
            data_flat = self.scaler.fit_transform(data_flat)
            data = data_flat.reshape(original_shape)
            print(f"   ✅ Dati normalizzati (Z-score)")
        
        # Converti a Tensor
        data_tensor = torch.FloatTensor(data)
        
        # Timestamps
        timestamps = df['datetime'].values if 'datetime' in df.columns else None
        
        return data_tensor, timestamps

    def train_dataloader(self):
        """
        DataLoader training con ottimizzazioni parallele
        """
        # ✅ FIX: pin_memory solo se CUDA disponibile
        use_pin_memory = self.pin_memory and torch.cuda.is_available()
        
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': use_pin_memory,
        }
        
        # ✅ persistent_workers e prefetch_factor SOLO se num_workers > 0
        if self.num_workers > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
        
        return DataLoader(self.train_dataset, **loader_kwargs)
    
    def val_dataloader(self):
        """
        DataLoader validation con ottimizzazioni parallele
        """
        use_pin_memory = self.pin_memory and torch.cuda.is_available()
        
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,  # ← Validation NON va mescolato!
            'num_workers': self.num_workers,
            'pin_memory': use_pin_memory,
        }
        
        if self.num_workers > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
        
        return DataLoader(self.val_dataset, **loader_kwargs)
    
    def test_dataloader(self):
        """DataLoader test (usa validation per ora)"""
        return self.val_dataloader()
    
    def inverse_transform(self, data):
        """Riporta dati dalla scala normalizzata a quella originale"""
        if hasattr(self.scaler, 'inverse_transform'):
            return self.scaler.inverse_transform(data)
        return data
    
    def get_event_data(self):
        """Ritorna i dati raw dell'evento validation"""
        val_start = self.val_event['start_row']
        val_end = self.val_event['end_row']
        
        return {
            'start': val_start,
            'end': val_end,
            'duration_hours': (val_end - val_start) * 0.5
        }
