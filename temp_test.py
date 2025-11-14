import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatioTemporalAGCLSTM(nn.Module):
    """
    Modello ibrido GCN-GRU con meccanismo di attenzione.

    Flusso dati: Input → GCN (per ogni t) → Aggregazione Nodi → GRU →
                 Attention → Concatenazione → FC → Output
    """
    def __init__(self, num_nodes, num_features, hidden_dim, rnn_layers,
                 output_window, output_dim, num_quantiles=1, dropout=0.2):
        super().__init__()
        assert num_nodes > 0, "num_nodes must be positive"
        assert num_features > 0, "num_features must be positive"
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert rnn_layers >= 1, "rnn_layers must be at least 1"
        assert output_window > 0, "output_window must be positive"
        assert output_dim >= 1, "output_dim must be at least 1"
        assert num_quantiles >= 1, "num_quantiles must be at least 1"
        assert 0.0 <= dropout < 1.0, "dropout must be between 0.0 and 1.0"

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_window = output_window
        self.output_dim = output_dim
        self.num_quantiles = num_quantiles

        self.gcn = GCNConv(num_features, hidden_dim)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        self.attention = self.Attention(hidden_dim)

        self.fc = nn.Linear(
            hidden_dim * 2,
            output_window * output_dim * num_quantiles
        )

        self.dropout = nn.Dropout(dropout)

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
            seq_len = encoder_outputs.size(1)
            hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
            attention = self.v(energy).squeeze(2)
            attention_weights = F.softmax(attention, dim=1)
            context_vector = torch.bmm(
                attention_weights.unsqueeze(1),
                encoder_outputs
            ).squeeze(1)
            return context_vector, attention_weights

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, seq_len, num_nodes, num_features = x.shape

        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            x_t_flat = x_t.reshape(batch_size * self.num_nodes, self.num_features)
            edge_index_batch = edge_index.repeat(1, batch_size) + torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(edge_index.size(1)) * self.num_nodes
            edge_weight_batch = edge_weight.repeat(batch_size) if edge_weight is not None else None
            gcn_out = self.gcn(x_t_flat, edge_index_batch, edge_weight=edge_weight_batch)
            gcn_out = F.relu(gcn_out)
            gcn_out_reshaped = gcn_out.view(batch_size, self.num_nodes, self.hidden_dim)
            gcn_outputs.append(gcn_out_reshaped)

        x_gcn_seq = torch.stack(gcn_outputs, dim=1)
        rnn_input = x_gcn_seq.mean(dim=2)
        rnn_outputs, rnn_hidden = self.rnn(rnn_input)
        last_hidden = rnn_hidden[-1]
        context_vector, attention_weights = self.attention(last_hidden, rnn_outputs)
        last_rnn_output = rnn_outputs[:, -1, :]
        combined = torch.cat([last_rnn_output, context_vector], dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)

        if self.num_quantiles > 1:
            output = output.reshape(batch_size, self.output_window, self.output_dim, self.num_quantiles)
        else:
            output = output.reshape(batch_size, self.output_window, self.output_dim)

        # Squeeze the last dimension if num_quantiles is 1 and output_dim is 1
        if self.num_quantiles == 1 and self.output_dim == 1:
            output = output.squeeze(-1)

        return output, attention_weights

def test_agclstm_architecture():
    """Test rapido dell'architettura."""
    batch_size, seq_len, num_nodes, num_features = 4, 10, 5, 3
    hidden_dim, rnn_layers = 32, 2
    output_window, output_dim = 5, 1

    model = SpatioTemporalAGCLSTM(
        num_nodes=num_nodes,
        num_features=num_features,
        hidden_dim=hidden_dim,
        rnn_layers=rnn_layers,
        output_window=output_window,
        output_dim=output_dim
    )

    x = torch.randn(batch_size, seq_len, num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    output, attention = model(x, edge_index)

    # Adjusted expected shape for when output_dim is 1
    expected_output_shape = (batch_size, output_window)
    assert output.shape == expected_output_shape, \
        f"Output shape errato: {output.shape}, atteso {expected_output_shape}"
    assert attention.shape == (batch_size, seq_len), \
        f"Attention shape errato: {attention.shape}"

    assert torch.allclose(attention.sum(dim=1), torch.ones(batch_size), atol=1e-5), \
        "Pesi attenzione non normalizzati"

    print("✅ Test architettura superato!")

if __name__ == "__main__":
    test_agclstm_architecture()
