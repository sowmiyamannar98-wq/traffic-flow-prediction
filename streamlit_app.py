import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

st.set_page_config(page_title="Traffic Flow Prediction", layout="wide")

# ── Model definition (must match training) ───────────────────────────────────

class NConv(nn.Module):
    def forward(self, x, A):
        return torch.einsum('ncvl,vw->ncwl', x, A).contiguous()

class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, (1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear((order * support_len + 1) * c_in, c_out)
        self.dropout, self.order = dropout, order
    def forward(self, x, supports):
        out = [x]
        for A in supports:
            x1 = self.nconv(x, A); out.append(x1)
            for _ in range(self.order - 1):
                x1 = self.nconv(x1, A); out.append(x1)
        return F.dropout(self.mlp(torch.cat(out, dim=1)), self.dropout, training=self.training)

class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, in_dim=2, out_dim=12, residual=32, dilation=32,
                 skip=256, end=512, blocks=4, layers=2, kernel=2,
                 dropout=0.3, supports=None, adp_emb=10):
        super().__init__()
        self.blocks, self.layers, self.dropout = blocks, layers, dropout
        self.supports = supports if supports is not None else []
        self.start_conv = nn.Conv2d(in_dim, residual, (1, 1))
        self.E1 = nn.Parameter(torch.randn(num_nodes, adp_emb))
        self.E2 = nn.Parameter(torch.randn(adp_emb, num_nodes))
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.residual_convs, self.skip_convs = nn.ModuleList(), nn.ModuleList()
        self.bn, self.gconv = nn.ModuleList(), nn.ModuleList()
        for _ in range(blocks):
            d = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.gate_convs.append(  nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.residual_convs.append(nn.Conv2d(dilation, residual, (1, 1)))
                self.skip_convs.append(    nn.Conv2d(dilation, skip,     (1, 1)))
                self.bn.append(nn.BatchNorm2d(residual))
                self.gconv.append(GCN(dilation, residual, dropout, support_len=len(self.supports)+1))
                d *= 2
        self.end_conv_1 = nn.Conv2d(skip, end,     (1, 1))
        self.end_conv_2 = nn.Conv2d(end,  out_dim, (1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        if x.size(3) < 13:
            x = F.pad(x, (13 - x.size(3), 0, 0, 0))
        x = self.start_conv(x); skip = 0
        adp = F.softmax(F.relu(torch.mm(self.E1, self.E2)), dim=1)
        supports = self.supports + [adp]
        for i in range(self.blocks * self.layers):
            res = x
            f = torch.tanh(self.filter_convs[i](res))
            g = torch.sigmoid(self.gate_convs[i](res))
            x = f * g
            s = self.skip_convs[i](x)
            skip = s if isinstance(skip, int) else s + skip[..., -s.size(3):]
            x = self.gconv[i](x, supports)
            x = x + res[..., -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x.squeeze(-1).permute(0, 2, 1)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    supports = []
    if ckpt.get("P_fwd") is not None:
        supports = [torch.tensor(ckpt["P_fwd"]), torch.tensor(ckpt["P_bwd"])]
    model = GraphWaveNet(
        num_nodes=ckpt["num_nodes"], in_dim=2, out_dim=ckpt["seq_out"],
        supports=supports
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

@st.cache_data
def load_processed(npz_path):
    d = np.load(npz_path)
    return d["X_te"], d["Y_te"]

def predict(model, x_window, mean, std):
    with torch.no_grad():
        inp = torch.tensor(x_window[None], dtype=torch.float32)
        out = model(inp).squeeze(0)
        return out.numpy() * std + mean

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Traffic Flow Prediction — PEMS-BAY")
st.markdown("Graph WaveNet · Spatio-Temporal GNN · 325 Bay Area Sensors")

CKPT = "gwn_pems_bay.ckpt"
NPZ  = "pems_bay_processed.npz"

if not os.path.exists(CKPT):
    st.error(f"`{CKPT}` not found. Place the checkpoint in the same folder as this app.")
    st.stop()

model, ckpt = load_checkpoint(CKPT)
mean_val = float(ckpt["mean"])
std_val  = float(ckpt["std"])
num_nodes = ckpt["num_nodes"]
seq_out   = ckpt["seq_out"]

st.sidebar.header("Settings")
sensor_idx = st.sidebar.slider("Sensor index", 0, num_nodes - 1, 50)
horizon_map = {"15 min (step 3)": 2, "30 min (step 6)": 5, "60 min (step 12)": 11}
horizon_label = st.sidebar.selectbox("Forecast horizon", list(horizon_map.keys()))
horizon = horizon_map[horizon_label]

if os.path.exists(NPZ):
    X_te, Y_te = load_processed(NPZ)
    sample_idx = st.sidebar.slider("Test sample index", 0, len(X_te) - 1, 0)
    x_win = X_te[sample_idx]
    pred_all = predict(model, x_win, mean_val, std_val)
    true_speed = Y_te[sample_idx, :, sensor_idx]
    pred_speed = pred_all[:, sensor_idx]

    col1, col2, col3 = st.columns(3)
    mae  = np.mean(np.abs(pred_speed - true_speed))
    rmse = np.sqrt(np.mean((pred_speed - true_speed) ** 2))
    mape = np.mean(np.abs((pred_speed - true_speed) / np.clip(true_speed, 1e-4, None))) * 100
    col1.metric("MAE",      f"{mae:.3f} mph")
    col2.metric("RMSE",     f"{rmse:.3f} mph")
    col3.metric("MAPE",     f"{mape:.2f}%")

    st.subheader(f"Sensor {sensor_idx} — all 12 forecast steps")
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = [f"{(i+1)*5}m" for i in range(seq_out)]
    ax.plot(steps, true_speed, marker="o", label="Ground truth", lw=2)
    ax.plot(steps, pred_speed, marker="s", label="Prediction",   lw=2, alpha=0.85)
    ax.axvline(x=steps[horizon], color="gray", ls="--", alpha=0.6,
               label=f"Selected horizon ({horizon_label})")
    ax.set_xlabel("Forecast step"); ax.set_ylabel("Speed (mph)")
    ax.set_title(f"Sensor {sensor_idx} — 1-hour ahead forecast")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("All-sensor speed snapshot at selected horizon")
    pred_horizon = pred_all[horizon]
    true_horizon = Y_te[sample_idx, horizon]
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(true_horizon,  alpha=0.7, label="Ground truth", lw=1)
    ax2.plot(pred_horizon,  alpha=0.7, label="Prediction",   lw=1)
    ax2.set_xlabel("Sensor index"); ax2.set_ylabel("Speed (mph)")
    ax2.set_title(f"All 325 sensors — {horizon_label} forecast")
    ax2.legend(); ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    with st.expander("Raw prediction table"):
        df_out = pd.DataFrame({
            "Step": steps,
            "Predicted (mph)": pred_speed.round(2),
            "Ground truth (mph)": true_speed.round(2),
            "Error (mph)": (pred_speed - true_speed).round(2),
        })
        st.dataframe(df_out, use_container_width=True)
else:
    st.warning(
        f"`{NPZ}` not found — place the processed dataset file alongside the app "
        "to enable interactive predictions."
    )
    st.info(
        "Checkpoint loaded successfully. Model has "
        f"**{sum(p.numel() for p in model.parameters()):,}** trainable parameters "
        f"and covers **{num_nodes}** sensors."
    )
