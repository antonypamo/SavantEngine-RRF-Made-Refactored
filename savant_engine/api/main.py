import os
import sys
import math
from typing import Optional, Dict, Any

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import joblib

# ============================
# Configuración de modelos
# ============================
ENCODER_MODEL_ID   = "antonypamo/RRFSAVANTMADE"
META_LOGIT_REPO    = "antonypamo/RRFSavantMetaLogit"
META_LOGIT_FILENAME = "logreg_rrf_savant_15.joblib"

print("🔄 [Startup] Cargando encoder RRFSAVANTMADE...", flush=True)
try:
    encoder = SentenceTransformer(ENCODER_MODEL_ID)
    print("✅ [Startup] Encoder cargado.", flush=True)
except Exception as e:
    print(f"❌ [Startup] Error al cargar encoder: {e}", file=sys.stderr, flush=True)
    raise

print("🔄 [Startup] Descargando meta-logit desde HF Hub...", flush=True)
try:
    meta_logit_path = hf_hub_download(
        repo_id=META_LOGIT_REPO,
        filename=META_LOGIT_FILENAME,
        token=os.environ.get("HF_TOKEN"),  # si el repo es público, puede ser None
    )
    print(f"🔄 [Startup] Cargando modelo meta-logit '{META_LOGIT_FILENAME}'...", flush=True)
    meta_logit = joblib.load(meta_logit_path)
    try:
        print(f"🔎 [Startup] Meta-logit espera {meta_logit.n_features_in_} features.", flush=True)
    except Exception:
        print("⚠️ [Startup] No se pudo leer n_features_in_.", flush=True)
    print("✅ [Startup] Meta-logit cargado.", flush=True)
except Exception as e:
    print(f"❌ [Startup] Error al cargar meta-logit: {e}", file=sys.stderr, flush=True)
    raise

# ============================
# Geometría icosaédrica Φ12.0
# ============================

phi = (1 + np.sqrt(5)) / 2
nodes = np.array([
    [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
], dtype=float)
nodes /= norm(nodes, axis=1, keepdims=True)
N = nodes.shape[0]  # 12 nodos

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_IN(M, N_sites):
    return np.kron(M, np.eye(N_sites, dtype=complex))


def site_op(block_2x2, i, j, N_sites):
    K = np.zeros((N_sites, N_sites), dtype=complex)
    K[i, j] = 1.0
    return np.kron(K, block_2x2)


def geodesic_kernel(nodes, sigma=0.618, alpha_log=0.10):
    diff = nodes[:, None, :] - nodes[None, :, :]
    dist = norm(diff, axis=-1)

    W = np.exp(-(dist ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 0.0)

    if alpha_log > 0.0:
        corr = 1.0 + alpha_log * np.log1p(dist ** 2)
        corr[range(N), range(N)] = 1.0
        W = W / corr

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def u1_edge_phases(nodes, flux_vector=(0.0, 0.0, 0.0), q=1.0, gauge_scale=1.0):
    A = gauge_scale * np.asarray(flux_vector, dtype=float)
    midpoints = (nodes[:, None, :] + nodes[None, :, :]) / 2.0
    theta = (midpoints @ A).astype(float)
    theta = 0.5 * (theta - theta.T)
    return theta * q


def build_dirac_hamiltonian(
    m=0.25,
    v=1.0,
    sigma=0.618,
    alpha_log=0.10,
    q=1.0,
    flux_vector=(0.0, 0.0, 0.0),
    gauge_scale=0.0,
):
    W = geodesic_kernel(nodes, sigma=sigma, alpha_log=alpha_log)

    if gauge_scale != 0.0 and any(flux_vector):
        theta = u1_edge_phases(nodes, flux_vector=flux_vector,
                               q=q, gauge_scale=gauge_scale)
        U = np.exp(1j * theta)
    else:
        U = np.ones((N, N), dtype=complex)

    H = np.kron(np.eye(N, dtype=complex), m * sigma_z)

    diff = nodes[:, None, :] - nodes[None, :, :]
    dist = norm(diff, axis=-1) + 1e-12
    d_hat = diff / dist[..., None]

    for i in range(N):
        for j in range(N):
            if i == j or W[i, j] == 0:
                continue
            nvec = d_hat[i, j]
            S = (nvec[0] * sigma_x +
                 nvec[1] * sigma_y +
                 nvec[2] * sigma_z)
            H += v * W[i, j] * U[i, j] * site_op(S, i, j, N)

    H = 0.5 * (H + H.conj().T)
    return H


def site_probs(psi):
    N2 = psi.shape[0]
    n = N2 // 2
    psi_mat = psi.reshape(n, 2)
    return np.sum(np.abs(psi_mat) ** 2, axis=1).real


def chirality(psi):
    S = kron_IN(sigma_z, N)
    return float(np.vdot(psi, S @ psi).real)


def energy_expectation(psi, H):
    return float(np.vdot(psi, H @ psi).real)


def spatial_entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)).real)


def evolve_dirac_shell(psi0, H, dt=0.05, steps=100, record_every=25):
    U = expm(-1j * dt * H)
    psi = psi0.copy()

    probs_hist = []
    energy_hist = []
    chir_hist = []
    ent_hist = []

    for t in range(steps + 1):
        if t % record_every == 0:
            p = site_probs(psi)
            probs_hist.append(p)
            energy_hist.append(energy_expectation(psi, H))
            chir_hist.append(chirality(psi))
            ent_hist.append(spatial_entropy(p))

        psi = U @ psi
        psi /= np.sqrt(np.vdot(psi, psi))

    return {
        "probs": np.array(probs_hist, dtype=float),
        "energy": np.array(energy_hist, dtype=float),
        "chirality": np.array(chir_hist, dtype=float),
        "entropy": np.array(ent_hist, dtype=float),
        "dt": dt,
        "record_every": record_every,
    }

# ============================
# Core RRF: embeddings + features + scores
# ============================

def get_embedding(text: str) -> np.ndarray:
    emb = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return emb[0]


def compute_rrf_features(prompt: str, answer: str) -> Dict[str, float]:
    # Embeddings
    e_p = get_embedding(prompt)
    e_a = get_embedding(answer)

    cosine_pa = float(np.dot(e_p, e_a))
    len_ratio = len(answer) / (len(prompt) + 1.0)

    # Simulación Dirac shell determinista (semilla por prompt+answer)
    rng = np.random.default_rng(abs(hash(prompt + answer)) % (2 ** 32))
    vec = rng.normal(0, 1, (2 * N,)) + 1j * rng.normal(0, 1, (2 * N,))
    vec /= np.sqrt(np.vdot(vec, vec))
    psi0 = vec

    H = build_dirac_hamiltonian(
        m=0.25, v=1.0, sigma=0.618,
        alpha_log=0.10, q=1.0,
        flux_vector=(0.0, 0.0, 0.0),
        gauge_scale=0.0,
    )

    out = evolve_dirac_shell(psi0, H, dt=0.05, steps=100, record_every=25)

    entropy = out["entropy"]
    energy = out["energy"]
    chir = out["chirality"]

    S_final = float(entropy[-1])
    S_initial = float(entropy[0])
    S_delta = S_final - S_initial
    C_final = float(chir[-1])
    E_mean = float(np.mean(energy))
    E_std = float(np.std(energy))

    # Núcleo de 7 features
    feats: Dict[str, float] = {
        "cosine_pa": cosine_pa,
        "len_ratio": len_ratio,
        "dirac_entropy_final": S_final,
        "dirac_entropy_delta": S_delta,
        "dirac_chirality_final": C_final,
        "dirac_energy_mean": E_mean,
        "dirac_energy_std": E_std,
    }

    # Derivadas para llegar a 15 (igual que en el CSV)
    S_max = math.log(N)
    feats["entropy_norm"]      = feats["dirac_entropy_final"] / S_max
    feats["entropy_abs_delta"] = abs(feats["dirac_entropy_delta"])
    feats["chirality_abs"]     = abs(feats["dirac_chirality_final"])
    feats["energy_abs_mean"]   = abs(feats["dirac_energy_mean"])
    feats["energy_std_sq"]     = feats["dirac_energy_std"] ** 2
    feats["cosine_sq"]         = feats["cosine_pa"] ** 2
    feats["len_log"]           = math.log1p(feats["len_ratio"])
    feats["len_inv"]           = 1.0 / (1.0 + feats["len_ratio"])

    return feats


def features_to_vector(feats: Dict[str, float]) -> np.ndarray:
    keys = [
        "cosine_pa",
        "len_ratio",
        "dirac_entropy_final",
        "dirac_entropy_delta",
        "dirac_chirality_final",
        "dirac_energy_mean",
        "dirac_energy_std",
        "entropy_norm",
        "entropy_abs_delta",
        "chirality_abs",
        "energy_abs_mean",
        "energy_std_sq",
        "cosine_sq",
        "len_log",
        "len_inv",
    ]
    return np.array([feats[k] for k in keys], dtype=float)


def compute_scores_srff_crff_ephi(prompt: str, answer: str):
    feats = compute_rrf_features(prompt, answer)
    x = features_to_vector(feats).reshape(1, -1)

    proba = meta_logit.predict_proba(x)[0]
    p_good = float(proba[1])

    # Definimos SRRF/CRRF/E_phi a partir de p_good y entropía
    SRRF = p_good
    CRRF = p_good * feats["cosine_pa"]

    S_max = math.log(N)
    norm_entropy = float(feats["dirac_entropy_final"] / S_max)
    E_phi = 0.5 * (SRRF + norm_entropy)

    scores = {
        "SRRF": SRRF,
        "CRRF": CRRF,
        "E_phi": E_phi,
        "p_good": p_good,
    }
    return scores, feats

# ============================
# FastAPI app
# ============================

class EvaluateRequest(BaseModel):
    prompt: str
    answer: str
    model_label: Optional[str] = None


class EvaluateResponse(BaseModel):
    scores: Dict[str, float]
    features: Dict[str, float]
    sim_summary: Dict[str, Any]


# Para poder reutilizar EvaluateRequest en /quality_remote
class QualityRemoteRequest(EvaluateRequest):
    """Mismo schema que EvaluateRequest, usado para el alias /quality_remote."""
    pass


app = FastAPI(
    title="Savant RRF Φ12.0 API",
    description="Dirac-Resonant conceptual quality layer for LLM-generated text.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Savant RRF Φ12.0 API running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    try:
        scores, feats = compute_scores_srff_crff_ephi(req.prompt, req.answer)

        # resumen de una simulación adicional (fresca) solo para info
        H = build_dirac_hamiltonian(
            m=0.25, v=1.0, sigma=0.618,
            alpha_log=0.10, q=1.0,
            flux_vector=(0.0, 0.0, 0.0),
            gauge_scale=0.0,
        )
        rng = np.random.default_rng(abs(hash(req.prompt + req.answer + "sim")) % (2 ** 32))
        vec = rng.normal(0, 1, (2 * N,)) + 1j * rng.normal(0, 1, (2 * N,))
        vec /= np.sqrt(np.vdot(vec, vec))
        psi0 = vec
        sim = evolve_dirac_shell(psi0, H, dt=0.05, steps=60, record_every=20)

        sim_summary = {
            "entropy_initial": float(sim["entropy"][0]),
            "entropy_final": float(sim["entropy"][-1]),
            "chirality_initial": float(sim["chirality"][0]),
            "chirality_final": float(sim["chirality"][-1]),
            "energy_mean": float(np.mean(sim["energy"])),
            "energy_std": float(np.std(sim["energy"])),
            "N_sites": int(N),
        }

        return EvaluateResponse(
            scores=scores,
            features=feats,
            sim_summary=sim_summary,
        )
    except Exception as e:
        print(f"❌ [Runtime] Error en /evaluate: {e}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# === SAVANT QUALITY_REMOTE PATCH (alias local de /evaluate) ===
@app.post("/quality_remote", response_model=EvaluateResponse)
def quality_remote(req: QualityRemoteRequest):
    """
    Alias de /evaluate para exponer la calidad RRF como /quality_remote.
    Entrada:
        {
          "prompt": "...",
          "answer": "...",
          "model_label": "..."   # opcional
        }
    Salida:
        El mismo JSON que /evaluate:
        {
          "scores": {...},
          "features": {...},
          "sim_summary": {...}
        }
    """
    # Aquí simplemente reutilizamos la misma lógica de evaluate
    return evaluate(req)

