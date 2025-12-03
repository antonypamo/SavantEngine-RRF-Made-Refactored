## 🚀 Run Locally

```bash
# 1) Clone the repo
git clone https://github.com/antonypamo/SavantEngine-RRF-Made.git
cd SavantEngine-RRF-Made

# 2) (Optional but recommended) Create and activate a virtual environment
python -m venv .venv

# On Linux / macOS
source .venv/bin/activate

# On Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Download and cache the base RRF model
python - << 'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("antonypamo/RRFSAVANTMADE")
EOF

# 5) Run an example or your own script
# (Adjust this to your entrypoint, e.g.)
# python savant_engine.py

## What is SavantEngine RRF?

SavantEngine RRF is a resonant quality and reranking engine for scientific and technical language.
It combines several RRF-based components:

- **Embeddings**: RRFSAVANTMADE / SAVANT-RRF-MADE-SymbioticModel for domain-specific semantic vectors.
- **Resonant Language Core**: ProSavantEngine Φ-series models to measure and enhance conceptual resonance.
- **Reasoning Layer (optional)**: Icosahedral GNN reasoner for role-based spectral coherence analysis.
- **Meta-Decision**: RRFSavantMetaLogit to gate and control heavy pipelines.

### Quickstart (local API)

```bash
git clone https://github.com/antonypamo/SavantEngine-RRF-Made-Refactored.git
cd SavantEngine-RRF-Made-Refactored
python -m pip install --upgrade pip
pip install -e .
uvicorn savant_engine.api.main:app --reload --port 8002

```
