import io
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Parakeet Call Preprocessing • Explainer & Dashboard",
    layout="wide"
)

# Styling 
st.markdown("""
<style>
.small { font-size:0.9rem; line-height:1.35; }
.code-caption { font-size:0.85rem; color:#666; margin-top:-0.4rem; }
</style>
""", unsafe_allow_html=True)

# Hardcoded GitHub results base
BASE_URL = "https://raw.githubusercontent.com/Madison-LH/cmse830_Midtermproject/refs/heads/main/results/"

# Expected filenames
EXPECTED = {
    "feat": "acoustic_features_with_contact_posteriors.csv",
    "flags": "contact_posterior_flags.csv",
    "ici": "contact_intercall_intervals.csv",
    "sr": "wav_samplerate_summary.csv",
    "img_dur_peak": "duration_vs_peakfreq.png",
    "img_pca": "cluster_pca_pc1_pc2.png",
    "img_thresh": "cluster_threshold_counts.png",
}

# Helper functions
def _join(base: str, name: str) -> str:
    base = base.strip()
    return base + name if base.endswith("/") else base + "/" + name

@st.cache_data(show_spinner=False)
def load_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(show_spinner=False)
def load_image_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

# Header
st.title("🐦 Parakeet Call Preprocessing — Explainer & Results Dashboard")

st.markdown(f"""
This app **explains** the parakeet call preprocessing pipeline and visualizes outputs
fetched automatically from:

> [{BASE_URL}]({BASE_URL})
""")

# Overview
with st.expander("🎯 Research Aim & Pipeline Overview", expanded=True):
    st.subheader("Goal")
    st.markdown("""
Build a reproducible workflow to:
- Read **Raven selection tables** and match them to `.wav` files  
- Join **recording** and **call metadata**  
- Filter low-quality or noisy calls  
- Extract **acoustic features**  
- Use **unsupervised clustering** to identify contact-call groups
""")

    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F5F7FB"];
      A[label="1) Read Raven tables (*.txt) + map to .wav"];
      B[label="2) Join metadata + filter quality/noise"];
      C[label="3) Build EST (warbleR)"];
      D[label="4) Standardize metadata"];
      E[label="5) Extract acoustic features"];
      F[label="6) IDA/EDA plots"];
      G[label="7) PCA → GMM clustering"];
      H[label="8) Export contact-call set"];
      A->B->C->D->E->F->G->H;
    }
    """)

    st.code(r"""
# Example R outline
raven_df <- readr::read_delim("*.txt", delim="\t") |>
  dplyr::mutate(call_uid = paste0(sound.files, "__", selec))

sel_meta <- raven_df |>
  dplyr::left_join(call_df, by="call_uid") |>
  dplyr::filter(call_quality %in% c("high","medium"),
                background_noise == FALSE)

est <- warbleR::extended_selection_table(X = sel_meta, path = AUDIO_DIR)
meas <- warbleR::spectro_analysis(X = est, img = FALSE)
""", language="r")

# Tabs
tabs = st.tabs([
    "Purpose & Design",
    "IDA & EDA",
    "Results",
    "Dimensionality Reduction",
    "Imputation",
    "Repro Tips"
])

# Tab 1: Purpose & Design

with tabs[0]:
    st.header("Purpose & Design")
    st.markdown("""
**Why this pipeline?**
- Ensure *reproducibility*  
- Perform *standardized acoustic extraction*  
- Enable *data-driven discovery* of contact-call clusters
""")

    st.subheader("Key Steps")
    st.code(r"""
# Filter by call quality & background noise
call_df <- call_df |>
  dplyr::filter(call_quality %in% c("high","medium"),
                background_noise == FALSE)
""", language="r")
    st.markdown("<div class='small'>Filtering early prevents noisy or poor-quality calls from biasing later analyses.</div>", unsafe_allow_html=True)

# Tab 2: IDA & EDA
with tabs[1]:
    st.header("Initial & Exploratory Data Analysis")

    # --- Samplerate Summary ---
    st.subheader("Samplerate Consistency")
    try:
        df_sr = load_csv_url(_join(BASE_URL, EXPECTED["sr"]))
        st.dataframe(df_sr)
        st.markdown("<div class='small'>All recordings should share the same sample rate to avoid feature drift.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['sr']}: {e}")

    # --- Duration vs Peak Frequency ---
    st.subheader("Duration vs Peak Frequency")
    try:
        img = load_image_url(_join(BASE_URL, EXPECTED["img_dur_peak"]))
        st.image(img, caption="Duration (s) vs Peak frequency (kHz)")
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['img_dur_peak']}: {e}")

    # --- Inter-Call Intervals (ICIs) ---
    st.subheader("Inter-Call Intervals (ICIs)")
    try:
        ici_df = load_csv_url(_join(BASE_URL, EXPECTED["ici"]))
        if "ici_sec" in ici_df.columns:
            fig, ax = plt.subplots()
            ax.hist(ici_df["ici_sec"].dropna(), bins=30)
            ax.set_xlabel("ICI (s)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.markdown("<div class='small'>ICI histograms show rhythmic patterns of contact calling; multiple peaks can indicate context or individual differences.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['ici']}: {e}")

# Tab 3: Results
with tabs[2]:
    st.header("Results & Clustering Outcomes")

    # Flags / contact probabilities
    try:
        flags_df = load_csv_url(_join(BASE_URL, EXPECTED["flags"]))
        st.dataframe(flags_df.head(100))
        if "contact_thresh_pass" in flags_df.columns:
            pct = 100 * flags_df["contact_thresh_pass"].fillna(False).astype(bool).mean()
            st.markdown(f"**{pct:.1f}% of calls passed the ≥0.80 contact threshold.**")
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['flags']}: {e}")

    # Cluster plots
    cols = st.columns(2)
    with cols[0]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_pca"]))
            st.image(img, caption="PCA clusters (PC1 vs PC2)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_pca']}: {e}")
    with cols[1]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_thresh"]))
            st.image(img, caption="Cluster composition (≥80% vs <80%)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_thresh']}: {e}")

    st.markdown("---")
    st.subheader("Full Features with Posteriors")
    try:
        feat_df = load_csv_url(_join(BASE_URL, EXPECTED["feat"]))
        st.dataframe(feat_df.head(200))
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['feat']}: {e}")

# Tab 4: Dimensionality Reduction
with tabs[3]:
    st.header("Dimensionality Reduction Techniques Used")

    st.markdown("""
**Goal:** compress correlated acoustic features into a small number of informative axes that
stabilize downstream clustering.

**Primary method: PCA (Principal Component Analysis)**
- Works on the correlation structure to produce orthogonal components.
- We retain a small, fixed number of PCs (e.g., 2–3) to keep runs reproducible and avoid overfitting GMMs.
- PCA is linear, fast, and easy to interpret (loadings show which features drive each PC).
""")

    st.code(r"""
# R (in your pipeline)
# 1) Select numeric acoustic features and scale
Z <- scale(as.matrix(features_numeric))

# 2) PCA
pca <- prcomp(Z, center = FALSE, scale. = FALSE)

# 3) Keep a fixed number of PCs for stability (e.g., 3)
PCS <- 3
Zp <- pca$x[, 1:min(PCS, ncol(pca$x)), drop = FALSE]
""", language="r")
    st.markdown("<div class='code-caption'>We fix the number of PCs to reduce run-to-run variance and keep the GMM search well-conditioned.</div>", unsafe_allow_html=True)

    st.subheader("Alternative DR methods (when and why)")
    st.markdown("""
- **t-SNE** (nonlinear, good for visualization; not ideal as input to parametric clustering).
- **UMAP** (preserves local/global structure; useful for exploratory plots; also nonlinear).
- **PCA + Whitening** (optional): sometimes improves spherical clustering methods.

**Recommendation for this project:** keep **PCA for clustering** (inputs to GMM) and use **UMAP/t-SNE only for visualization** so cluster assignments remain stable and reproducible.
""")

# Tab 5: Imputation
with tabs[4]:
    st.header("Imputation Techniques Used (and Options)")

    st.markdown("""
**Why impute?** Acoustic extraction can produce missing values (short/noisy calls, measurement failures).
Imputing before PCA/clustering avoids dropping rows and stabilizes components.

**Current approach in your R script:** simple **median imputation** per feature.
""")

    st.code(r"""
# R (current approach summarized)
X <- as.data.frame(features_numeric)
for (j in seq_along(X)) {
  v <- X[[j]]
  med <- suppressWarnings(stats::median(v, na.rm = TRUE))
  if (!is.finite(med)) med <- 0
  v[!is.finite(v)] <- med
  X[[j]] <- v
}
Z <- scale(as.matrix(X))
""", language="r")
    st.markdown("<div class='code-caption'>Median is robust and simple, but ignores correlations among features.</div>", unsafe_allow_html=True)

    st.subheader("Stronger options (drop-in ideas)")
    st.markdown("""
- **KNN imputation**: fills a call’s missing values from acoustically similar calls.
- **Iterative (multivariate) imputation**: models each feature using the others (regression-style).
- **PCA-informed imputation** (`missMDA`): iterates PCA and reconstruction for coherent fills.
- **MICE**: multiple imputations with predictive mean matching (good with mixed distributions).
""")

    st.code(r"""
# R: PCA-based imputation (missMDA)
library(missMDA)
acoustic <- as.data.frame(features_numeric)
ncp_est <- estim_ncpPCA(acoustic)     # estimate number of components
imp <- imputePCA(acoustic, ncp = ncp_est$ncp)
X_imp <- imp$completeObs               # use this for PCA/GMM
""", language="r")
    st.markdown("<div class='code-caption'>Keeps imputations consistent with the PCA structure used for clustering.</div>", unsafe_allow_html=True)

    st.code(r"""
# R: MICE (predictive mean matching)
library(mice)
imp <- mice(acoustic, m = 1, method = "pmm", maxit = 5, seed = 123)
X_imp <- complete(imp)
""", language="r")
    st.markdown("<div class='code-caption'>Captures nonlinear relations via predictive mean matching; set m>1 for multiple imputations.</div>", unsafe_allow_html=True)

    st.code(r"""
# Python: KNN and Iterative imputation (if you port this step)
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np

X = features_numeric.values  # numpy array
knn = KNNImputer(n_neighbors=5)
X_knn = knn.fit_transform(X)

it = IterativeImputer(random_state=123)
X_iter = it.fit_transform(X)
""", language="python")
    st.markdown("<div class='code-caption'>Use KNN when neighbors are meaningful; Iterative when linear-ish relations hold.</div>", unsafe_allow_html=True)

    st.subheader("Practical guidance")
    st.markdown("""
- Start with **median** (baseline), then try **missMDA** or **KNN** and compare:
  - PCA variance explained, GMM BIC, and cluster separability plots.
- Keep **the same imputation strategy** across runs for reproducibility.
- Document the chosen method in your Methods section.
""")

# Tab 6: Repro Tips
with tabs[5]:
    st.header("Reproducibility Tips")
    st.markdown("""
- Pin R and package versions with **renv**  
- Save intermediate selection_table and feature-screening results  
- Log samplerate and clipping summaries  
- Keep outputs in a versioned `preproc_outputs/` directory
""")

st.markdown("---")
st.caption("Run locally:  `pip install streamlit pandas matplotlib requests`  →  `streamlit run parakeet_preproc_app.py`")
