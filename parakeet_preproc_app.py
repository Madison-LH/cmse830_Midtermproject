import io
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Parakeet Call Preprocessing • Explainer & Dashboard",
    layout="wide"
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
.small { font-size:0.9rem; line-height:1.35; }
.code-caption { font-size:0.85rem; color:#666; margin-top:-0.4rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Data locations
# ------------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/Madison-LH/cmse830_Midtermproject/refs/heads/main/results/"

# Expected filenames (old + new cluster plots)
EXPECTED = {
    "feat": "acoustic_features_with_contact_posteriors.csv",
    "flags": "contact_posterior_flags.csv",
    "ici": "contact_intercall_intervals.csv",
    "sr": "wav_samplerate_summary.csv",

    # old plots
    "img_dur_peak": "duration_vs_peakfreq.png",
    "img_pca": "cluster_pca_pc1_pc2.png",
    "img_thresh": "cluster_threshold_counts.png",

    # new cluster / embedding plots
    "img_coassoc": "cluster_coassociation_heatmap.png",
    "img_clustergram": "clustergram_gmm_vs_kmeans.png",
    "img_importance": "cluster_feature_importance.png",
    "img_tsne_clusters": "embedding_tsne_clusters.png",
    "img_tsne_calltype": "embedding_tsne_calltype.png",
    "img_umap_clusters": "embedding_umap_clusters.png",
    "img_umap_calltype": "embedding_umap_calltype.png",
    "img_pca_new": "cluster_pca_pc1_pc2_sixcluster.png",

    # interactive embeddings table
    "embed": "embeddings_for_streamlit.csv",
}


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


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🐦 Parakeet Call Preprocessing — Explainer & Results Dashboard")

st.markdown(f"""
This app **explains** the parakeet call preprocessing pipeline and visualizes outputs
fetched from:

> [{BASE_URL}]({BASE_URL})
""")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tabs = st.tabs([
    "Purpose & Design",
    "Dataset & Scaling",
    "IDA & EDA",
    "Results (old cluster plots)",
    "Dimensionality Reduction & New Cluster Plots",
    "Imputation",
    "Repro Tips"
])

# ------------------------------------------------------------------
# Tab 1: Purpose & Design
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Purpose & Design")

    st.markdown("""
**Goal:**  
Build a reproducible pipeline that:

- Reads **Raven selection tables** and matches them to `.wav` files  
- Joins call-level and recording-level metadata  
- Filters low-quality / noisy calls early  
- Extracts standardized **acoustic features**  
- Uses **unsupervised clustering** to identify contact-call groups
""")

    st.subheader("Pipeline overview")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F5F7FB"];
      A[label="1) Read Raven tables + map to .wav"];
      B[label="2) Join metadata & filter quality/noise"];
      C[label="3) Build EST (warbleR)"];
      D[label="4) Standardization checks"];
      E[label="5) Extract acoustic features"];
      F[label="6) Plot / unsupervised clustering"];
      G[label="7) Export contact-call set"];
      A->B->C->D->E->F->G;
    }
    """)

    st.subheader("Key step example")
    st.code(r"""
# Filter by call quality & background noise
call_df <- call_df |>
  dplyr::filter(call_quality %in% c("high","medium"),
                background_noise == FALSE)
""", language="r")
    st.markdown(
        "<div class='small'>Filtering early prevents noisy/poor calls from distorting PCA and clustering downstream.</div>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------
# Tab 2: Dataset & Scaling
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Dataset & Scaling")

    st.markdown("""
### Each call is its own mini-dataset

For every call we extract a **vector of acoustic features**, including:

- Dominant frequency statistics (mean, median, quartiles)  
- Peak frequency and bandwidth  
- Duration and temporal quartiles  
- Spectral and temporal entropy  

So each row in the feature table is like a tiny dataset summarizing one call's
time–frequency structure.
""")

    try:
        feat_df_ds = load_csv_url(_join(BASE_URL, EXPECTED["feat"]))
        n_calls = len(feat_df_ds)
        n_numeric = feat_df_ds.select_dtypes("number").shape[1]
        n_clusters = feat_df_ds["cluster_unsup"].nunique(dropna=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Number of call observations", f"{n_calls}")
        c2.metric("Numeric features per call", f"{n_numeric}")
        c3.metric("GMM clusters", f"{n_clusters}")

        st.markdown(
            f"""
Earlier runs used a much smaller number of calls.  
In this **scaled-up run** there are **over 27 different calls** and in total
**{n_calls} call-level observations**, each with **{n_numeric} acoustic features**.
The larger dataset lets the Gaussian mixture model stably resolve **six clusters**
instead of just a coarse split.
"""
        )

    except Exception as e:
        st.error(f"Failed to load {EXPECTED['feat']} for dataset summary: {e}")

    st.markdown("""
### What changed vs earlier versions?

- More recordings and calls were included after quality/noise filtering.  
- All numeric features were **z-scored** before PCA/clustering.  
- The GMM search range for number of clusters was widened (e.g. 2–8),
  which now supports a six-cluster solution.
""")

# ------------------------------------------------------------------
# Tab 3: IDA & EDA
# ------------------------------------------------------------------
with tabs[2]:
    st.header("Initial & Exploratory Data Analysis")

    # Samplerate consistency
    st.subheader("Samplerate consistency across WAVs")
    try:
        df_sr = load_csv_url(_join(BASE_URL, EXPECTED["sr"]))
        st.dataframe(df_sr)
        st.markdown(
            "<div class='small'>Uniform sample rates ensure that spectrogram parameters mean the same thing across files.</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['sr']}: {e}")

    # Duration vs peak frequency
    st.subheader("Duration vs peak frequency")
    try:
        img = load_image_url(_join(BASE_URL, EXPECTED["img_dur_peak"]))
        st.image(img, caption="Scatter: call duration (s) vs peak frequency (kHz)")
        st.markdown(
            "<div class='small'>Distinct clouds suggest different call families even before clustering.</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['img_dur_peak']}: {e}")

    # Inter-call intervals
    st.subheader("Inter-call intervals (ICIs) for contact-like calls")
    try:
        ici_df = load_csv_url(_join(BASE_URL, EXPECTED["ici"]))
        if "ici_sec" in ici_df.columns:
            fig, ax = plt.subplots()
            ax.hist(ici_df["ici_sec"].dropna(), bins=30)
            ax.set_xlabel("ICI (s)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.markdown(
                "<div class='small'>ICI distributions show calling rhythm; multiple modes can reflect different behavioral contexts or individuals.</div>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['ici']}: {e}")

# ------------------------------------------------------------------
# Tab 4: Results (old cluster plots)
# ------------------------------------------------------------------
with tabs[3]:
    st.header("Results & Clustering Outcomes (original plots)")

    # Posterior flags
    try:
        flags_df = load_csv_url(_join(BASE_URL, EXPECTED["flags"]))
        st.dataframe(flags_df.head(100))
        if "contact_thresh_pass" in flags_df.columns:
            pct = 100 * flags_df["contact_thresh_pass"].fillna(False).astype(bool).mean()
            st.markdown(f"**{pct:.1f}% of calls passed the ≥0.80 contact posterior threshold.**")
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['flags']}: {e}")

    st.subheader("Original cluster plots")

    cols = st.columns(2)
    with cols[0]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_pca"]))
            st.image(img, caption="OLD: PCA clusters (PC1 vs PC2)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_pca']}: {e}")
    with cols[1]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_thresh"]))
            st.image(img, caption="OLD: cluster composition (≥80% vs <80%)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_thresh']}: {e}")

    st.markdown(
        "<div class='small'>These are the original cluster plots from the earlier version of the pipeline: a PCA scatter and a bar plot showing how many calls per cluster pass the contact threshold.</div>",
        unsafe_allow_html=True
    )

    st.markdown("**Updated PCA cluster plot for the scaled-up dataset:**")

    try:
        img = load_image_url(_join(BASE_URL, EXPECTED["img_pca_new"]))
        st.image(
            img,
            caption=(
                "NEW: PCA of 28 acoustic features with 6 GMM clusters "
                "(larger dataset, fixed PCs, G ∈ 2–8)"
            )
        )
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['img_pca_new']}: {e}")


        st.markdown("""
### How this new PCA cluster plot differs from the old one

The **old PCA cluster plot** was generated from a much smaller dataset and a narrower
clustering search range. With fewer calls available, the Gaussian mixture model
(GMM) could only reliably recover **2–3 coarse groups**, and the PCA structure
appeared simpler and more compact.

The **new PCA cluster plot** is based on a **scaled-up dataset** with more than
27 call types and hundreds of call instances. Because the acoustic feature space
is now better sampled, the GMM (searched over **G ∈ 2–8**) converges consistently
on **6 stable clusters**. Several important differences emerge:

- The PCA structure becomes **richer and more stratified**, revealing sub-clusters
  that were previously invisible.
- Call-type shapes (contact-like vs non-contact) align more cleanly with
  **specific GMM clusters**.
- Ellipses highlight **distinct covariance patterns**, especially for the
  non-contact calls.
- The larger dataset creates **clearer separation along PC1 and PC2**, reflecting
  real acoustic variability rather than noise.

In short:  
the **old plot shows broad acoustic categories**, while the **new plot shows
fine-grained, biologically interpretable subtypes** that only emerge when the
dataset is large enough to support full GMM structure.
""")
        
    st.markdown("---")
    st.subheader("Feature table with posteriors")
    try:
        feat_df = load_csv_url(_join(BASE_URL, EXPECTED["feat"]))
        st.dataframe(feat_df.head(200))
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['feat']}: {e}")

# ------------------------------------------------------------------
# Tab 5: Dimensionality Reduction & NEW cluster plots
# ------------------------------------------------------------------
with tabs[4]:
    st.header("Dimensionality Reduction & New Cluster Diagnostics")

    st.markdown("""
We compress correlated acoustic features into a smaller number of dimensions and
then examine cluster structure in several ways.

This run uses:

- **PCA** for linear dimensionality reduction  
- **t-SNE** and **UMAP** for nonlinear embeddings  
- **Repeated GMM runs** for stability (co-association)  
- A **random forest** to rank which features best separate clusters
""")

    st.subheader("PCA in the R pipeline (fixed PC count)")
    st.code(r"""
# R sketch
Z <- scale(as.matrix(features_numeric))
pca <- prcomp(Z, center = FALSE, scale. = FALSE)

PCS <- 3
Zp <- pca$x[, 1:min(PCS, ncol(pca$x)), drop = FALSE]
""", language="r")
    st.markdown(
        "<div class='code-caption'>We fix the number of principal components (e.g. 3) to keep the GMM search well-conditioned and comparable across runs.</div>",
        unsafe_allow_html=True
    )

    # ------------------ Interactive embeddings ------------------
    st.subheader("Interactive embedding explorer")

    try:
        embed_df = load_csv_url(_join(BASE_URL, EXPECTED["embed"]))
        df = embed_df.copy()

        if "cluster" in df.columns:
            df["cluster"] = df["cluster"].astype("Int64")
            df["cluster_label"] = df["cluster"].map(
                lambda c: f"C{int(c)}" if pd.notnull(c) else None
            )

        c1, c2, c3 = st.columns(3)
        emb_choice = c1.selectbox(
            "Embedding space",
            ["PCA (PC1 vs PC2)", "t-SNE (1 vs 2)", "UMAP (1 vs 2)"]
        )
        color_choice = c2.selectbox(
            "Color by",
            ["Cluster", "Call type", "P(contact)", "Contact ≥ 0.8?"]
        )
        size_choice = c3.selectbox(
            "Point size",
            ["Small", "Medium", "Large"],
            index=1
        )
        size_map = {"Small": 30, "Medium": 60, "Large": 100}
        point_size = size_map[size_choice]

        if emb_choice.startswith("PCA"):
            x_col, y_col = "PC1", "PC2"
        elif emb_choice.startswith("t-SNE"):
            x_col, y_col = "TSNE1", "TSNE2"
        else:
            x_col, y_col = "UMAP1", "UMAP2"

        missing = [c for c in (x_col, y_col) if c not in df.columns]
        if missing:
            st.warning(f"Embedding columns {missing} are missing in {EXPECTED['embed']}.")
        else:
            if color_choice == "Cluster":
                color_enc = alt.Color("cluster_label:N", title="Cluster")
            elif color_choice == "Call type":
                color_enc = alt.Color("call_type:N", title="Call type")
            elif color_choice == "P(contact)":
                color_enc = alt.Color(
                    "p_contact:Q",
                    title="P(contact cluster)",
                    scale=alt.Scale(scheme="viridis")
                )
            else:
                df["contact_flag"] = df["contact_thresh_pass"].map(
                    lambda v: "≥ 0.8" if bool(v) else "< 0.8"
                )
                color_enc = alt.Color("contact_flag:N", title="Contact posterior")

            tooltip = [
                "call_uid",
                "sound.files",
                "call_type",
                "cluster_label",
                alt.Tooltip("p_contact:Q", format=".2f"),
            ]

            chart = (
                alt.Chart(df)
                .mark_circle(size=point_size, opacity=0.8)
                .encode(
                    x=alt.X(f"{x_col}:Q", title=x_col),
                    y=alt.Y(f"{y_col}:Q", title=y_col),
                    color=color_enc,
                    tooltip=tooltip,
                )
                .properties(height=550)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("""
**How to interpret this:**

- Each point is a single call plotted in PCA, t-SNE, or UMAP space.  
- You can color by unsupervised cluster, metadata call_type, or contact posterior.  
- The interactive view lets you inspect individual calls via tooltips.
""")

    except Exception as e:
        st.warning(f"Interactive embeddings not available yet: {e}")

    st.markdown("---")
    st.subheader("Original vs new cluster plots")

    st.markdown("**Original cluster plots (repeated here for comparison):**")

    cols_old = st.columns(2)
    with cols_old[0]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_pca"]))
            st.image(img, caption="OLD: PCA clusters (PC1 vs PC2)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_pca']}: {e}")
    with cols_old[1]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_thresh"]))
            st.image(img, caption="OLD: cluster composition (≥80% vs <80%)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_thresh']}: {e}")

    st.markdown(
        "<div class='small'>These are the same plots as in the Results tab — kept here so you can compare them directly to the new diagnostics below.</div>",
        unsafe_allow_html=True
    )

    st.markdown("**New cluster diagnostics added in this run:**")

    # t-SNE plots
    cols_tsne = st.columns(2)
    with cols_tsne[0]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_tsne_clusters"]))
            st.image(img, caption="NEW: t-SNE embedding colored by cluster")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_tsne_clusters']}: {e}")
    with cols_tsne[1]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_tsne_calltype"]))
            st.image(img, caption="NEW: t-SNE embedding colored by call_type")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_tsne_calltype']}: {e}")

    # UMAP plots
    cols_umap = st.columns(2)
    with cols_umap[0]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_umap_clusters"]))
            st.image(img, caption="NEW: UMAP embedding colored by cluster")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_umap_clusters']}: {e}")
    with cols_umap[1]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_umap_calltype"]))
            st.image(img, caption="NEW: UMAP embedding colored by call_type")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_umap_calltype']}: {e}")

    # Stability + clustergram
    cols_stab = st.columns(2)
    with cols_stab[0]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_coassoc"]))
            st.image(img, caption="NEW: cluster co-association heatmap")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_coassoc']}: {e}")
    with cols_stab[1]:
        try:
            img = load_image_url(_join(BASE_URL, EXPECTED["img_clustergram"]))
            st.image(img, caption="NEW: clustergram (GMM vs k-means)")
        except Exception as e:
            st.error(f"Failed to load {EXPECTED['img_clustergram']}: {e}")

    st.subheader("Feature importance for cluster separation")
    try:
        img = load_image_url(_join(BASE_URL, EXPECTED["img_importance"]))
        st.image(img, caption="NEW: random-forest feature importance")
    except Exception as e:
        st.error(f"Failed to load {EXPECTED['img_importance']}: {e}")

    st.markdown("""
**Summary of the new plots:**

- t-SNE and UMAP show that clusters form coherent islands in nonlinear embeddings.  
- The co-association heatmap checks that clusters are stable across repeated GMM runs.  
- The GMM vs k-means clustergram checks agreement between two different clustering
  algorithms.  
- The random-forest importance plot shows which acoustic features actually drive
  the separation between clusters.
""")

# ------------------------------------------------------------------
# Tab 6: Imputation
# ------------------------------------------------------------------
with tabs[5]:
    st.header("Imputation Techniques Used (and Options)")

    st.markdown("""
Missing values can occur when acoustic extraction fails for very short/noisy calls.
Imputing before PCA and clustering avoids dropping calls and stabilises the components.

**Current R approach in this project:** simple, robust **median imputation** per feature.
""")

    st.code(r"""
# R: median imputation then scaling
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
    st.markdown(
        "<div class='code-caption'>Median is simple and outlier-robust, but it ignores correlations among features.</div>",
        unsafe_allow_html=True
    )

    st.subheader("Potential future upgrades")
    st.markdown("""
- **KNN imputation**: fills a call's missing values from acoustically similar calls.  
- **Iterative/multivariate imputation**: models each feature using the others.  
- **PCA-informed imputation** (`missMDA`): iterates PCA and reconstruction.  
- **MICE**: multiple imputations with predictive mean matching.
""")

# ------------------------------------------------------------------
# Tab 7: Repro Tips
# ------------------------------------------------------------------
with tabs[6]:
    st.header("Reproducibility Tips")
    st.markdown("""
- Pin R and package versions with **renv**.  
- Save intermediate EST objects and feature-screening tables.  
- Log samplerate and clipping summaries.  
- Keep all outputs in a versioned `preproc_outputs/` directory.  
- Export embeddings and diagnostics so this app can always reflect the latest run.
""")

st.markdown("---")
st.caption(
    "Run locally: `pip install streamlit pandas matplotlib requests altair` → "
    "`streamlit run parakeet_preproc_app.py`"
)
