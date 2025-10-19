# cmse830_Midtermproject
# 🐦
This dashboard accompanies the **Parakeet Call Preprocessing Pipeline**, which performs:

1. Reading and standardizing Raven selection tables (`*.txt`)
2. Joining recording-level and call-level metadata
3. Filtering low-quality or noisy calls
4. Extracting acoustic features via `warbleR::spectro_analysis()`
5. Reducing feature space using PCA
6. Clustering calls using Gaussian Mixture Models (GMMs)
7. Identifying “contact call” clusters based on posterior probability thresholds (≥ 0.80)

The Streamlit app provides an interactive overview of each step, along with reproducibility guidance and summaries of my pipeline's results.

R source code for the pipeline cannot be provided due to this pertaining to a lab project as well. 
