import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import os

# Dossier où sont tes notebooks
NOTEBOOK_DIR = "notebooks"

def run_notebook(notebook_name):
    path = os.path.join(NOTEBOOK_DIR, notebook_name)
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
    html_exporter = HTMLExporter()
    html_data, _ = html_exporter.from_notebook_node(nb)
    return html_data

st.title("Tumor Detection Notebooks")

# Boutons pour chaque notebook
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Exploration Dataset"):
        with st.spinner("Running dataset..."):
            run_notebook("01_exploration_dataset.ipynb")
        st.image("outputs/dataset.png", caption="Dataset Visualization")

with col2:
    if st.button("Data Analysis"):
        with st.spinner("Running analysis..."):
            run_notebook("02_data_analysis.ipynb")
        st.image("outputs/analysis.png", caption="Analysis Visualization")


with col3:
    if st.button("Grad-CAM Demo"):
        with st.spinner("Running gradcam..."):
            run_notebook("03_gradcam_demo.ipynb")
        st.image("outputs/gradcam_result.png", caption="Grad-CAM Result")