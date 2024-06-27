import streamlit as st
from components.utils import (
    get_nested_value,
    get_options_id,
    load_yaml,
    run_workflow,
    save_yaml,
    update_value,
)

st.logo("imgs/msiFlow_logo.svg", icon_image="imgs/msiFlow_logo.svg")

WORKFLOW = "molecular_heterogeneity_flow"
CONFIGWORKFLOW = f"{WORKFLOW}-config"

st.header("Molecular Heterogeneity Workflow")

if CONFIGWORKFLOW not in st.session_state:
    config = load_yaml(f"{WORKFLOW}/data/config.yaml")
    st.session_state[CONFIGWORKFLOW] = config
else:
    config = st.session_state[CONFIGWORKFLOW]


@st.experimental_fragment
def get_inputs():
    with st.expander("data", expanded=True):
        config["data"] = st.text_input(
            "input path",
            value=get_nested_value(config, "data"),
            key="data",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "data",
            ),
        )
        config["dot_size"] = st.number_input(
            "dot size",
            value=get_nested_value(config, "dot_size"),
            key="dot_size",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "dot_size",
            ),
        )
    with st.expander("UMAP", expanded=True):
        config["umap"]["metric"] = st.selectbox(
            "metric",
            ["chebyshev", "cosine", "correlation", "euclidean"],
            get_options_id(
                get_nested_value(config, "umap.metric"),
                ["chebyshev", "cosine", "correlation", "euclidean"],
            ),
            key="umap.metric",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "umap.metric",
            ),
        )
        config["umap"]["n_neighbors"] = st.number_input(
            "number of neighbors",
            value=get_nested_value(config, "umap.n_neighbors"),
            key="umap.n_neighbors",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "umap.n_neighbors",
            ),
        )
        config["umap"]["min_dist"] = st.number_input(
            "min. distance",
            value=get_nested_value(config, "umap.min_dist"),
            key="umap.min_dist",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "umap.min_dist",
            ),
        )
        config["umap"]["use_model"] = st.checkbox(
            "use model",
            value=get_nested_value(config, "umap.use_model"),
            key="umap.use_model",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "umap.use_model",
            ),
        )
    with st.expander("HDBSCAN", expanded=True):
        config["hdbscan"]["min_samples"] = st.number_input(
            "min. samples",
            value=get_nested_value(config, "hdbscan.min_samples"),
            key="hdbscan.min_samples",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "hdbscan.min_samples",
            ),
        )
        config["hdbscan"]["min_cluster_size"] = st.number_input(
            "min. cluster size",
            value=get_nested_value(config, "hdbscan.min_cluster_size"),
            key="hdbscan.min_cluster_size",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "hdbscan.min_cluster_size",
            ),
        )


get_inputs()

cols = st.columns([4.5, 1])
with cols[0]:
    reset_config = st.button("reset config")
with cols[1]:
    run = st.button("run workflow")

if reset_config:
    st.session_state[CONFIGWORKFLOW] = load_yaml(f"streamlit/backups/{WORKFLOW}.yaml")
    st.rerun()

if run:
    save_yaml(config, WORKFLOW)
    run_workflow(WORKFLOW)
