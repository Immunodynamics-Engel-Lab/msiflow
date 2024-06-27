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

WORKFLOW = "molecular_signatures_flow"
CONFIGWORKFLOW = f"{WORKFLOW}-config"

st.header("Molecular Signatures Workflow")

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

    with st.expander("classification", expanded=True):
        config["classification"]["model"] = st.selectbox(
            "model",
            ["XGBoost", "LGBoost", "AdaBoost", "CatBoost", "GBoost", "RandomForest"],
            get_options_id(
                get_nested_value(config, "classification.model"),
                [
                    "XGBoost",
                    "LGBoost",
                    "AdaBoost",
                    "CatBoost",
                    "GBoost",
                    "RandomForest",
                ],
            ),
            key="classification.model",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.model",
            ),
        )
        config["classification"]["img_channels"] = st.text_input(
            "image channels",
            value=get_nested_value(config, "classification.img_channels"),
            key="classification.img_channels",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.img_channels",
            ),
        )
        config["classification"]["multiclass"] = st.checkbox(
            "multiclass",
            value=get_nested_value(config, "classification.multiclass"),
            key="classification.multiclass",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.multiclass",
            ),
        )
        config["classification"]["class_balancing_method"] = st.selectbox(
            "class balancing method",
            ["standard", "smote", "undersample", "oversample", "weights"],
            get_options_id(
                get_nested_value(config, "classification.class_balancing_method"),
                ["standard", "smote", "undersample", "oversample", "weights"],
            ),
            key="classification.class_balancing_method",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.class_balancing_method",
            ),
        )
        config["classification"]["num_top_feat"] = st.number_input(
            "number of top features",
            value=get_nested_value(config, "classification.num_top_feat"),
            key="classification.num_top_feat",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.num_top_feat",
            ),
        )
        config["classification"]["save_ion_imgs"] = st.checkbox(
            "save ion images",
            value=get_nested_value(config, "classification.save_ion_imgs"),
            key="classification.save_ion_imgs",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.save_ion_imgs",
            ),
        )
        config["classification"]["save_umap_imgs"] = st.checkbox(
            "save UMAP images",
            value=get_nested_value(config, "classification.save_umap_imgs"),
            key="classification.save_umap_imgs",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.save_umap_imgs",
            ),
        )
        config["classification"]["n_folds"] = st.number_input(
            "number of folds",
            value=get_nested_value(config, "classification.n_folds"),
            key="classification.n_folds",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.n_folds",
            ),
        )
        config["classification"]["annotate"] = st.checkbox(
            "annotate",
            value=get_nested_value(config, "classification.annotate"),
            key="classification.annotate",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "classification.annotate",
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
