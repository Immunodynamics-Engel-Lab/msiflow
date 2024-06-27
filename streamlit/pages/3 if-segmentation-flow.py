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

WORKFLOW = "if_segmentation_flow"
CONFIGWORKFLOW = f"{WORKFLOW}-config"

st.header("IF Segmentation Workflow")

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
            get_nested_value(config, "data"),
            key="data",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "data",
            ),
        )

    with st.expander("segmentation", expanded=True):
        config["if_segmentation"]["threshold_algorithm"] = st.selectbox(
            "threshold algorithm",
            ["otsu", "yen", "isodata", "mean", "minimum", "triangle"],
            get_options_id(
                get_nested_value(config, "if_segmentation.threshold_algorithm"),
                ["otsu", "yen", "isodata", "mean", "minimum", "triangle"],
            ),
            key="if_segmentation.threshold_algorithm",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_segmentation.threshold_algorithm",
            ),
        )
        config["if_segmentation"]["gauss_sigma"] = st.number_input(
            "Gauss sigma",
            value=get_nested_value(config, "if_segmentation.gauss_sigma"),
            key="if_segmentation.gauss_sigma",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_segmentation.gauss_sigma",
            ),
        )
        config["if_segmentation"]["min_size"] = st.number_input(
            "min object size",
            value=get_nested_value(config, "if_segmentation.min_size"),
            key="if_segmentation.min_size",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_segmentation.min_size",
            ),
        )
        config["if_segmentation"]["img_channels_to_segment"] = st.text_input(
            "image channels to segment",
            value=get_nested_value(config, "if_segmentation.img_channels_to_segment"),
            key="if_segmentation.img_channels_to_segment",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_segmentation.img_channels_to_segment",
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
