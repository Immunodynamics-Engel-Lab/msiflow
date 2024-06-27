import streamlit as st
from components.utils import (
    get_nested_value,
    load_yaml,
    run_workflow,
    save_yaml,
    update_value,
)

st.logo("imgs/msiFlow_logo.svg", icon_image="imgs/msiFlow_logo.svg")

WORKFLOW = "msi_if_registration_flow"
CONFIGWORKFLOW = f"{WORKFLOW}-config"

st.header("Registration Workflow")

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

    with st.expander("IF preprocessing", expanded=True):
        config["if_preprocessing"]["radius"] = st.number_input(
            "radius",
            value=get_nested_value(config, "if_preprocessing.radius"),
            key="if_preprocessing.radius",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_preprocessing.radius",
            ),
        )
        config["if_preprocessing"]["sigma"] = st.number_input(
            "sigma",
            value=get_nested_value(config, "if_preprocessing.sigma"),
            key="if_preprocessing.sigma",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_preprocessing.sigma",
            ),
        )
        config["if_preprocessing"]["lower_perc"] = st.number_input(
            "lower percentile",
            value=get_nested_value(config, "if_preprocessing.lower_perc"),
            key="if_preprocessing.lower_perc",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_preprocessing.lower_perc",
            ),
        )
        config["if_preprocessing"]["upper_perc"] = st.number_input(
            "upper percentile",
            value=get_nested_value(config, "if_preprocessing.upper_perc"),
            key="if_preprocessing.upper_perc",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "if_preprocessing.upper_perc",
            ),
        )

    with st.expander("registration", expanded=True):
        config["registration"]["af_chan"] = st.number_input(
            "autofluorescence image channel",
            value=get_nested_value(config, "registration.af_chan"),
            key="registration.af_chan",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "registration.af_chan",
            ),
        )
        config["registration"]["mask_val_chan"] = st.number_input(
            "validation mask channel",
            value=get_nested_value(config, "registration.mask_val_chan"),
            key="registration.mask_val_chan",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "registration.mask_val_chan",
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
