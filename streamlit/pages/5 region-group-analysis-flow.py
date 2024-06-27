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

WORKFLOW = "region_group_analysis_flow"
CONFIGWORKFLOW = f"{WORKFLOW}-config"

st.header("Region Group Analysis Workflow")

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
    config["annotate"] = st.checkbox(
        "annotate",
        value=get_nested_value(config, "annotate"),
        key="annotate",
        on_change=update_value,
        args=(
            CONFIGWORKFLOW,
            "annotate",
        ),
    )

    with st.expander("summarize spectra", expanded=True):
        config["summarize_spectra"]["method"] = st.selectbox(
            "method",
            ["mean", "median"],
            get_options_id(
                get_nested_value(config, "summarize_spectra.method"), ["mean", "median"]
            ),
            key="summarize_spectra.method",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "summarize_spectra.method",
            ),
        )

    with st.expander("statistical analysis", expanded=True):
        config["statistical_analysis"]["fold_change_thr"] = st.number_input(
            "log2 fold change treshold",
            value=get_nested_value(config, "statistical_analysis.fold_change_thr"),
            key="statistical_analysis.fold_change_thr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "statistical_analysis.fold_change_thr",
            ),
        )

        config["statistical_analysis"]["infected_grp"] = st.text_input(
            "infected group",
            value=get_nested_value(config, "statistical_analysis.infected_grp"),
            key="statistical_analysis.infected_grp",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "statistical_analysis.infected_grp",
            ),
        )
        config["statistical_analysis"]["control_grp"] = st.text_input(
            "control group",
            value=get_nested_value(config, "statistical_analysis.control_grp"),
            key="statistical_analysis.control_grp",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "statistical_analysis.control_grp",
            ),
        )
        config["statistical_analysis"]["save_ion_imgs"] = st.checkbox(
            "save ion images",
            value=get_nested_value(config, "statistical_analysis.save_ion_imgs"),
            key="statistical_analysis.save_ion_imgs",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "statistical_analysis.save_ion_imgs",
            ),
        )

    with st.expander("heatmap", expanded=True):
        config["heatmap"]["row_norm"] = st.checkbox(
            "row normalisation",
            value=get_nested_value(config, "heatmap.row_norm"),
            key="heatmap.row_norm",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "heatmap.row_norm",
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
