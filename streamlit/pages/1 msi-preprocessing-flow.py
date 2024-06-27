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

WORKFLOW = "msi_preprocessing_flow"
CONFIGWORKFLOW = f"{WORKFLOW}-config"

st.header("MSI Preprocessing Workflow")

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

    with st.expander("general", expanded=True):
        config["general"]["matrix_removal"] = st.checkbox(
            "matrix removal",
            value=get_nested_value(config, "general.matrix_removal"),
            key="general.matrix_removal",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "general.matrix_removal",
            ),
        )
        config["general"]["peak_filtering"] = st.checkbox(
            "peak filtering",
            value=get_nested_value(config, "general.peak_filtering"),
            key="general.peak_filtering",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "general.peak_filtering",
            ),
        )
        config["general"]["norm"] = st.checkbox(
            "normalisation",
            value=get_nested_value(config, "general.norm"),
            key="general.norm",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "general.norm",
            ),
        )
        config["general"]["outlier_removal"] = st.checkbox(
            "outlier removal",
            value=get_nested_value(config, "general.outlier_removal"),
            key="general.outlier_removal",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "general.outlier_removal",
            ),
        )
        config["general"]["deisotoping"] = st.checkbox(
            "deisotoping",
            value=get_nested_value(config, "general.deisotoping"),
            key="general.deisotoping",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "general.deisotoping",
            ),
        )

    with st.expander("peak picking", expanded=True):
        config["peak_picking"]["snr"] = st.number_input(
            "signal-to-noise-ratio",
            value=get_nested_value(config, "peak_picking.snr"),
            key="peak_picking.snr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "peak_picking.snr",
            ),
        )
        config["peak_picking"]["smooth"] = st.number_input(
            "smoothing",
            value=get_nested_value(config, "peak_picking.smooth"),
            key="peak_picking.smooth",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "peak_picking.smooth",
            ),
        )
        config["peak_picking"]["window"] = st.number_input(
            "window",
            value=get_nested_value(config, "peak_picking.window"),
            key="peak_picking.window",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "peak_picking.window",
            ),
        )
        config["peak_picking"]["order"] = st.number_input(
            "order",
            value=get_nested_value(config, "peak_picking.order"),
            key="peak_picking.order",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "peak_picking.order",
            ),
        )

    with st.expander("alignment", expanded=True):
        config["alignment"]["num_pixel_percentage"] = st.number_input(
            "number pixel percentage",
            value=get_nested_value(config, "alignment.num_pixel_percentage"),
            key="alignment.num_pixel_percentage",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "alignment.num_pixel_percentage",
            ),
        )
        config["alignment"]["mz_resolution"] = st.number_input(
            "mz resolution",
            value=get_nested_value(config, "alignment.mz_resolution"),
            key="alignment.mz_resolution",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "alignment.mz_resolution",
            ),
        )
        config["alignment"]["pixel_percentage"] = st.number_input(
            "pixel percentage",
            value=get_nested_value(config, "alignment.pixel_percentage"),
            key="alignment.pixel_percentage",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "alignment.pixel_percentage",
            ),
        )
        config["alignment"]["max_shift"] = st.number_input(
            "max. shift",
            value=get_nested_value(config, "alignment.max_shift"),
            key="alignment.max_shift",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "alignment.max_shift",
            ),
        )

    with st.expander("matrix removal", expanded=True):
        config["matrix_removal"]["clustering"] = st.checkbox(
            "clustering",
            value=get_nested_value(config, "matrix_removal.clustering"),
            key="matrix_removal.clustering",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.clustering",
            ),
        )
        config["matrix_removal"]["dim_reduction"] = st.selectbox(
            "dimensionality reduction",
            ["pca", "t-sne", "umap"],
            get_options_id(
                get_nested_value(config, "matrix_removal.dim_reduction"),
                ["pca", "t-sne", "umap"],
            ),
            key="matrix_removal.dim_reduction",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.dim_reduction",
            ),
        )
        config["matrix_removal"]["n_components"] = st.number_input(
            "number of components",
            value=get_nested_value(config, "matrix_removal.n_components"),
            key="matrix_removal.n_components",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.n_components",
            ),
        )
        config["matrix_removal"]["cluster_algorithm"] = st.selectbox(
            "cluster algorithm",
            [
                "gaussian_mixture",
                "hdbscan",
                "hierarchical",
                "k-means",
                "spatial-k-means",
            ],
            get_options_id(
                get_nested_value(config, "matrix_removal.cluster_algorithm"),
                [
                    "gaussian_mixture",
                    "hdbscan",
                    "hierarchical",
                    "k-means",
                    "spatial-k-means",
                ],
            ),
            key="matrix_removal.cluster_algorithm",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.cluster_algorithm",
            ),
        )
        tabs = st.tabs(["UMAP params", "HDBSCAN params", "k-means params"])
        with tabs[0]:
            config["matrix_removal"]["umap_params"]["metric"] = st.selectbox(
                "metric",
                ["chebyshev", "cosine", "correlation", "euclidean"],
                get_options_id(
                    get_nested_value(config, "matrix_removal.umap_params.metric"),
                    ["chebyshev", "cosine", "correlation", "euclidean"],
                ),
                key="matrix_removal.umap_params.metric",
                on_change=update_value,
                args=(
                    CONFIGWORKFLOW,
                    "matrix_removal.umap_params.metric",
                ),
            )
            config["matrix_removal"]["umap_params"]["n_neighbors"] = st.number_input(
                "number of neighbors",
                value=get_nested_value(
                    config, "matrix_removal.umap_params.n_neighbors"
                ),
                key="matrix_removal.umap_params.n_neighbors",
                on_change=update_value,
                args=(
                    CONFIGWORKFLOW,
                    "matrix_removal.umap_params.n_neighbors",
                ),
            )
            config["matrix_removal"]["umap_params"]["min_dist"] = st.number_input(
                "min. distance",
                value=get_nested_value(config, "matrix_removal.umap_params.min_dist"),
                key="matrix_removal.umap_params.min_dist",
                on_change=update_value,
                args=(
                    CONFIGWORKFLOW,
                    "matrix_removal.umap_params.min_dist",
                ),
            )
        with tabs[1]:
            config["matrix_removal"]["hdbscan_params"]["min_samples"] = st.number_input(
                "min. samples",
                value=get_nested_value(
                    config, "matrix_removal.hdbscan_params.min_samples"
                ),
                key="matrix_removal.hdbscan_params.min_samples",
                on_change=update_value,
                args=(
                    CONFIGWORKFLOW,
                    "matrix_removal.hdbscan_params.min_samples",
                ),
            )
            config["matrix_removal"]["hdbscan_params"]["min_cluster_size"] = (
                st.number_input(
                    "min. cluster size",
                    value=get_nested_value(
                        config, "matrix_removal.hdbscan_params.min_cluster_size"
                    ),
                    key="matrix_removal.hdbscan_params.min_cluster_size",
                    on_change=update_value,
                    args=(
                        CONFIGWORKFLOW,
                        "matrix_removal.hdbscan_params.min_cluster_size",
                    ),
                )
            )
        with tabs[2]:
            config["matrix_removal"]["kmeans_params"]["n_clusters"] = st.number_input(
                "number of clusters",
                value=get_nested_value(
                    config, "matrix_removal.kmeans_params.n_clusters"
                ),
                key="matrix_removal.kmeans_params.n_clusters",
                on_change=update_value,
                args=(
                    CONFIGWORKFLOW,
                    "matrix_removal.kmeans_params.n_clusters",
                ),
            )
        config["matrix_removal"]["pixel_removal"] = st.checkbox(
            "pixel removal",
            value=get_nested_value(config, "matrix_removal.pixel_removal"),
            key="matrix_removal.pixel_removal",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.pixel_removal",
            ),
        )
        config["matrix_removal"]["matrix_subtraction"] = st.checkbox(
            "matrix subtraction",
            value=get_nested_value(config, "matrix_removal.matrix_subtraction"),
            key="matrix_removal.matrix_subtraction",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.matrix_subtraction",
            ),
        )
        config["matrix_removal"]["matrix_peak_removal"] = st.checkbox(
            "matrix peak removal",
            value=get_nested_value(config, "matrix_removal.matrix_peak_removal"),
            key="matrix_removal.matrix_peak_removal",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.matrix_peak_removal",
            ),
        )

        config["matrix_removal"]["num_matrix_peaks"] = st.number_input(
            "number of matrix peaks",
            value=get_nested_value(config, "matrix_removal.num_matrix_peaks"),
            key="matrix_removal.num_matrix_peaks",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.num_matrix_peaks",
            ),
        )

        config["matrix_removal"]["matrix_corr_thr"] = st.number_input(
            "matrix correlation threshold",
            value=get_nested_value(config, "matrix_removal.matrix_corr_thr"),
            key="matrix_removal.matrix_corr_thr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.matrix_corr_thr",
            ),
        )

        config["matrix_removal"]["pixel_perc_thr"] = st.number_input(
            "pixel percentage threshold",
            value=get_nested_value(config, "matrix_removal.pixel_perc_thr"),
            key="matrix_removal.pixel_perc_thr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.pixel_perc_thr",
            ),
        )

        config["matrix_removal"]["matrix_postproc"] = st.checkbox(
            "matrix postprocessing",
            value=get_nested_value(config, "matrix_removal.matrix_postproc"),
            key="matrix_removal.matrix_postproc",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.matrix_postproc",
            ),
        )

        config["matrix_removal"]["matrix_mzs"] = st.text_input(
            "matrix m/z list",
            value=get_nested_value(config, "matrix_removal.matrix_mzs"),
            key="matrix_removal.matrix_mzs",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.matrix_mzs",
            ),
        )

        config["matrix_removal"]["thr_method"] = st.text_input(
            "thresholding algorithm",
            value=get_nested_value(config, "matrix_removal.thr_method"),
            key="matrix_removal.thr_method",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "matrix_removal.thr_method",
            ),
        )

    with st.expander("peak filtering", expanded=True):
        config["peak_filtering"]["sum"] = st.selectbox(
            "summarize",
            ["max", "mean", "min"],
            get_options_id(
                get_nested_value(config, "peak_filtering.sum"),
                ["max", "mean", "min"],
            ),
            key="peak_filtering.sum",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "peak_filtering.sum",
            ),
        )

        config["peak_filtering"]["thr"] = st.number_input(
            "threshold",
            value=get_nested_value(config, "peak_filtering.thr"),
            key="peak_filtering.thr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "peak_filtering.thr",
            ),
        )

    with st.expander("intra-normalisation", expanded=True):
        config["intranorm"]["method"] = st.selectbox(
            "method",
            ["mean", "mfc", "sum"],
            get_options_id(
                get_nested_value(config, "intranorm.method"),
                ["mean", "mfc", "sum"],
            ),
            key="intranorm.method",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "intranorm.method",
            ),
        )

    with st.expander("inter-normalisation", expanded=True):
        config["internorm"]["method"] = st.selectbox(
            "method",
            ["mean", "mfc", "sum"],
            get_options_id(
                get_nested_value(config, "internorm.method"),
                ["mean", "mfc", "sum"],
            ),
            key="internorm.method",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "internorm.method",
            ),
        )

    with st.expander("outlier detection", expanded=True):
        config["outlier_detection"]["umap_params"]["n_neighbors"] = st.number_input(
            "UMAP number of neighbors",
            value=get_nested_value(config, "outlier_detection.umap_params.n_neighbors"),
            key="outlier_detection.umap_params.n_neighbors",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "outlier_detection.umap_params.n_neighbors",
            ),
        )
        config["outlier_detection"]["hdbscan_params"]["min_samples"] = st.number_input(
            "HDBSCAN min. samples",
            value=get_nested_value(
                config, "outlier_detection.hdbscan_params.min_samples"
            ),
            key="outlier_detection.hdbscan_params.min_samples",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "outlier_detection.hdbscan_params.min_samples",
            ),
        )
        config["outlier_detection"]["hdbscan_params"]["min_cluster_size"] = (
            st.number_input(
                "HDBSCAN min. cluster size",
                value=get_nested_value(
                    config, "outlier_detection.hdbscan_params.min_cluster_size"
                ),
                key="outlier_detection.hdbscan_params.min_cluster_size",
                on_change=update_value,
                args=(
                    CONFIGWORKFLOW,
                    "outlier_detection.hdbscan_params.min_cluster_size",
                ),
            )
        )
        config["outlier_detection"]["cluster_thr"] = st.number_input(
            "cluster threshold",
            value=get_nested_value(config, "outlier_detection.cluster_thr"),
            key="outlier_detection.cluster_thr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "outlier_detection.cluster_thr",
            ),
        )
        config["outlier_detection"]["sample_thr"] = st.number_input(
            "sample threshold",
            value=get_nested_value(config, "outlier_detection.sample_thr"),
            key="outlier_detection.sample_thr",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "outlier_detection.sample_thr",
            ),
        )
        config["outlier_detection"]["remove_ssc"] = st.checkbox(
            "remove sample-specific-clusters",
            value=get_nested_value(config, "outlier_detection.remove_ssc"),
            key="outlier_detection.remove_ssc",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "outlier_detection.remove_ssc",
            ),
        )

    with st.expander("deisotoping", expanded=True):
        config["deisotoping"]["tolerance"] = st.number_input(
            "tolerance",
            value=get_nested_value(config, "deisotoping.tolerance"),
            key="deisotoping.tolerance",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "deisotoping.tolerance",
            ),
        )
        config["deisotoping"]["min_isotopes"] = st.number_input(
            "min. isotopes",
            value=get_nested_value(config, "deisotoping.min_isotopes"),
            key="deisotoping.min_isotopes",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "deisotoping.min_isotopes",
            ),
        )
        config["deisotoping"]["max_isotopes"] = st.number_input(
            "max. isotopes",
            value=get_nested_value(config, "deisotoping.max_isotopes"),
            key="deisotoping.max_isotopes",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "deisotoping.max_isotopes",
            ),
        )
        config["deisotoping"]["openMS"] = st.checkbox(
            "openMS",
            value=get_nested_value(config, "deisotoping.openMS"),
            key="deisotoping.openMS",
            on_change=update_value,
            args=(
                CONFIGWORKFLOW,
                "deisotoping.openMS",
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
