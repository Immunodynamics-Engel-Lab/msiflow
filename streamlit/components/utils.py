import yaml
import streamlit as st
import subprocess

import time
import re


def get_nested_value(c_dict: dict, key_string: str) -> any:
    """
    Retrieve a nested value from a dictionary using a dot-separated key string.

    Parameters
    ----------
    c_dict : dict
        The dictionary to search.
    key_string : str
        The dot-separated key string representing the path to the nested value.

    Returns
    -------
    any
        The nested value if found.

    Raises
    ------
    st.error
        If the nested value is not found or if an intermediate key does not point to a dictionary.
    """
    for subkey in key_string.split("."):
        c_dict = c_dict.get(subkey)
        if c_dict is None:
            st.error("Input not found")
            st.stop()
        if not isinstance(c_dict, dict):
            return c_dict


def set_nested_value(configworkflow: str, key_path: str, value) -> None:
    """
    Set a nested value in a dictionary within the Streamlit session state.

    Parameters
    ----------
    configworkflow : str
        The configuration workflow key.
    key_path : str
        The dot-separated key path representing where to set the value.
    value : any
        The value to set.

    Returns
    -------
    None
    """
    keys = (configworkflow + "." + key_path).split(".")
    current_dict = st.session_state
    # Traverse the dictionary until the second last key
    for key in (configworkflow + "." + key_path).split(".")[:-1]:
        current_dict = current_dict[key]
    # Set the value at the final key
    current_dict[keys[-1]] = value


def get_options_id(value: any, options: list) -> int:
    """
    Get the index of a value in a list of options from the Streamlit input.

    Parameters
    ----------
    value : any
        The value to find in the options list.
    options : list
        The list of options.

    Returns
    -------
    int
        The index of the value in the options list.
    """
    for idx, option in enumerate(options):
        if value == option:
            return idx


def update_value(configworkflow: str, key: str) -> None:
    """
    Update a nested value in the Streamlit session state using a key.

    Parameters
    ----------
    configworkflow : str
        The configuration workflow key.
    key : str
        The key for the value to update.

    Returns
    -------
    None
    """
    set_nested_value(configworkflow, key, st.session_state[key])


def load_yaml(config: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config : str
        The YAML configuration string.

    Returns
    -------
    dict
        The loaded configuration dictionary.

    Raises
    ------
    st.error
        If there is an error parsing the YAML or if the configuration is empty.
    """
    try:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            assert config is not None
            return config
    except yaml.YAMLError as e:
        st.error(f"Error parsing config YAML: {e}")
        st.stop()
    except AssertionError:
        st.error("Configuration File is empty")
        st.stop()


def save_yaml(config: dict, workflow: str) -> None:
    """
    Save a YAML configuration file.

    Parameters
    ----------
    config : dict
        The YAML configuration dictionary.
    workflow : str
        The path to the config directory.

    Raises
    ------
    st.error
        If there is an error writing the YAML file.
    """
    try:
        with open("data" + "/config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with open(workflow + "/data" + "/config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except yaml.YAMLError as e:
        st.error(f"Error parsing config YAML: {e}")
        st.stop()


def run_workflow(workflow: str) -> None:
    """
    Execute a Snakemake workflow and display its progress.

    Parameters
    ----------
    workflow : str
        The path to the workflow directory containing the Snakefile.

    Returns
    -------
    None
    """

    def get_progress(log: str) -> str | None:
        """
        Extract the progress percentage from the Snakemake log.

        Parameters
        ----------
        log : str
            The Snakemake log output.

        Returns
        -------
        str | None
            The progress percentage string extracted from the log, or None if not found.
        """
        pattern = re.compile(r"([A-Za-z0-9]+( [A-Za-z0-9]+)+)\s\(([^)]*)\)\sdone")
        matches = list(pattern.finditer(log))
        if matches:
            match = matches[-1]
            return match.group(0)

    # Start Snakemake pipeline
    process = subprocess.Popen(
        [
            "snakemake",
            "--snakefile",
            f"{workflow}/Snakefile",
            "--cores",
            "all",
            "--nolock",
            "--rerun-incomplete"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Display pipeline output live
    progress_text = "Workflow running: 0%"
    progress_bar = st.progress(0, progress_text)
    #with st.status("Executing workflow: 0%") as status:
        #mdoutput = st.empty()
        #output = ""
    while process.poll() is None:
        new_line = process.stderr.readline()
        if new_line:
            progress = get_progress(new_line)
            if progress:
                progress = int(progress.split("(")[-1].split(")")[0][:-1])
                #status.update(
                #    label=f"Executing workflow: {progress}",
                #)
                progress_text = f"Workflow running: {progress}%"
                progress_bar.progress(progress, text=progress_text)
                #new_line + "\n"
            #output += new_line
            #container = mdoutput.container(height=500, border=False)
            #container.markdown(output)
            #time.sleep(0.02)
        else:
            time.sleep(0.2)
    progress_bar.progress(100, text="Workflow completed!")
    # Read the final output
    _, _ = process.communicate()
    #if remaining_output:
        #output += remaining_output
        #container = mdoutput.container(height=500, border=False)
        #container.markdown(output)
    #status.update(label="Workflow complete!", state="complete")
