from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from pkg.utils import booltoint


# -----------------------------
# FUNCTIONS
# -----------------------------
def calc_ppm_error(exp_mz, calc_mz):
    return (exp_mz - calc_mz) / calc_mz * 1e6


def within_tolerance(exp_mz, calc_mz, ppm_tol):
    return abs(calc_ppm_error(exp_mz, calc_mz)) <= ppm_tol


def is_even_chain(lipid_name):
    match = re.search(r'\b(\d+):\d+\b', lipid_name)
    if match:
        total_carbons = int(match.group(1))
        return total_carbons % 2 == 0
    else:
        return True  # fallback: include if cannot parse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs mass matching to annotate lipids')
    parser.add_argument('sdf_file', type=str, help='sdf file')
    parser.add_argument('msi_file', type=str, help='csv file with m/z values in first column')
    parser.add_argument('output_dir', type=str, default='', help='output dir')
    parser.add_argument('-unit', type=str, default='ppm', help='ppm or Da for mass matching tolerance')
    parser.add_argument('-tol', type=float, default=5, help='ppm or Da tolerance for mass matching')
    parser.add_argument("-even_chains_only", type=lambda x: booltoint(x), default=0, help="if != 0 only even-numbered fatty acyl chains")
    parser.add_argument('-lipid_classes', type=str,
                        default='FA,Cer,SM,HexCer,Hex2Cer,TG,DG,ST,PC,PA,PS,PE,PG',
                        help='Comma-separated lipid class abbreviations to include, e.g., PC,TG,PE')
    parser.add_argument("-adducts", type=str, default="H,Na,K", help="Comma-separated adducts to include, e.g., H,Na,K")
    parser.add_argument("-plot", type=int, default=0, help="if != 0 plot")
    args = parser.parse_args()

    # -----------------------------
    # SETTINGS
    # -----------------------------
    SDF_FILE = args.sdf_file
    MSI_FILE = args.msi_file
    ERROR_MODE = args.unit
    TOLERANCE = args.tol
    OUT_FILE = os.path.join(args.output_dir, 'annotation.csv')
    EVEN_CHAINS_ONLY = args.even_chains_only
    LIPID_CLASSES_TO_INCLUDE = [x.strip() for x in args.lipid_classes.split(",")]
    ADDUCT_MASSES_DICT = {
        "H": 1.007276,
        "Na": 22.989218,
        "K": 38.963158,
        "H-H2O": 1.007276 - 18.010565  # Proton minus water
        # add more if needed
    }

    # Filter adducts by user input
    user_adducts = [x.strip() for x in args.adducts.split(",")]
    adducts = {ad: ADDUCT_MASSES_DICT[ad] for ad in user_adducts if ad in ADDUCT_MASSES_DICT}

    print("Filtering lipid classes:", LIPID_CLASSES_TO_INCLUDE)
    print("Using adducts:", adducts)

    # Example experimental data (replace with your own)
    # experimental_mz = [760.585, 788.615, 810.600]
    df = pd.read_csv(MSI_FILE)
    # Assume first column contains the m/z values
    experimental_mz = df.iloc[:, 0].tolist()  # converts first column to a list of floats
    # experimental_mz = [369.351578,577.519037,760.585082,768.553782,551.503387,703.574852,734.569432,786.600732]
    # experimental_mz = [369.37363, 577.5137, 760.57996, 768.5446, 551.49493, 703.5677, 734.5612, 786.59375]
    # experimental_mz = [369.35135, 577.5175, 760.5843, 768.55273, 551.5027, 703.57275, 734.5669, 786.6002]
    print("Total experimental m/z values:", len(experimental_mz))

    # -----------------------------
    # STEP 1: BUILD DATABASE
    # -----------------------------
    print("Loading SDF and building database...")

    db_entries = []

    supplier = Chem.SDMolSupplier(SDF_FILE)

    failed_count = 0
    for i, mol in enumerate(supplier):
        if mol is None:
            failed_count += 1
            continue

        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        exact_mass = Descriptors.ExactMolWt(mol)
        #name = mol.GetProp("NAME") if mol.HasProp("NAME") else "Unknown"
        name = mol.GetProp("ABBREVIATION") if mol.HasProp("ABBREVIATION") else "Unknown"
        #lipid_class = mol.GetProp("MAIN_CLASS") if mol.HasProp("MAIN_CLASS") else "Unknown"
        lipid_class = mol.GetProp("ABBREVIATION").split()[0] if mol.HasProp("ABBREVIATION") else "Unknown"

        for adduct_name, adduct_mass in adducts.items():
            mz = exact_mass + adduct_mass

            db_entries.append({
                "LipidName": name,
                "LipidClass": lipid_class,
                "Formula": formula,
                "ExactMass": round(exact_mass, 6),
                "Adduct": adduct_name,
                "Calc_mz": round(mz, 6)
            })

    print(f"Skipped {failed_count} invalid molecules in SDF")

    db = pd.DataFrame(db_entries)

    # Filter based on lipid classes
    db = db[db["LipidClass"].isin(LIPID_CLASSES_TO_INCLUDE)].reset_index(drop=True)

    # Keep only one row per unique Formula + ExactMass + Adduct
    db = db.drop_duplicates(subset=["LipidName", "LipidClass", "Formula", "ExactMass", "Adduct"]).reset_index(drop=True)

    # Optional: speed optimization (sorting)
    db = db.sort_values("Calc_mz").reset_index(drop=True)

    print(f"Database size (with adducts): {len(db)} entries")

    # -----------------------------
    # STEP 2: MATCHING
    # -----------------------------
    print("Matching experimental m/z values...")

    results = []

    for exp_m in experimental_mz:
        # Calculate both errors
        ppm_errors = (exp_m - db["Calc_mz"]) / db["Calc_mz"] * 1e6
        da_errors = exp_m - db["Calc_mz"]

        # Choose error type and tolerance
        if ERROR_MODE == "ppm":
            mask = ppm_errors.abs() <= TOLERANCE
            selected_error = ppm_errors
        elif ERROR_MODE == "Da":
            mask = da_errors.abs() <= TOLERANCE
            selected_error = da_errors
        else:
            raise ValueError("ERROR_MODE must be 'ppm' or 'Da'")

        hits = db[mask].copy()

        if EVEN_CHAINS_ONLY != 0 and not hits.empty:
            # Parse total carbon count from LipidName (e.g., 'PC 34:1')
            hits = hits[hits["LipidName"].apply(is_even_chain)]

        if not hits.empty:
            hits["Exp_mz"] = exp_m
            hits["mass_error"] = selected_error[mask]
            results.append(hits)

    # Combine all matches
    if results:
        results_df = pd.concat(results)

        # Optional: sort by ppm error
        results_df["abs_mass_error"] = results_df["mass_error"].abs()
        #results_df = results_df.sort_values("abs_mass_error")
        results_df = results_df.sort_values("Exp_mz")
        print(results_df)

        # Keep only relevant columns
        results_df = results_df[["Exp_mz", "LipidName", "LipidClass", "Formula", "Adduct", "Calc_mz", "ExactMass", "abs_mass_error"]].reset_index(drop=True)
    else:
        results_df = pd.DataFrame(columns=["Exp_mz", "LipidName", "LipidClass", "Formula", "Adduct", "Calc_mz", "ExactMass", "abs_mass_error"])

    # -----------------------------
    # OUTPUT
    # -----------------------------
    # print(results_df.head(20))

    save_df = results_df[[
        "Exp_mz", "LipidName", "LipidClass", "Formula", "Adduct", "Calc_mz", "ExactMass", "abs_mass_error"
    ]].rename(columns={
        "Exp_mz": "Input Mass",
        "LipidName": "Name",
        "LipidClass": "Class",
        "Formula": "Formula",
        "Adduct": "Adduct Type",
        "Calc_mz": "Calculated Mass",
        "ExactMass": "Exact Mass",
        "abs_mass_error": "Mass Error"
    }).reset_index(drop=True)

    # Save results
    save_df.to_csv(OUT_FILE, index=False)

    # -----------------------------
    # QC
    # -----------------------------

    # Count number of candidates per experimental m/z
    candidate_counts = results_df.groupby("Exp_mz").size().reset_index(name="NumCandidates")

    print(candidate_counts)

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y="NumCandidates", data=candidate_counts)
    plt.ylabel("Number of candidate lipids per experimental m/z")
    plt.title("Candidate Lipid Matches per m/z")
    plt.savefig(os.path.join(args.output_dir, 'candidate_lipid_matches.svg'))
    if args.plot != 0:
        plt.show()

    # Histogram
    # Step 1: Count number of candidates per m/z
    num_candidates = results_df.groupby("Exp_mz").size()  # Series: index=Exp_mz, value=# candidates

    # Step 2: Count how often each number of candidates occurs
    hist_counts = num_candidates.value_counts().sort_index()  # index=#candidates, value=#m/z with that many candidates

    # Step 3: Plot as bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(hist_counts.index.astype(int), hist_counts.values, color='skyblue', edgecolor='black', width=0.8)
    #plt.ylim(0, 500)
    plt.xlabel("Number of candidate lipids per m/z")
    plt.ylabel("Number of m/z values")
    plt.title("Distribution of candidate lipid matches per m/z")
    plt.xticks(hist_counts.index.astype(int))
    plt.savefig(os.path.join(args.output_dir, 'distribution_candidate_lipid_matches.svg'))
    if args.plot != 0:
        plt.show()

    # -----------------------------
    # PIE CHART: lipid classes overall
    # -----------------------------
    class_counts = results_df["LipidClass"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10})
    plt.title("Lipid Class Distribution (All Annotations)")
    plt.savefig(os.path.join(args.output_dir, 'pie_lipid_classes_overall.svg'))
    if args.plot != 0:
        plt.show()

    # -----------------------------
    # PIE CHARTS: lipid classes per histogram bin
    # -----------------------------
    for num_candidates in hist_counts.index:
        mzs_in_bin = num_candidates.index if hasattr(num_candidates, 'index') else results_df.groupby("Exp_mz").size()
        hits_in_bin = results_df[results_df.groupby("Exp_mz")["Exp_mz"].transform('size') == num_candidates]
        if hits_in_bin.empty:
            continue

        class_counts_bin = hits_in_bin["LipidClass"].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(class_counts_bin.values, labels=class_counts_bin.index, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 10})
        plt.title(f"Lipid Class Distribution (m/z with {num_candidates} candidates)")
        plt.savefig(os.path.join(args.output_dir, f'pie_lipid_classes_{num_candidates}_candidates.svg'))
        if args.plot != 0:
            plt.show()