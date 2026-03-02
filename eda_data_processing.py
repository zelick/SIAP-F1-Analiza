# eda_data_processing.py

import pandas as pd


def build_race_level_dataset(
    results_path="podaci/results.csv",
    races_path="podaci/races.csv",
    circuits_path="podaci/circuits.csv",
    start_year=2000,
    end_year=2024,
):
    results = pd.read_csv(results_path)
    races = pd.read_csv(races_path)
    circuits = pd.read_csv(circuits_path)

    results["position"] = pd.to_numeric(results["position"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")

    races_filtered = races[
        (races["year"] >= start_year) &
        (races["year"] <= end_year)
    ]

    # Only drivers that finished race
    results_finished = results[results["position"].notna()]

    # Merge results with races + add year and circuitId
    df = results_finished.merge(
        races_filtered[["raceId", "year", "circuitId"]],
        on="raceId",
        how="inner"
    )

    df = df.merge(
        circuits[["circuitId", "name"]],
        on="circuitId",
        how="left"
    )

    results_clean = df[[
        "raceId",
        "year",
        "circuitId",
        "name",
        "driverId",
        "grid",
        "position"
    ]]

    results_clean = results_clean.rename(columns={
        "name": "circuit_name",
        "grid": "start_position",
        "position": "finish_position"
    })

    # Remove invalid start positions (starts from pitlane, disqualified, etc.)
    results_clean = results_clean[
        (results_clean["start_position"].notna()) &
        (results_clean["start_position"] >= 1)
    ]

    results_clean = results_clean.reset_index(drop=True)

    return results_clean