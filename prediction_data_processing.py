# prediction_data_processing.py

import pandas as pd


def build_season_dataset(
    results_path="podaci/results.csv",
    races_path="podaci/races.csv",
    standings_path="podaci/driver_standings.csv",
    start_year=2000,
    end_year=2024,
    early_race_limit=12
):
    results = pd.read_csv(results_path)
    races = pd.read_csv(races_path)
    driver_standings = pd.read_csv(standings_path)

    results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["points"] = pd.to_numeric(results["points"], errors="coerce")

    # Merge results with races
    df = results.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="inner"
    )

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.sort_values(["year", "driverId", "round"])

    # Early season filter (set to first 12 races)
    df["early_phase"] = df["round"] <= df.groupby("year")["round"].transform(
        lambda x: min(early_race_limit, x.max())
    )

    first_part = df[df["early_phase"]]

    # Feature engineering: Aggregate driver stats from the early season (e.g., podiums, points, average grid position).
    season_features = (
        first_part
        .groupby(["year", "driverId"])
        .agg(
            podiums_first12=("positionOrder", lambda x: (x <= 3).sum()),
            total_points_first12=("points", "sum"),
            avg_start_position=("grid", "mean"),
        )
        .reset_index()
    )

    standings = driver_standings.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="left"
    )

    standings = standings[
        (standings["year"] >= start_year) &
        (standings["year"] <= end_year)
    ]

    last_round_per_year = (
        standings.groupby("year")["round"]
        .max()
        .reset_index()
    )

    final_standings = standings.merge(
        last_round_per_year,
        on=["year", "round"],
        how="inner"
    )

    # Mark champions
    champions = final_standings[final_standings["position"] == 1][
        ["year", "driverId"]
    ]
    champions["is_champion"] = 1


    # Merge driver features with champion label
    season_dataset = season_features.merge(
        champions,
        on=["year", "driverId"],
        how="left"
    )

    season_dataset["is_champion"] = (
        season_dataset["is_champion"]
        .fillna(0)
        .astype(int)
    )

    season_dataset = season_dataset.sort_values(
        ["year", "driverId"]
    ).reset_index(drop=True)

    return season_dataset