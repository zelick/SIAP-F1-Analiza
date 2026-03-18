# prediction_data_processing.py

import pandas as pd


def build_season_dataset(
    results_path="podaci/results.csv",
    races_path="podaci/races.csv",
    standings_path="podaci/driver_standings.csv",
    qualifying_path="podaci/qualifying.csv",
    constructor_standings_path="podaci/constructor_standings.csv",
    start_year=2000,
    end_year=2024,
    early_race_limit=12
):
    results = pd.read_csv(results_path)
    races = pd.read_csv(races_path)
    driver_standings = pd.read_csv(standings_path)
    qualifying = pd.read_csv(qualifying_path)
    constructor_standings = pd.read_csv(constructor_standings_path)

    results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["points"] = pd.to_numeric(results["points"], errors="coerce")
    results["rank"] = pd.to_numeric(results["rank"], errors="coerce")
    qualifying["position"] = pd.to_numeric(qualifying["position"], errors="coerce")
    constructor_standings["position"] = pd.to_numeric(constructor_standings["position"], errors="coerce")

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

    # Feature engineering: Aggregate driver stats from the early season
    season_features = (
        first_part
        .groupby(["year", "driverId"])
        .agg(
            podiums_first12=("positionOrder", lambda x: (x <= 3).sum()),
            wins_first12=("positionOrder", lambda x: (x == 1).sum()),
            total_points_first12=("points", "sum"),
            avg_start_position=("grid", "mean"),
            avg_finish_position_first12=("positionOrder", "mean"),
            # std of points per race – low std = consistent performance
            points_std_first12=("points", "std"),
            fastest_laps_first12=("rank", lambda x: (x == 1).sum()),
            # DNF rate: statusId == 1 means "Finished", anything else is a DNF/retirement
            dnf_rate_first12=("statusId", lambda x: (pd.to_numeric(x, errors="coerce") != 1).mean()),
        )
        .reset_index()
    )

    # Fill NaN std (drivers with only 1 race) with 0
    season_features["points_std_first12"] = season_features["points_std_first12"].fillna(0)

    # --- Driver standings position after race 12 (or last available round <= 12) ---
    standings = driver_standings.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="left"
    )

    standings = standings[
        (standings["year"] >= start_year) &
        (standings["year"] <= end_year)
    ]

    # Find the last round within the early phase for each year
    early_round_per_year = (
        standings[standings["round"] <= early_race_limit]
        .groupby("year")["round"]
        .max()
        .reset_index()
        .rename(columns={"round": "early_round"})
    )

    standings = standings.merge(early_round_per_year, on="year", how="left")
    standings_after12 = standings[standings["round"] == standings["early_round"]][
        ["year", "driverId", "position"]
    ].rename(columns={"position": "standings_position_after12"})

    season_features = season_features.merge(standings_after12, on=["year", "driverId"], how="left")

    # --- Average qualifying position from first 12 races ---
    quali_df = qualifying.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="inner"
    )
    quali_df = quali_df[
        (quali_df["year"] >= start_year) &
        (quali_df["year"] <= end_year) &
        (quali_df["round"] <= early_race_limit)
    ]
    avg_quali = (
        quali_df.groupby(["year", "driverId"])["position"]
        .mean()
        .reset_index()
        .rename(columns={"position": "avg_qualifying_position_first12"})
    )
    season_features = season_features.merge(avg_quali, on=["year", "driverId"], how="left")

    # --- Constructor (team) standings position after race 12 ---
    # Map constructorId for each driver in early phase (use most frequent team that season)
    driver_constructor = (
        first_part.groupby(["year", "driverId"])["constructorId"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    con_standings = constructor_standings.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="left"
    )
    con_standings = con_standings[
        (con_standings["year"] >= start_year) &
        (con_standings["year"] <= end_year)
    ]
    con_early_round = (
        con_standings[con_standings["round"] <= early_race_limit]
        .groupby("year")["round"]
        .max()
        .reset_index()
        .rename(columns={"round": "early_round"})
    )
    con_standings = con_standings.merge(con_early_round, on="year", how="left")
    con_standings_after12 = con_standings[con_standings["round"] == con_standings["early_round"]][
        ["year", "constructorId", "position"]
    ].rename(columns={"position": "constructor_position_after12"})

    # Join: driver → constructorId → constructor standing
    driver_constructor = driver_constructor.merge(
        con_standings_after12, on=["year", "constructorId"], how="left"
    )
    season_features = season_features.merge(
        driver_constructor[["year", "driverId", "constructor_position_after12"]],
        on=["year", "driverId"],
        how="left"
    )

    # --- Determine champions (same as before) ---
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

    champions = final_standings[final_standings["position"] == 1][
        ["year", "driverId"]
    ].copy()
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