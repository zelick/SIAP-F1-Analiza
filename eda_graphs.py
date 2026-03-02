import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# GRAFIKON 1
# Globalna korelacija startne i finalne pozicije
# =====================================================
def plot_start_vs_finish(results_clean):

    # kopija i ciscenje
    df = results_clean.dropna(
        subset=["start_position", "finish_position"]
    ).copy()

    # validne pozicije
    df = df[
        (df["start_position"] > 0) &
        (df["finish_position"] > 0)
    ]

    # scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["start_position"],
        df["finish_position"],
        alpha=0.15,
        s=12
    )

    # regresiona linija
    # trend linija - regresiona linija
    # Greska - vertikalna udaljenost tacke od linije
    # Najbolja linija minimizira zbir kvadrata tih gresaka
    # np.polyfit - vraca koeficijente pravca (m, b) za y = mx + b
    m, b = np.polyfit(
        df["start_position"],
        df["finish_position"],
        1
    )

    x = np.array([
        df["start_position"].min(),
        df["start_position"].max()
    ])

    plt.plot(x, m * x + b)

    plt.title("Globalna korelacija startne i finalne pozicije")
    plt.xlabel("Startna pozicija (start_position)")
    plt.ylabel("Konačna pozicija (finish_position)")

    plt.show()

    # statistika
    print(f"Broj uzoraka: {len(df)}")

    # Pearsonov koeficijent korelacije meri linearnu zavisnost izmedju dve promenljive
    # Vrednost blizu 1 ili -1 ukazuje na jaku pozitivnu ili negativnu korelaciju, dok vrednost blizu 0 ukazuje na slabiju korelaciju
    corr = df["start_position"].corr(df["finish_position"])
    print("Pearson korelacija:", corr)


# =====================================================
# GRAFIKON 2
# Dinamika trke po stazama
# Prosečna apsolutna promena pozicije
# =====================================================
def plot_race_dynamics_by_circuit(results_clean):

    df = results_clean.dropna(subset=[
        "circuit_name",
        "start_position",
        "finish_position"
    ]).copy()

    # apsolutna promena pozicije
    df["abs_change"] = (
        df["start_position"] - df["finish_position"]
    ).abs()

    # broj uzoraka po stazi
    n_per = df.groupby("circuit_name").size().rename("n")

    # prosecna promena
    avg_abs_change = df.groupby("circuit_name")["abs_change"].mean()

    circuit_stats = pd.concat(
        [n_per, avg_abs_change],
        axis=1
    ).reset_index()

    circuit_stats.columns = [
        "circuit_name",
        "n",
        "avg_abs_change"
    ]

    # filtriranje staza sa dovoljno podataka
    MIN_N = 200
    circuit_stats = circuit_stats[
        circuit_stats["n"] >= MIN_N
    ]

    # top staze po broju uzoraka
    TOP_K = 15
    top = (
        circuit_stats
        .sort_values("n", ascending=False)
        .head(TOP_K)
        .sort_values("avg_abs_change", ascending=False)
    )

    # grafikon
    plt.figure(figsize=(12, 6))
    plt.barh(
        top["circuit_name"],
        top["avg_abs_change"]
    )

    plt.xlabel("Prosečna apsolutna promena pozicije")
    plt.ylabel("Staza")
    plt.title("Grafikon 2: Dinamika trke po stazama (Top 15, n ≥ 200)")
    plt.tight_layout()
    plt.show()

    print("\nTop staze po dinamici trke:\n")
    display(top)


# =====================================================
# GRAFIKON 3
# Stabilnost startne pozicije po stazama
# Verovatnoća da vozač završi na istoj poziciji
# =====================================================
def plot_start_position_stability(results_clean):

    df = results_clean.dropna(subset=[
        "circuit_name",
        "start_position",
        "finish_position"
    ]).copy()

    # da li je pozicija ostala ista
    df["same_position"] = (
        df["start_position"] == df["finish_position"]
    )

    # verovatnoca zadrzavanja pozicije
    prob_same = (
        df.groupby("circuit_name")["same_position"]
        .mean()
        .reset_index()
    )

    # broj uzoraka po stazi
    counts = (
        df.groupby("circuit_name")
        .size()
        .reset_index(name="n")
    )

    prob_same = prob_same.merge(counts, on="circuit_name")

    # filtriranje
    MIN_N = 150
    prob_same = prob_same[prob_same["n"] >= MIN_N]

    # grafikon
    plt.figure(figsize=(10, 6))
    plt.barh(
        prob_same["circuit_name"],
        prob_same["same_position"]
    )

    plt.xlabel("Verovatnoća zadržavanja startne pozicije")
    plt.title("Stabilnost starta po stazama (n ≥ 150)")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    print("\nStabilnost startne pozicije po stazama:\n")
    display(prob_same.sort_values("same_position", ascending=False))


# =====================================================
# GRAFIKON 4
# Rizik velikog pada za vrh grida (P1–P5)
# =====================================================
def plot_top5_big_drop_risk(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position"]
    ).copy()

    # promena pozicije (pozitivno = napredovao)
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # samo startne pozicije 1–5
    top5 = df[df["start_position"] <= 5].copy()

    # veliki pad (izgubljeno >= 5 mesta)
    top5["big_drop"] = top5["position_change"] <= -5

    # verovatnoca
    prob_drop = top5["big_drop"].mean()

    # grafikon
    plt.figure(figsize=(6, 4))
    plt.bar(["P1–P5"], [prob_drop])

    plt.ylabel("Verovatnoća velikog pada (≥5 mesta)")
    plt.title("Rizik velikog pada za vrh grida")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("Verovatnoća velikog pada za P1–P5:", prob_drop)


# =====================================================
# GRAFIKON 5
# Veliko napredovanje iz donjeg dela grida (P15–P20)
# =====================================================
def plot_bottom_grid_big_gain(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position"]
    ).copy()

    # promena pozicije
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # donji deo grida
    bottom_group = df[
        df["start_position"] >= 15
    ].copy()

    # veliko napredovanje (>=5 mesta)
    bottom_group["big_gain"] = (
        bottom_group["position_change"] >= 5
    )

    # verovatnoca
    prob_gain = bottom_group["big_gain"].mean()

    # grafikon
    plt.figure(figsize=(6, 4))
    plt.bar(["P15–P20"], [prob_gain])

    plt.ylabel("Verovatnoća velikog napredovanja (≥5 mesta)")
    plt.title("Napredovanje iz donjeg dela grida")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("Verovatnoća velikog napredovanja za P15–P20:", prob_gain)
    print("Broj vozača u analizi:", len(bottom_group))


# =====================================================
# GRAFIKON 6
# Napredovanje (>= +5) po segmentima grida, po stazama
# (Top staze po broju uzoraka)
# =====================================================
def plot_big_gain_by_grid_segment_and_circuit(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "circuit_name"]
    ).copy()

    # promena pozicije: pozitivno = napredovao
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # segmentacija grida
    def segment_grid(pos):
        if pos <= 5:
            return "P1–P5"
        elif pos <= 10:
            return "P6–P10"
        elif pos <= 15:
            return "P11–P15"
        else:
            return "P16–P20"

    order = ["P1–P5", "P6–P10", "P11–P15", "P16–P20"]

    df["grid_segment"] = df["start_position"].apply(segment_grid)

    # veliki napredak (>= +5 pozicija)
    df["big_gain"] = df["position_change"] >= 5

    # parametri filtriranja
    MIN_N = 200
    TOP_CIRCUITS = 15

    # top staze po broju uzoraka (uz MIN_N filter)
    counts = (
        df.groupby("circuit_name")
        .size()
        .reset_index(name="n")
    )

    counts = (
        counts[counts["n"] >= MIN_N]
        .sort_values("n", ascending=False)
    )

    top_circuits = counts.head(TOP_CIRCUITS)["circuit_name"].tolist()
    df_top = df[df["circuit_name"].isin(top_circuits)].copy()

    # verovatnoce po (staza, segment)
    stats = (
        df_top.groupby(["circuit_name", "grid_segment"])["big_gain"]
        .mean()
        .reset_index()
    )

    # obezbedi redosled segmenata
    stats["grid_segment"] = pd.Categorical(
        stats["grid_segment"],
        categories=order,
        ordered=True
    )
    stats = stats.sort_values(["circuit_name", "grid_segment"])

    # crtanje
    plt.figure(figsize=(12, 6))
    for seg in order:
        tmp = stats[stats["grid_segment"] == seg]
        plt.scatter(
            tmp["circuit_name"],
            tmp["big_gain"],
            s=90,
            label=seg
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Verovatnoća velikog napredovanja (≥5 mesta)")
    plt.xlabel("Staza")
    plt.title(
        f"Napredovanje (≥+5) po segmentima grida, po stazama "
        f"(Top {TOP_CIRCUITS}, n ≥ {MIN_N})"
    )
    plt.ylim(0, 1)
    plt.legend(title="Startni segment")
    plt.tight_layout()
    plt.show()

    display(stats)

    return stats


# =====================================================
# GRAFIKON 7
# Distribucija promena pozicije pre i posle DRS-a
# =====================================================
def plot_position_change_distribution_drs(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "year"]
    ).copy()

    # apsolutna promena pozicije
    df["abs_change"] = (
        df["start_position"] - df["finish_position"]
    ).abs()

    # period (pre / posle DRS)
    df["period"] = df["year"].apply(
        lambda y: "Pre DRS (2000–2010)"
        if y <= 2010
        else "Posle DRS (2011–2024)"
    )

    plt.figure(figsize=(8, 5))

    for period, color in zip(
        ["Pre DRS (2000–2010)", "Posle DRS (2011–2024)"],
        ["#0553a0", "#e74c3c"]
    ):
        subset = df[df["period"] == period]

        plt.hist(
            subset["abs_change"],
            bins=15,
            alpha=0.5,
            label=period
        )

    plt.xlabel("Apsolutna promena pozicije")
    plt.ylabel("Frekvencija")
    plt.title("Distribucija promena pozicije pre i posle DRS-a")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nBroj uzoraka po periodu:")
    print(df["period"].value_counts())

# =====================================================
# GRAFIKON 8
# Promena pozicije pre vs posle DRS (boxplot)
# =====================================================
def plot_position_change_boxplot_drs(results_clean):

    df_drs = results_clean.dropna(
        subset=["drs_period", "position_change"]
    ).copy()

    # stabilan redosled perioda
    order = [
        p for p in ["pre_drs", "post_drs"]
        if p in df_drs["drs_period"].unique()
    ]

    data = [
        df_drs.loc[
            df_drs["drs_period"] == p,
            "position_change"
        ].values
        for p in order
    ]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        data,
        tick_labels=order
    )

    plt.xlabel("Period")
    plt.ylabel("Promena pozicije (start - finish)")
    plt.title("Promena pozicije — pre vs posle DRS (boxplot)")

    plt.tight_layout()
    plt.show()

    print("\nStatistika po periodu:")
    print(
        df_drs.groupby("drs_period")["position_change"]
        .describe()[["mean", "std", "min", "max"]]
    )

# =====================================================
# GRAFIKON 9
# Mala vs velika napredovanja pre i posle DRS-a
# =====================================================
def plot_small_vs_big_gains_drs(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "year"]
    ).copy()

    # promena pozicije
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # period
    df["period"] = df["year"].apply(
        lambda y: "Pre DRS" if y <= 2010 else "Posle DRS"
    )

    # definicije napredovanja
    df["small_gain"] = df["position_change"].between(1, 3)
    df["big_gain"] = df["position_change"] >= 5

    # verovatnoce po periodu
    stats = (
        df.groupby("period")[["small_gain", "big_gain"]]
        .mean()
        .reset_index()
    )

    # stabilan redosled
    order = ["Pre DRS", "Posle DRS"]
    stats["period"] = pd.Categorical(
        stats["period"],
        categories=order,
        ordered=True
    )
    stats = stats.sort_values("period")

    # grouped bar plot
    x = np.arange(len(stats["period"]))
    w = 0.35

    plt.figure(figsize=(7, 4))

    plt.bar(
        x - w/2,
        stats["small_gain"],
        width=w,
        label="Malo napredovanje (+1 do +3)",
        color="#2ecc71"
    )

    plt.bar(
        x + w/2,
        stats["big_gain"],
        width=w,
        label="Veliko napredovanje (≥ +5)",
        color="#e74c3c"
    )

    plt.xticks(x, stats["period"])
    plt.ylabel("Verovatnoća")
    plt.title("Napredovanja pre i posle DRS-a (mala vs velika)")
    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nVerovatnoće napredovanja:")
    display(stats)


# =====================================================
# GRAFIKON 10
# Trend korelacije startne i finalne pozicije kroz vreme
# =====================================================
def plot_start_finish_correlation_trend(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "year"]
    ).copy()

    # sigurnost: numeric tipovi
    df["start_position"] = pd.to_numeric(
        df["start_position"], errors="coerce"
    )
    df["finish_position"] = pd.to_numeric(
        df["finish_position"], errors="coerce"
    )

    df = df.dropna(
        subset=["start_position", "finish_position", "year"]
    )

    # korelacija po godinama
    year_corr = (
        df.groupby("year")[["start_position", "finish_position"]]
          .corr()
          .iloc[0::2, -1]   # start vs finish korelacija
          .reset_index()
    )

    year_corr.columns = ["year", "level_1", "correlation"]
    year_corr = year_corr[
        year_corr["level_1"] == "start_position"
    ][["year", "correlation"]]

    # plot
    plt.figure(figsize=(10, 5))

    plt.plot(
        year_corr["year"],
        year_corr["correlation"]
    )

    # vertikalna linija — DRS
    plt.axvline(
        2011,
        linestyle="--",
        label="Uvođenje DRS (2011)"
    )

    plt.xlabel("Sezona")
    plt.ylabel("Korelacija start–finish")
    plt.title("Trend korelacije startne i finalne pozicije kroz vreme")

    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nProsečna korelacija:")
    print(year_corr["correlation"].describe()[["mean", "min", "max"]])

