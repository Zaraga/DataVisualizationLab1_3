from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).with_name("spotify-2023.csv")
OUTPUT_DIR = Path(__file__).with_name("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def make_histogram(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(8, 4))
    df[column].hist(bins=20)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{column}_hist.png")
    plt.close()


def make_bar_chart(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(8, 4))
    df[column].value_counts(dropna=False).plot(kind="bar")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{column}_bar.png")
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA_FILE, encoding="ISO-8859-1")

    quant_cols = ["bpm", "danceability_%", "energy_%", "speechiness_%", "acousticness_%"]
    cat_cols = ["key", "mode"]

    print("Loaded rows:", len(df))
    print("\nColumns used:")
    print(["track_name", "artist(s)_name", *quant_cols, *cat_cols])

    print("\n=== Quantitative feature summaries ===")
    ranges = {}
    for col in quant_cols:
        make_histogram(df, col)
        desc = df[col].describe()
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        ranges[col] = (q1, q3)
        print(f"\n{col}")
        print(desc)
        print(f"Typical middle-50% range: {q1:.2f} to {q3:.2f}")

    print("\n=== Categorical feature summaries ===")
    modes = {}
    for col in cat_cols:
        make_bar_chart(df, col)
        counts = df[col].value_counts(dropna=False)
        modes[col] = df[col].mode(dropna=True)[0] if col == "key" else df[col].mode()[0]
        print(f"\n{col}")
        print(counts)
        print(f"Most common value: {modes[col]}")

    bpm_low, bpm_high = ranges["bpm"]
    dance_low, dance_high = ranges["danceability_%"]
    energy_low, energy_high = ranges["energy_%"]
    speech_low, speech_high = ranges["speechiness_%"]
    acoustic_low, acoustic_high = ranges["acousticness_%"]

    print("\n=== Proposed guaranteed smash hit profile ===")
    print(f"Key: {modes['key']}")
    print(f"Mode: {modes['mode']}")
    print(f"BPM: {bpm_low:.0f} to {bpm_high:.0f}")
    print(f"Danceability: {dance_low:.0f} to {dance_high:.0f}")
    print(f"Energy: {energy_low:.0f} to {energy_high:.0f}")
    print(f"Speechiness: {speech_low:.0f} to {speech_high:.0f}")
    print(f"Acousticness: {acoustic_low:.0f} to {acoustic_high:.0f}")

    matches = df[
        (df["key"] == modes["key"]) &
        (df["mode"] == modes["mode"]) &
        (df["bpm"].between(bpm_low, bpm_high)) &
        (df["danceability_%"].between(dance_low, dance_high)) &
        (df["energy_%"].between(energy_low, energy_high)) &
        (df["speechiness_%"].between(speech_low, speech_high)) &
        (df["acousticness_%"].between(acoustic_low, acoustic_high))
    ].copy()

    print("\n=== Songs that match all criteria ===")
    if matches.empty:
        print("No songs matched all criteria.")
    else:
        result_cols = ["track_name", "artist(s)_name", "streams", "in_spotify_charts", "in_apple_charts"]
        print(matches[result_cols].sort_values("streams", ascending=False).to_string(index=False))

    hit_song = df[df["track_name"].str.lower() == "blinding lights"].copy()
    print("\n=== Example known smash hit comparison ===")
    if hit_song.empty:
        print("Blinding Lights was not found in the dataset.")
    else:
        compare_cols = [
            "track_name", "artist(s)_name", "streams", "bpm", "key", "mode",
            "danceability_%", "energy_%", "speechiness_%", "acousticness_%"
        ]
        print(hit_song[compare_cols].to_string(index=False))

    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
