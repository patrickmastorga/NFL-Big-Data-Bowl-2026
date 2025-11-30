import os
import pandas as pd


def compute_stats_from_file(path: str):
    df = pd.read_csv(path)

    # ensure necessary columns exist
    required_cols = {
        'game_id', 'play_id', 'nfl_id', 'frame_id',
        'player_to_predict', 'num_frames_output'
    }
    if not required_cols.issubset(df.columns):
        print(f"Skipping {path} because required columns are missing.")
        return None

    results = {
        "max_A": 0,
        "max_P": 0,
        "max_T_pre": 0,
        "max_T_post": 0
    }

    # compute max_T_pre and max_T_post from columnwise maxes
    results["max_T_pre"] = df["frame_id"].max()
    results["max_T_post"] = df["num_frames_output"].max()

    # group data by play
    for (g, p), play_df in df.groupby(["game_id", "play_id"]):

        # compute n_players: number of distinct players in this play, update max_P
        n_players = play_df["nfl_id"].nunique()
        results["max_A"] = max(results["max_A"], n_players)

        # compute n_players_S: number of distinct players to predict in this play, update max_S
        n_players_S = play_df[play_df['player_to_predict'] == 1]["nfl_id"].nunique()
        results["max_P"] = max(results["max_P"], n_players_S)

    return results

def compute_global_stats(folder_path: str):
    global_stats = {
        "max_A": 0,
        "max_P": 0,
        "max_T_pre": 0,
        "max_T_post": 0
    }

    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv"):
            continue
        if fname.startswith("output"):  # skip output templates
            continue

        path = os.path.join(folder_path, fname)
        stats = compute_stats_from_file(path)

        if stats is None:
            continue

        print(f"{fname}: {stats}")

        # update global
        global_stats["max_A"] = max(global_stats["max_A"], stats["max_A"])
        global_stats["max_P"] = max(global_stats["max_P"], stats["max_P"])
        global_stats["max_T_pre"] = max(global_stats["max_T_pre"], stats["max_T_pre"])
        global_stats["max_T_post"] = max(global_stats["max_T_post"], stats["max_T_post"])

    print("\n========================")
    print("GLOBAL TRAINING STATS:")
    for k, v in global_stats.items():
        print(f"{k}: {v}")
    print("========================\n")

    return global_stats