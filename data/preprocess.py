import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def preprocess_folder_into_dataset(folder_path: str, save_path: str) -> None:
    """
    Processes the NFL big data bowl training data contained in [input/output]_[...].csv files and the formated specified by the competition
    into an python list of dicts containing the entries
    ```python
    [
        {
            "game_id": int,
            "play_id": int,
            "pre_xy": np.array([...]),        # shape (A, T_pre, 2)
            "post_xy": np.array([...]),       # shape (P, T_post, 2)
            "roles": np.array([...]),         # shape (A,)
            "roles_S": np.array([...]),       # shape (P,)
            "ball_landing": np.array([x, y]), # shape (2,)
        },
        ...
    ] 
    ```
    Args:
        folder_path (str): the path to the folder containing the NFL big data bowl training data files
        save_path (str): the file path for saving the generated python list of dicts to the disk using pickle.dump()
    """
    # scan the dataset to build vocabulary of player_role
    role_set = set()
    for fname in os.listdir(folder_path):
        if not (fname.startswith("input_") and fname.endswith(".csv")):
            continue
        path_in = os.path.join(folder_path, fname)
        print(f"Scanning {path_in}...")
        df_in = pd.read_csv(path_in)

        # accumulate roles
        if "player_role" in df_in.columns:
            role_set.update(df_in["player_role"].dropna().unique())

    role_to_id = {r: i for i, r in enumerate(sorted(role_set))}
    print(f"Built role vocabulary: {role_to_id}")

    # build the dataset
    plays = []
    for fname in os.listdir(folder_path):
        # read the input and output .csv files
        if not (fname.startswith("input_") and fname.endswith(".csv")):
            continue

        path_in = os.path.join(folder_path, fname)
        out_name = fname.replace("input_", "output_")
        path_out = os.path.join(folder_path, out_name)

        if not os.path.exists(path_out):
            print(f"Warning: no matching output file for {path_in}, skipping.")
            continue
            
        df_in = pd.read_csv(path_in)
        df_out = pd.read_csv(path_out)

        # ensure necessary columns exist
        required_cols_in = {
            'game_id', 'play_id', 'nfl_id', 'frame_id', 'player_to_predict', 'player_role',
            'ball_land_x', 'ball_land_y', 'absolute_yardline_number', 'play_direction', 'x', 'y'
        }
        required_cols_out = {
            'game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y'
        }
        if not required_cols_in.issubset(df_in.columns):
            print(f"Skipping {path_in} because required input columns are missing.")
            continue
        if not required_cols_out.issubset(df_out.columns):
            print(f"Skipping {path_in} because required output columns are missing.")
            continue

        # process the training data
        print(f"Processing {path_in}...")
        # group by play
        for (game_id, play_id), play_in in tqdm(df_in.groupby(["game_id", "play_id"])):
            play_out = df_out[(df_out["game_id"] == game_id) & (df_out["play_id"] == play_id)]
            if play_out.empty:
                continue

            # players
            predict_ids = np.sort(play_in.loc[play_in["player_to_predict"] == True, "nfl_id"].unique())
            P = len(predict_ids)
            not_predict_ids = np.sort(play_in.loc[play_in["player_to_predict"] == False, "nfl_id"].unique())
            player_ids = np.concatenate([predict_ids, not_predict_ids]) # force the players in S to be the first entries in pre_xy
            A = len(player_ids)

            # frames
            T_pre = play_in["frame_id"].max()
            T_post = play_out["frame_id"].max()

            # position arrays
            pre_xy = np.zeros((A, T_pre, 2), dtype=np.float32)
            post_xy = np.zeros((P, T_post, 2), dtype=np.float32)

            for i, nfl_id in enumerate(player_ids):
                for t in range(T_pre):
                    pre_xy[i, t] = play_in[(play_in["nfl_id"] == nfl_id) & (play_in["frame_id"] == t + 1)][["x", "y"]].values

            for i, nfl_id in enumerate(predict_ids):
                for t in range(T_post):
                    post_xy[i, t] = play_out[(play_out["nfl_id"] == nfl_id) & (play_out["frame_id"] == t + 1)][["x", "y"]].values

            # other data
            roles = np.array([
                role_to_id[play_in[play_in["nfl_id"] == nfl_id].iloc[0]["player_role"]]
                 for nfl_id in player_ids
            ])
            ball_landing = play_in[["ball_land_x", "ball_land_y"]].iloc[0].values.astype(np.float32)

            # Absolute yardline & direction for normalization
            yardline = float(play_in["absolute_yardline_number"].iloc[0])

            # Convert play direction to numeric: left = -1, right = 1
            direction_str = play_in["play_direction"].iloc[0].lower()
            direction = -1 if direction_str == "left" else 1

            play_dict = {
                "game_id": int(game_id),
                "play_id": int(play_id),
                "pre_xy": pre_xy,
                "post_xy": post_xy,
                "roles": roles,
                "ball_landing": ball_landing,
                "yardline": yardline,
                "direction": direction,
            }
            plays.append(play_dict)

    # save to disk
    print(f"Processed {len(plays)} plays!")
    print(f"Saving to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump(plays, f)