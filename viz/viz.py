import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

field_img = mpimg.imread('/home/patri/code/NFL-Big-Data-Bowl-2026/viz/nfl_field.png')

def create_play_animation(input_df: pd.DataFrame, output_df: pd.DataFrame, game_id: int, play_id: int, title: str | None=None) -> FuncAnimation:
    """
        Args:
            input_df (pd.DataFrame): Must contain the input data for the play to animate, in the format specified in the competition
            output_df (pd.DataFrame): Must contain the output data for the play to animate, in the format specified in the competition
            game_id (int): To specify the game
            play_id (int): To specify the play to animate
            title (str | None): Optional title for the animation
        Returns:
            FuncAnimation: Animation object that can be displayed or saved as a gif/mp4
    """
    input_play_df = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
    output_play_df = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]

    num_frames_input = input_play_df['frame_id'].max()
    num_frames_output = output_play_df['frame_id'].max()

    o_players = input_play_df[input_play_df['player_side'] == 'Offense']['nfl_id'].unique()
    ball_land_x, ball_land_y = input_play_df.iloc[0][['ball_land_x', 'ball_land_y']]
    line_of_scrimmage = input_play_df.iloc[0]['absolute_yardline_number']
    passer_x, passer_y = input_play_df[(input_play_df['player_role'] == 'Passer') & (input_play_df['frame_id'] == num_frames_input)].iloc[0][['x', 'y']]
    ball_flight = np.linspace(np.array([passer_x, passer_y]), np.array([ball_land_x, ball_land_y]), num_frames_output)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    o_scatter_predict = ax.scatter([], [], c='blue', s=200, marker='*')
    d_scatter_predict = ax.scatter([], [], c='red', s=200, marker='*')
    o_scatter_not_predict = ax.scatter([], [], c='blue', s=100)
    d_scatter_not_predict = ax.scatter([], [], c='red', s=100)
    ball_scatter = ax.scatter([], [], c='brown', s=100)

    def init():
        if title:
            ax.set_title(title, fontsize=20)
        ax.set_xlim(-2, 122)
        ax.set_ylim(-2, 166/3)
        ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax.set_yticks([0, 160/3])

        ax.imshow(field_img, extent=(-2, 122, -2, 166/3), aspect='equal')
        ax.axvline(x=line_of_scrimmage, color='yellow', linewidth=2)
        ax.scatter([ball_land_x], [ball_land_y], color='purple', marker='X', s=100)

        fig.tight_layout()
        return o_scatter_predict, d_scatter_predict, o_scatter_not_predict, d_scatter_not_predict, ball_scatter

    def update(frame):
        frame_id = frame + 1
        if frame_id <= num_frames_input:
            current_df = input_play_df[input_play_df['frame_id'] == frame_id]

            x = current_df.loc[(current_df['player_side'] == 'Offense') & (current_df['player_to_predict'] == True), ['x', 'y']].values
            o_scatter_predict.set_offsets(x)
            x = current_df.loc[(current_df['player_side'] == 'Defense') & (current_df['player_to_predict'] == True), ['x', 'y']].values
            d_scatter_predict.set_offsets(x)
            x = current_df.loc[(current_df['player_side'] == 'Offense') & (current_df['player_to_predict'] == False), ['x', 'y']].values
            o_scatter_not_predict.set_offsets(x)
            x = current_df.loc[(current_df['player_side'] == 'Defense') & (current_df['player_to_predict'] == False), ['x', 'y']].values
            d_scatter_not_predict.set_offsets(x)

            o_scatter_not_predict.set_alpha(1)
            d_scatter_not_predict.set_alpha(1)
            ball_scatter.set_alpha(0)

        else:
            frame_id -= num_frames_input
            current_df = output_play_df[output_play_df['frame_id'] == frame_id]

            x = current_df.loc[current_df['nfl_id'].isin(o_players), ['x', 'y']].values
            o_scatter_predict.set_offsets(x)
            x = current_df.loc[~current_df['nfl_id'].isin(o_players), ['x', 'y']].values
            d_scatter_predict.set_offsets(x)
            
            ball_scatter.set_offsets(ball_flight[frame_id - 1])

            o_scatter_not_predict.set_alpha(0.5)
            d_scatter_not_predict.set_alpha(0.5)
            ball_scatter.set_alpha(1)

        return o_scatter_predict, d_scatter_predict, o_scatter_not_predict, d_scatter_not_predict, ball_scatter

    return FuncAnimation(fig, update, frames=(num_frames_input + num_frames_output), init_func=init)

if __name__ == "__main__":
    input_df = pd.read_csv('/home/patri/code/NFL-Big-Data-Bowl-2026/nfl-big-data-bowl-2026-prediction/train/input_2023_w01.csv')
    output_df = pd.read_csv('/home/patri/code/NFL-Big-Data-Bowl-2026/nfl-big-data-bowl-2026-prediction/train/output_2023_w01.csv')
    game_id = 2023090700
    play_id = 101

    ani = create_play_animation(input_df, output_df, game_id, play_id, title="Play Animation")
    ani.save("play_animation.gif", writer='pillow', fps=10)
    plt.show()