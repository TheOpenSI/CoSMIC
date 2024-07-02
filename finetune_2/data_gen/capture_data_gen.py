import pandas as pd
import typing

from capture_data_util import CHESS_PIECE, INITIAL_POSITIONS, get_pair, get_explanation

RANDOM_STATE = 42

# load_data
df = pd.read_csv("/home/s448780/workspace/cognitive_ai/data/lichess_main.csv", index_col=False)
df = df.sample(50, random_state=RANDOM_STATE)
target_moves = df["moves"]

# dataframe 
capture_data = pd.DataFrame(columns=["moves", "game_length", "capture_explanantion"])

for i, move in enumerate(target_moves):
    # print(i)
    # print(move)
    move_pair, len_move_pair = get_pair(move)
    capture_explanation = get_explanation(move_pair, move)
    capture_data.loc[i] = [move, len_move_pair, capture_explanation]

print("[INFO] Saving File..")
capture_data.to_csv("./capture_explanation.csv", index=False)