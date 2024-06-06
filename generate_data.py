import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import textwrap

def process_response(response):
    lines = response.replace("", "").replace("", "").split("\n")
    wrapped_lines = [textwrap.fill(line, width=100) for line in lines]
    for wrapped_line in wrapped_lines:
        print(wrapped_line)

def get_api_key(args):
    if args.i:
        print(f"[Info] Loading token from {args.i}")
        with open(args.i, "r") as file :
            return file.readline()

    elif args.mI:
        print("[Info] User will be requested for token")
        return input("Please enter your OpenAI token.\n")

    else:
        load_dotenv()
        return os.getenv("openai_api_key")

def gen_cot_prompt(prev_moves : str, last_move : str) -> str:
    if prev_moves == "None":
        return f"White : {last_move}"
    else: 
        prev_moves = (prev_moves+f" {last_move}").split(" ")
        cot_moves = [f"White: {prev_moves[i]}, Black: {prev_moves[i+1]}" if i + 1 < len(prev_moves) else f"White: {prev_moves[i]}" for i in range(0, len(prev_moves), 2)]
        return "\n".join(cot_moves)

def generate_explanation(client : OpenAI, df : pd.DataFrame, sytem_content : str) -> pd.DataFrame:
    return_df = df.copy()

    for i in range(df.shape[0]):
        print(f"[Info] Generating explanation for row {i}")
        prev_moves = df.iloc[i]["prev_moves"]
        last_move = df.iloc[i]["last_move"]

        user_prompt_move = gen_cot_prompt(prev_moves, last_move)
        user_content_cot = f"Previous move pairs are - \n{user_prompt_move}"
        
        stream = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages = [{"role": "system",
                     "content": sytem_content},
                    {"role": "user",
                     "content": user_content_cot}],
        stream=True,)
        
        explanation = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                explanation += chunk.choices[0].delta.content
        return_df.loc[i, "explanation"] = explanation.strip()
    return return_df

def arg_setup() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="[Required] path to the input data cluster(csv file)")
    parser.add_argument("-i", help = "use a .txt file to pass token")
    parser.add_argument("-mI", action = "store_true", help = "manually enter your token")
    return parserget_file_path

def load_df(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.drop("opening_name", axis = 1, inplace=True)
    df = df.fillna("None")
    df["explanation"] = " "
    return df

def get_file_path(file_number):
      if 0 <= file_number <= 9:
          return f"./data/data_{file_number:02d}.csv"

      else:
          print(f"Invalid file number: {file_number}. Please enter a number between 0 and 19.")
          return None


if __name__ == "__main__":
    args = arg_setup().parse_args()

    openai_api_key = get_api_key(args)

    target_file_path = get_file_path(int(args.file))
    target_df  = load_df(target_file_path)
    # target_df = target_df.head().copy()

    # selected prompt
    system_content_cot = '''Assume you are a chess master.
    You will be provided with a list of chess move pairs in Algebraic Notation where 1st move is by White and 2nd by Black.
    Your task is to analyse each pair and then generate the rationale behind each White and Black move and then the last move.
    '''

    target_df = generate_explanation(OpenAI(api_key = openai_api_key), target_df, system_content_cot)
    target_df.to_csv(f"./exp_data/generated_data_{int(args.file):02d}.csv")
    print("[Info] CSV generated")