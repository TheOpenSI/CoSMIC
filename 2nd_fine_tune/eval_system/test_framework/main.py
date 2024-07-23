import os
import pandas as pd


def call_model(questions):
    return "LLM will be added here"


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def generate_output_path(input_path):
    path_components = input_path.split(os.sep)
    
    try:
        index = path_components.index('test_framework')
    except ValueError:
        raise ValueError("'test_framework' directory not found in the input path")
    
    # getting the path to the "result" directory inside "test_framework"
    result_dir = os.sep.join(path_components[:index + 1] + ['result'] + path_components[index + 1:-1])
    # if does not exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # _result suffix
    filename = path_components[-1]
    new_filename = f"{os.path.splitext(filename)[0]}_result.csv"
    
    output_path = os.path.join(result_dir, new_filename)
    
    return output_path


def process_csv_files():

    current_dir = "/home/s448780/workspace/cognitive_ai/2nd_fine_tune/eval_system/test_framework"

    all_tests = []

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        # check if directory
        if os.path.isdir(item_path):
            # CSV file in the subdirectory
            for file in os.listdir(item_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(item_path, file)
                    all_tests.append(file_path)

    print(all_tests)
    
    for test_path in all_tests:
        test = pd.read_csv(test_path)

        questions = test["Question"]
        answers = test["Answer"]

        print(f"[INFO] Evaluating test {test_path}..")

        response_df = pd.DataFrame(columns=["Question", "GT", "Response", "Score"])
        for i, question in enumerate(questions):
            response = call_model(question)
            gt = answers[i]
            response_df.loc[response_df.shape[0]] = {"Question": question, "GT": gt, "Response": response, "Score": 0}
            print(f"[INFO] Question {i+1} completed")
                    
        # save df
        new_file_path = generate_output_path(test_path)
        response_df.to_csv(new_file_path, index=False)
        print(f"[INFO] Test completed. Generated result at {new_file_path}")
        print("="*30)

if __name__ == "__main__":
    process_csv_files()

