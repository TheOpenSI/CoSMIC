import pandas as pd
data = {
    "Question": [
        "What's the capital of Australia?",
        "What's the capital of Bangladesh?",
        "What's the capital of Germany?",
        "What's the capital of Spain?",
        "__update__store__ Real Madrid won 15 UCL titles.",
        "How many UCL titles does Real Madrid have?",
        "What's the capital of Saudi Arabia?",
        "__next__move__ d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4.",
        "exit"
    ],
    "Answer": [
        "Canberra",
        "Dhaka",
        "Berlin",
        "Madrid",
        "",
        "15",
        "Riyadh",
        "",
        ""
    ]
}

df = pd.DataFrame(data)
df.to_csv("cognition_framework/tests/test.csv", index=False)

print("[INFO] CSV file created successfully.")
