import os


def print_answer(num: int, text: str) -> None:
    if not os.path.exists("answers"):
        os.makedirs("answers")

    with open(f"answers/a{num}.txt", "w") as f:
        f.write(text)

    print(text)
