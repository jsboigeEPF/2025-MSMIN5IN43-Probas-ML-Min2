import os
import random
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from captcha.image import ImageCaptcha

# Configuration
DATA_ROOT = "data"
OUTPUT_DIR = os.path.join(DATA_ROOT, "images")
CSV_FILE = os.path.join(DATA_ROOT, "dataset.csv")
ALPHABET = string.ascii_uppercase
WIDTH = 220
HEIGHT = 64
NUM_IMAGES = 12000  # 5000 images simples

# Captchas object Configuration
captcha: ImageCaptcha = ImageCaptcha(
    width=WIDTH,
    height=HEIGHT,
    fonts=[
        "./fonts/NotoSans-Regular.ttf",
        "./fonts/AdwaitaSans-Regular.ttf",
        "./fonts/Hack-Regular.ttf",
    ],
    font_sizes=(30, 40, 50),  # NOQA: Type hint was done wrong
)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def generate_random_text(length=6) -> str:
    return "".join(random.choices(ALPHABET, k=length))


def generate_random_length_text() -> str:
    text_length: int = random.randint(a=4, b=8)
    return "".join(random.choices(ALPHABET, k=text_length))


def generate_dataset():
    data = []
    (
        df,
        train_df,
        val_df,
    ) = (
        None,
        None,
        None,
    )
    print(f"Génération de {NUM_IMAGES} captchas image...")
    if not os.listdir(OUTPUT_DIR):
        for i in range(NUM_IMAGES):
            text = generate_random_text()
            text = generate_random_length_text()
            filename = f"{i}.png"
            filepath = f"{OUTPUT_DIR}/{filename}"
            data.append([filename, text])
            captcha.write(text, filepath)
            if i % 1000 == 0:
                print(f"  {i}/{NUM_IMAGES}...")
    else:
        raise Exception("There is already data in images")
    if not os.path.isfile(CSV_FILE):
        print("Data file doesn't exists. Generating a new one")
        df = pd.DataFrame(data, columns=["filename", "Label"])
        df.to_csv(CSV_FILE, index=False)
    if not os.path.isfile(DATA_ROOT + "/train.csv"):
        print(f"Train file doesn't exists. Generating a new one")
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df.to_csv(os.path.join(DATA_ROOT, "train.csv"), index=False)
    if not os.path.isfile(DATA_ROOT + "/val.csv"):
        print(f"Val file doesn't exists. Generating a new one")
        val_df.to_csv(os.path.join(DATA_ROOT, "val.csv"), index=False)
    print("Terminé !")


if __name__ == "__main__":
    generate_dataset()
