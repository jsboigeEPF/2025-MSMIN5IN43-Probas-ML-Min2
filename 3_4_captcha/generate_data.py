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
ALPHABET = string.ascii_uppercase + string.digits
WIDTH = 220
HEIGHT = 64
NUM_IMAGES = 8000

# Captchas object Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.join(SCRIPT_DIR, "fonts")

captcha: ImageCaptcha = ImageCaptcha(
    width=WIDTH,
    height=HEIGHT,
    fonts=[
        os.path.join(FONTS_DIR, "NotoSans-Regular.ttf"),
        os.path.join(FONTS_DIR, "AdwaitaSans-Regular.ttf"),
        os.path.join(FONTS_DIR, "Hack-Regular.ttf"),
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


def generate_dataset(force=False, root_dir=None, num_images=NUM_IMAGES):
    global DATA_ROOT, OUTPUT_DIR, CSV_FILE

    if root_dir:
        DATA_ROOT = root_dir
    elif os.path.exists("data"):
        DATA_ROOT = "data"
    elif os.path.exists("../data"):
        DATA_ROOT = "../data"
    else:
        # Fallback creation if it doesn't exist anywhere
        DATA_ROOT = "data"

    OUTPUT_DIR = os.path.join(DATA_ROOT, "images")
    CSV_FILE = os.path.join(DATA_ROOT, "dataset.csv")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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

    # Check if data exists
    if not os.listdir(OUTPUT_DIR) or force:
        print(f"Génération de {num_images} captchas image (force={force})...")

        # If forcing, maybe good to clear, but overwriting is faster if size is same.
        # We assume NUM_IMAGES is constant or growing.

        for i in range(num_images):
            text = generate_random_text()
            text = generate_random_length_text()
            filename = f"{i}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            data.append([filename, text])
            captcha.write(text, filepath)
            if i % 1000 == 0:
                print(f"  {i}/{num_images}...")
    else:
        # Check if CSV exists, if not we might need to regenerate anyway or error out?
        # Original logic raised exception if images existed.
        # We will preserve original behavior: if images exist and not force, raise.
        raise Exception("There is already data in images")

    # Always write/rewrite CSVs if we generated data (force=True) or if they differ
    if force or not os.path.isfile(CSV_FILE):
        print("Writing dataset.csv...")
        df = pd.DataFrame(data, columns=["filename", "Label"])
        df.to_csv(CSV_FILE, index=False)

    # Always recreate train/val splits if we generated new data
    if force or not os.path.isfile(os.path.join(DATA_ROOT, "train.csv")):
        print(f"Generating train/val splits...")
        if df is None:  # Should have been created above if force=True
            # If force=False but CSV missing, we need to load data?
            # Original script didn't handle "Images exist but CSV missing" well without generating 'data' list.
            # But here we focus on the force=True path for the user request.
            pass

        if df is not None:
            train_df, val_df = train_test_split(
                df, test_size=0.1, random_state=42
            )
            train_df.to_csv(os.path.join(DATA_ROOT, "train.csv"), index=False)
            val_df.to_csv(os.path.join(DATA_ROOT, "val.csv"), index=False)

    print("Terminé !")


if __name__ == "__main__":
    generate_dataset()
