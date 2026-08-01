import re
from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytesseract
import requests
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SURFACE_IMG = "https://shorestations.ucsd.edu/plots/SIO_surf_temp_now.png"
BOTTOM_IMG = "https://shorestations.ucsd.edu/plots/SIO_bot_temp_now.png"


def ocr_img_text(img_url: str) -> str:
    """
    convert the text in the image into text
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(img_url, headers=headers, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))

    img = img.convert("L")
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)

    return pytesseract.image_to_string(img)


# ---------------------------------------------------------------------------- #


def parse_temp(img_text: str) -> str:
    """
    parse out the temperature from the image text
    """
    c_match = re.search(r"(-?\d+\.?\d*)\s*°?\s*C", img_text)

    temp_c = float(c_match.group(1)) if c_match else None

    return temp_c


# ---------------------------------------------------------------------------- #


def parse_date(img_text: str) -> str:
    DATE_PATTERNS = [
        r"\d{1,2}/\d{1,2}/\d{2,4}",  # 7/27/2026 or 07/27/26
        r"\d{4}-\d{1,2}-\d{1,2}",  # 2026-07-27
        r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}",  # July 27, 2026 / Jul. 27 2026
    ]
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, img_text)
        if match:
            return match.group(0)
    return None


# ---------------------------------------------------------------------------- #


def fetch_water_info() -> pd.DataFrame:
    try:
        surface_img_text = ocr_img_text(SURFACE_IMG)

        surface_temp = parse_temp(surface_img_text)

        date_text = parse_date(surface_img_text)
        date_today = datetime.strptime(date_text, "%B %d, %Y")  # noqa: DTZ007

        bottom_img_text = ocr_img_text(BOTTOM_IMG)
        bottom_temp = parse_temp(bottom_img_text)

        df = pd.DataFrame(
            [{"date": date_today, "surf_temp_c": surface_temp, "bottom_temp": bottom_temp}]
        )
        
    except:
        date_today = datetime.now(tz=ZoneInfo("America/Los_Angeles"))
        df = pd.DataFrame(
            [{"date": date_today, "surf_temp_c": np.nan, "bottom_temp": np.nan}]
        )

    df['date'] = df.date.dt.normalize()
    df['date'] = df['date'].dt.tz_localize(None)
    return df
