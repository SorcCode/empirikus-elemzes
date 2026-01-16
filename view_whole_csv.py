import os
from pandasgui import show
from indicators import load_and_preprocess_data

# ==========================================
# SETTINGS - Change these to switch data
# ==========================================
ASSET = "EURO"    # Options: "BTC" or "EURO"
MODE  = "TEST"  # Options: "LEARN" or "TEST"

# ==========================================
# PATH LOGIC (Double-checked against your images)
# ==========================================
if ASSET == "BTC":
    folder = "BTC_USD 1 day"
    # LEARN = 2020_2023 | TEST = 2024_Present
    file = "BTC_Daily_Learning_2020_2023.csv" if MODE == "LEARN" else "BTC_Daily_Testing_2024_Present.csv"
else:
    folder = "EURO_USD 15min"
    # LEARN = 15 Mins Ask | TEST = test_EURUSD_Candlestick...
    file = "EURUSD_15_Mins_Ask_2020.12.06_2025.12.12.csv" if MODE == "LEARN" else "test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"

# Build the final path
csv_path = os.path.join("data", folder, file)

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    print(f"--- Loading {ASSET} {MODE} Mode ---")
    print(f"Path: {csv_path}")
    
    df, feature_cols = load_and_preprocess_data(csv_path)
    show(df)