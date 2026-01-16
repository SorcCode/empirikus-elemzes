import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        # Handle different Stable Baselines 3 versions
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        equity_curve.append(vec_env.get_attr("equity_usd")[0])

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    # ==========================================
    # 1. SETTINGS - Choose what to test
    # ==========================================
    ASSET = "BTC"   # "BTC" or "EURO"
    MODE  = "TEST"   # "LEARN" (for 20% validation) or "TEST" (for 2024+ data)
    
    # ==========================================
    # 2. DYNAMIC PATH LOGIC
    # ==========================================
    if ASSET == "BTC":
        folder = "BTC_USD 1 day"
        file = "BTC_Daily_Learning_2020_2023.csv" if MODE == "LEARN" else "BTC_Daily_Testing_2024_Present.csv"
    else:
        folder = "EURO_USD 15min"
        file = "EURUSD_15_Mins_Ask_2020.12.06_2025.12.12.csv" if MODE == "LEARN" else "test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"

    file_path = os.path.join("data", folder, file)
    print(f"--- Running Test on: {file_path} ---")

    # Load and Preprocess
    df, feature_cols = load_and_preprocess_data(file_path)

    # 3. SELECT DATA SLICE
    # If testing on the LEARN file, we use the last 20% (the part the bot didn't train on)
    # If testing on the TEST file, we use the whole thing
    if MODE == "LEARN":
        split_idx = int(len(df) * 0.8)
        eval_df = df.iloc[split_idx:].copy()
        print(f"Testing on 20% validation slice ({len(eval_df)} bars)")
    else:
        eval_df = df.copy()
        print(f"Testing on full Out-of-Sample file ({len(eval_df)} bars)")

    # ==========================================
    # 4. ENVIRONMENT CONFIG (Must match Training)
    # ==========================================
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30

    test_env = ForexTradingEnv(
        df=eval_df,
        window_size=WIN,
        sl_options=SL_OPTS,
        tp_options=TP_OPTS,
        spread_pips=1.0,
        commission_pips=0.0,
        max_slippage_pips=0.2,
        random_start=False,
        episode_max_steps=None,
        feature_columns=feature_cols,
        hold_reward_weight=0.0,
        open_penalty_pips=0.0, 
        time_penalty_pips=0.0, 
        unrealized_delta_weight=0.0
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # ==========================================
    # 5. LOAD MODEL & RUN
    # ==========================================
    model_path = "model_eurusd_best"
    if not os.path.exists(model_path + ".zip"):
         print(f"Error: {model_path}.zip not found! Train the agent first.")
         return

    model = PPO.load(model_path, env=vec_test_env)
    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # ==========================================
    # 6. RESULTS & PLOTTING
    # ==========================================
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = f"results_{ASSET}_{MODE}_history.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Success! {len(closed_trades)} trades saved to {out_csv}")
    else:
        print("Bot finished but no trades were closed.")

    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label=f"Equity ({ASSET} {MODE})", color='green')
    plt.title(f"Equity Curve - {ASSET} {MODE} Evaluation")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.axhline(y=equity_curve[0], color='r', linestyle='--', label="Starting Balance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()