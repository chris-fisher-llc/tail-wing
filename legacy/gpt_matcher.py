import os
import csv
import io
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# === Load API Key ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Config ===
#INPUT_FILE = "raw_scraped_data.csv"
OUTPUT_FILE = "matched_output.csv"

# === Prompt Builder ===
def build_prompt(csv_chunk: str) -> str:
    return (
        "You are helping normalize sportsbook betting data.\n"
        "Each row includes: player, odds, odds_type, and book.\n"
        "Group rows by normalized_player and bet_type (e.g., 'to hit 1+ HR').\n"
        "Pivot the data so that each row corresponds to one normalized_player and one bet_type.\n"
        "Create one column per sportsbook using the format: DraftKings_Odds, FanDuel_Odds, BetMGM_Odds, etc.\n"
        "Fill in odds under the correct book column. Leave missing books blank.\n"
        "Return a CSV with columns: normalized_player, bet_type, player, DraftKings_Odds, FanDuel_Odds, BetMGM_Odds, Caesars_Odds, etc.\n"
        "Output ONLY the CSV. No commentary.\n\n"
        f"{csv_chunk}"
    )

# === GPT Call ===
def call_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# === Main Normalization Function ===
def normalize_and_match(df: pd.DataFrame):
    csv_content = df.to_csv(index=False)
    prompt = build_prompt(csv_content)
    print(f"📤 Sending {len(df)} rows to GPT...")

    raw = call_gpt(prompt)
    print("\n🧾 Raw GPT output preview:\n")
    print(raw[:1500])

    # Clean GPT formatting
    if "```csv" in raw:
        raw = raw.split("```csv")[1]
    if "```" in raw:
        raw = raw.split("```")[0]
    raw = raw.strip()

    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames
    fieldnames += ["value", "value_book", "value_odds", "value_flag"]

    if not fieldnames or len(fieldnames) < 2:
        print("⚠️ Detected malformed CSV — only one column found.")
        print("Raw preview:\n", raw[:500])
        return

    all_rows = []
    for row in reader:
        all_rows.append({k: str(row.get(k, "")).strip() for k in fieldnames})

    # Detect value bets
    for row in all_rows:
        odds_fields = [k for k in row if k.endswith("_Odds") and row[k].strip()]
        try:
            odds_values = [(book, int(row[book])) for book in odds_fields]
        except ValueError:
            continue

        if len(odds_values) < 2:
            continue

        max_book, max_odds = max(odds_values, key=lambda x: x[1])
        other_odds = [odds for book, odds in odds_values if book != max_book]
        avg_other = sum(other_odds) / len(other_odds)

        if max_odds >= avg_other * 1.10:
            row["value"] = max_odds / avg_other
            row["value_book"] = max_book
            row["value_odds"] = str(max_odds)
            row["value_flag"] = "TRUE"
        else:
            row["value"] = max_odds / avg_other
            row["value_book"] = max_book
            row["value_odds"] = str(max_odds)
            row["value_flag"] = "FALSE"

    if not all_rows:
        print("❌ No valid data parsed.")
        return
    # with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(all_rows)

    # print(f"\n✅ Normalized data written to {OUTPUT_FILE} with {len(all_rows)} rows.")
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    normalize_and_match()
