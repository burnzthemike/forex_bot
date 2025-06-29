import os
import pandas as pd

def convert_mt4_csv(input_path, output_path):
    # Read file without headers
    df = pd.read_csv(input_path, header=None)
    
    # Set correct headers
    df.columns = ["date", "time", "open", "high", "low", "close", "volume"]
    
    # Combine date and time
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M")
    
    # Reorder and drop extra columns
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    
    # Save clean file
    df.to_csv(output_path, index=False)
    print(f"[✓] Converted and saved: {output_path}")

if __name__ == "__main__":
    data_folder = "mt4_data_raw"
    output_folder = "mt4_data_cleaned"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            input_file = os.path.join(data_folder, file)
            output_file = os.path.join(output_folder, f"cleaned_{file}")
            convert_mt4_csv(input_file, output_file)
