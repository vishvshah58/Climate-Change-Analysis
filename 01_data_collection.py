import pandas as pd
import os

def run_phase_1():
    print("\n" + "="*45)
    print("\033[1m   PHASE 1: DATASET ACQUISITION STARTED\033[0m")
    print("="*45)

    datasets = {}
    
    # The list of files you actually uploaded
    uploaded_files = [
        "reduced_Environment_Temperature_change_E_All_Data_NOFLAG.csv",
        "reduced_FAOSTAT_data_1-10-2022.csv",
        "reduced_FAOSTAT_data_11-24-2020.csv",
        "reduced_FAOSTAT_data_en_11-1-2024.csv",
        "reduced_GlobalWeatherRepository.csv",
        "reduced_GlobalLandTemperaturesByCountry.csv",
        "reduced_GlobalLandTemperaturesByMajorCity.csv",
        "reduced_GlobalLandTemperaturesByState.csv",
        "reduced_GlobalTemperatures.csv"
    ]

    # Search paths in order of priority
    search_locations = [".", "data/raw", "../data/raw"]

    for file in uploaded_files:
        found_path = None
        for loc in search_locations:
            check_path = os.path.join(loc, file)
            if os.path.exists(check_path):
                found_path = check_path
                break
        
        print(f"\n\033[1m=== DATASET: {file} ===\033[0m")
        if found_path:
            try:
                # low_memory=False and latin-1 encoding handle global datasets robustly
                df = pd.read_csv(found_path, encoding='latin-1', low_memory=False)
                key_name = file.replace('.csv', '')
                datasets[key_name] = df
                print(f"✅ Loaded successfully from: {found_path}")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:5]}...")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        else:
            # We don't error out here, just warn, so the rest can load
            print(f"⚠️  Skipping: File not found in any standard location.")

    print("\n" + "="*45)
    print(f"   PHASE 1 COMPLETE: {len(datasets)} datasets ready.")
    print("="*45)
    return datasets

if __name__ == "__main__":
    raw_datasets = run_phase_1()
