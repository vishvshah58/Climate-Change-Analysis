import pandas as pd
import os

def clean_data(datasets):
    print("\n" + "="*45)
    print("\033[1m   PHASE 2: DATA CLEANING STARTED\033[0m")
    print("="*45)

    # We will focus on the main temperature change dataset
    key = 'reduced_Environment_Temperature_change_E_All_Data_NOFLAG'
    df_temp = datasets.get(key)

    if df_temp is not None:
        print(f"Cleaning {key}...")

        # 1. Filter for 'Temperature change' to ignore 'Standard Deviation' rows
        df_temp = df_temp[df_temp['Element'] == 'Temperature change']

        # 2. Convert 'Wide' format to 'Long' format
        # This takes years (Y1961, Y1962...) from columns and puts them in rows
        year_cols = [col for col in df_temp.columns if col.startswith('Y') and col[1:].isdigit()]
        id_cols = ['Area', 'Months', 'Element']

        df_melted = df_temp.melt(id_vars=id_cols, value_vars=year_cols,
                                 var_name='Year', value_name='Temp_Change')

        # 3. Format 'Year' column
        df_melted['Year'] = df_melted['Year'].str.replace('Y', '').astype(int)

        # 4. Handle Missing Values
        # We use interpolation (filling the gap based on the numbers before and after)
        before_nulls = df_melted['Temp_Change'].isnull().sum()
        df_melted['Temp_Change'] = df_melted['Temp_Change'].interpolate()
        after_nulls = df_melted['Temp_Change'].isnull().sum()

        print(f"✅ Handled {before_nulls - after_nulls} missing values.")
        print(f"✅ Final Cleaned Shape: {df_melted.shape}")

        # 5. Create 'data/processed' folder and save
        output_dir = "data/processed"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, "climate_cleaned.csv")
        df_melted.to_csv(output_path, index=False)
        print(f"💾 Cleaned data saved to: {output_path}")
    else:
        print(f"❌ ERROR: {key} not found. Please run Phase 1 first.")

    print("\n" + "="*45)
    print("   PHASE 2 COMPLETE")
    print("="*45)

if __name__ == "__main__":
    # This part runs Phase 1 automatically to get the data for Phase 2
    try:
        # run_phase_1 is defined in an earlier cell in this notebook
        raw_data = run_phase_1()
        if raw_data:
            clean_data(raw_data)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
