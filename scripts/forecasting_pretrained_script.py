
# This script runs the forecasting token prediction on 26/27 open source datasets, using the given prediction lengths in the paper
# It uses the pretrained checkpoint in HF: amazon/chronos-t5-small

import pandas as pd
import numpy as np
import torch
import logging
import time
import subprocess
from src.chronos import ChronosPipeline
# from chronos import ChronosPipeline

filenames = ['australian_electricity_demand_dataset', 'car_parts_dataset_with_missing_values', 'cif_2016_dataset', 'covid_deaths_dataset', 'ERCOT_load_2004_2021Sept', 'ETT_15min', 'ETT_hourly', 'exchange_rate', 'fred_md_dataset',
            'hospital_dataset', 'm1_monthly_dataset', 'm1_quarterly_dataset', 'm1_yearly_dataset', 'm3_monthly_dataset', 'm3_quarterly_dataset', 'm3_yearly_dataset', 'm4_quarterly_dataset', 'm4_yearly_dataset', 'm5_dataset',
            'nn5_daily_dataset', 'nn5_weekly_dataset', 'tourism_monthly_dataset', 'tourism_quarterly_dataset', 'tourism_yearly_dataset', 'traffic_hourly_dataset', 'weather_dataset_new']

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# shift all NaN values in training dataset to the front of each time series
def shift_nans_to_front(row):
    non_nans = row.dropna().tolist()  # Get all non-NaN values
    nans = [np.nan] * (len(row) - len(non_nans))  # Create a list of NaNs
    return pd.Series(nans + non_nans)  # Concatenate NaNs at the front and non-NaNs at the end

file_to_length = {'australian_electricity_demand_dataset': 48, 'car_parts_dataset_with_missing_values': 12, 'cif_2016_dataset': 12, 'covid_deaths_dataset': 30, 'ERCOT_load_2004_2021Sept': 24,
        'ETT_15min': 24, 'ETT_hourly': 24, 'exchange_rate': 30, 'fred_md_dataset': 12, 'hospital_dataset': 12, 'm1_monthly_dataset': 18, 'm1_quarterly_dataset': 8, 'm1_yearly_dataset': 6,
        'm3_monthly_dataset': 18, 'm3_quarterly_dataset': 8, 'm3_yearly_dataset': 6, 'm4_quarterly_dataset': 8, 'm4_yearly_dataset': 6, 'm5_dataset': 28, 'nn5_daily_dataset': 56,
        'nn5_weekly_dataset': 8, 'tourism_monthly_dataset': 24, 'tourism_quarterly_dataset': 8, 'tourism_yearly_dataset': 4, 'traffic_hourly_dataset': 24, 'weather_dataset_new': 30
}

file_to_num_series = {'australian_electricity_demand_dataset': (5, 232224), 'car_parts_dataset_with_missing_values': (2674, 39), 'cif_2016_dataset': (72, 108), 'covid_deaths_dataset': (266, 182),
        'ERCOT_load_2004_2021Sept': (8, 154830), 'ETT_15min': (14, 69656), 'ETT_hourly': (14, 17396), 'exchange_rate': (8, 7564), 'fred_md_dataset': (107, 716), 'hospital_dataset': (767, 72),
        'm1_monthly_dataset': (617, 132), 'm1_quarterly_dataset': (203, 106), 'm1_yearly_dataset': (181, 52), 'm3_monthly_dataset': (1428, 126), 'm3_quarterly_dataset': (756, 64),
        'm3_yearly_dataset': (645, 41), 'm4_quarterly_dataset': (24000, 858), 'm4_yearly_dataset': (23000, 829), 'm5_dataset': (30490, 1918), 'nn5_daily_dataset': (111, 735), 'nn5_weekly_dataset': (111, 105),
        'tourism_monthly_dataset': (366, 309), 'tourism_quarterly_dataset': (427, 122), 'tourism_yearly_dataset': (518, 43), 'traffic_hourly_dataset': (862, 17520), 'weather_dataset_new': (3010, 65951)
}
sorted_dict = dict(sorted(file_to_num_series.items(), key=lambda item: item[1]))

pretrained_pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",                                                                                                                      # TODO: "cuda", "cpu" for CPU inference, or "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

logging.info("Starting the prediction process...")
# for file in sorted_dict:
# for file in filenames:
for file in ['weather_dataset_new', 'm4_yearly_dataset', 'm4_quarterly_dataset', 'm5_dataset']:
    logging.info(f"Processing file: {file} with prediction_length: {file_to_length[file]}")

    df = pd.read_csv(f'/chronos-forecasting-finetune/scripts/finetuning_csv/{file}_zeroshot_train.csv')                                     # TODO: change filename
    # df = df.apply(shift_nans_to_front, axis=1)                                                                                            # TODO: zeroshot datasets already prepend with NaNs
    logging.info(f"Read in {file} with shape {df.shape}")
    start_time = time.time() # Record start time
    try:
        new_forecasting = pretrained_pipeline.predict(
            context=torch.tensor(df.values), 
            prediction_length=file_to_length[file], 
            num_samples=20,                                                                                                                 # TODO: change as necessary
            temperature=1.0,                                                                                                                # TODO: change as necessary
            top_k=50,                                                                                                                        # TODO: change as necessary
            top_p=1.0,                                                                                                                        # TODO: change as necessary
        )
        logging.debug(f"Prediction completed for: {file}")

        # Convert predictions to DataFrame and concatenate
        df = pd.DataFrame(np.quantile(new_forecasting.numpy(), 0.5, axis=1))                                                                                # TODO: choose method of creating df
        df.to_csv(f'/chronos-forecasting-finetune/scripts/predictions_pretrained/{file}_zeroshot_pred.csv', index=False)                    # TODO: choose filename
        subprocess.run(['aws', 's3', 'cp', f'./scripts/predictions_pretrained/{file}_zeroshot_pred.csv',                                    # TODO: choose filename
                        f's3://mosaicml-internal-checkpoints-shared/kushal/chronos-zeroshot-forecast-quantile/{file}_zeroshot_pred.csv'],     # TODO: choose filename
                        capture_output=True, text=True) # save to s3 cluster
        logging.info(f"Saved pretrained predictions for {file}")
    except Exception as e:
        logging.error(f"ERROR during prediction for file {file}: {e}")
        continue

    end_time = time.time() # Record end time
    logging.info(f"Time taken for prediction for {file}: {end_time - start_time:.2f} seconds")

# subprocess.run(['rm', '-rf', '/chronos-forecasting-finetune/scripts/finetuning_csv'], capture_output=True, text=True)
