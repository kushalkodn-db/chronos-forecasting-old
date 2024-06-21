import numpy as np
import pandas as pd
import torch
import logging
import time
import gluonts
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from src.chronos import ChronosPipeline
from collections import defaultdict


filenames = ['australian_electricity_demand', 'car_parts_without_missing', 'cif_2016', 'covid_deaths', 'dominick', 'ercot', 'ett_small_15min', 'ett_small_1h', 'exchange_rate', 
    'fred_md', 'hospital', 'm1_monthly', 'm1_quarterly', 'm1_yearly', 'm3_monthly', 'm3_quarterly', 'm3_yearly', 'm4_quarterly', 'm4_yearly', 'm5', 
    'nn5_daily_without_missing', 'nn5_weekly', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly', 'traffic', 'weather']
file_horizon_mapping = {'australian_electricity_demand': 48, 'car_parts_without_missing': 12, 'cif_2016': 12, 'covid_deaths': 30, 'dominick': 8, 'ercot': 24, 'ett_small_15min': 24, 'ett_small_1h': 24, 
    'exchange_rate_nips': 30, 'fred_md': 12, 'hospital': 12, 'm1_monthly': 18, 'm1_quarterly': 8, 'm1_yearly': 6, 'm3_monthly': 18, 'm3_quarterly': 8, 'm3_yearly': 6, 'm4_quarterly': 8, 'm4_yearly': 6, 'm5': 28, 
    'nn5_daily_without_missing': 56, 'nn5_weekly': 8, 'tourism_monthly': 24, 'tourism_quarterly': 8, 'tourism_yearly': 4, 'traffic_nips': 24, 'weather': 30}

file_to_num_series = {'australian_electricity_demand': (5, 232224), 'car_parts_without_missing': (2674, 39), 'cif_2016': (72, 108), 'covid_deaths': (266, 182), 'dominick': (100014, 399), 
    'ercot': (8, 154830), 'ett_small_15min': (14, 69656), 'ett_small_1h': (14, 17396), 'exchange_rate': (8, 7564), 'fred_md': (107, 716), 'hospital': (767, 72),
    'm1_monthly': (617, 132), 'm1_quarterly': (203, 106), 'm1_yearly': (181, 52), 'm3_monthly': (1428, 126), 'm3_quarterly': (756, 64),
    'm3_yearly': (645, 41), 'm4_quarterly': (24000, 858), 'm4_yearly': (23000, 829), 'm5': (30490, 1918), 'nn5_daily_without_missing': (111, 735), 'nn5_weekly': (111, 105),
    'tourism_monthly': (366, 309), 'tourism_quarterly': (427, 122), 'tourism_yearly': (518, 43), 'traffic': (862, 17520), 'weather': (3010, 65951)
}
sorted_dict = dict(sorted(file_to_num_series.items(), key=lambda item: item[1]))

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

all_metrics = pd.DataFrame(columns=['filename', 'MASE[0.5]', 'mean_weighted_sum_quantile_loss'])

batch_size = 32
num_samples = 20
# for filename in filenames:
# for filename in sorted_dict: # 'hospital', 'm5' DO NOT WORK
for filename in ['exchange_rate_nips', 'traffic_nips']:
    # Load dataset
    prediction_length = file_horizon_mapping[filename]
    try:
        curr_file_results = pd.DataFrame(columns=['MASE[0.5]', 'mean_weighted_sum_quantile_loss'])
        dataset = get_dataset(dataset_name=filename)
        # dataset = get_dataset(dataset_name=filename, prediction_length=prediction_length)
        logging.debug(f"Loaded in {filename} with prediction_length: {prediction_length}")

        # Load Chronos
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )

        for _ in range(3):
            # Split dataset for evaluation
            _, test_template = split(dataset.test, offset=-prediction_length)
            test_data = test_template.generate_instances(prediction_length)
            logging.debug(f"Split dataset for testing: {filename}")

            start_time = time.time() # Record start time
            # Generate forecast samples
            forecast_samples = []
            for batch in tqdm(batcher(test_data.input, batch_size=32)):
                context = [torch.tensor(entry["target"]) for entry in batch]
                forecast_samples.append(
                    pipeline.predict(
                        context,
                        prediction_length=prediction_length,
                        num_samples=num_samples,
                    ).numpy()
                )
            forecast_samples = np.concatenate(forecast_samples)
            logging.debug(f"Generated forecasts for: {filename}")

            # Convert forecast samples into gluonts SampleForecast objects
            sample_forecasts = []
            for item, ts in zip(forecast_samples, test_data.input):
                forecast_start_date = ts["start"] + len(ts["target"])
                sample_forecasts.append(
                    SampleForecast(samples=item, start_date=forecast_start_date)
                )

            # Evaluate
            metrics_df = evaluate_forecasts(
                sample_forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                ],
            )
            end_time = time.time() # Record end time
            logging.info(f"Time taken for forecasting {filename}: {end_time - start_time:.2f} seconds")

            curr_file_results = pd.concat([curr_file_results, metrics_df], ignore_index=True)
            # curr_row = metrics_df.iloc[0].to_dict()
            # curr_row["filename"] = filename

        mean_values = curr_file_results.mean(axis=0)
        mean_df =pd.DataFrame({
            'filename': [filename], 
            'MASE[0.5]': [mean_values['MASE[0.5]']], 
            'mean_weighted_sum_quantile_loss': [mean_values['mean_weighted_sum_quantile_loss']], 
        })
        logging.info(f"Evaluation metrics for {filename} :: {mean_df}")
        all_metrics = pd.concat([all_metrics, mean_df], ignore_index=True)

    except Exception as e:
        logging.debug(f"ERROR with {filename}: {e}")


logging.info(f"All evaluation metrics in large DF :: {all_metrics}")
print(all_metrics)
