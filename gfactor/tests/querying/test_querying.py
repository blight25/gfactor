# Standard library
import unittest
import datetime as dt
import random

from gfactor.querying.LISIRDQuerying import LISIRDRetriever
from gfactor.querying.NISTQuerying import NISTRetriever
from gfactor.main.gfactorsolar import SolarSpectrum

from pathlib import Path

import math

import numpy as np

import requests
import time

from tqdm import tqdm

from datetime import datetime as dt
from datetime import date, timedelta


class TestLISIRDQuerying(unittest.TestCase):

    NUM_SAMPLES = 25
    retriever = LISIRDRetriever()
    irradiance_names = retriever.irradiance_names
    other_names = retriever.other_names
    error_loc = "data/errors"

    def test_irradiance_legal(self):

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        print("\nIrradiance Querying Test")
        print(f"Number of samples per dataset: {TestLISIRDQuerying.NUM_SAMPLES}")
 
        for name in TestLISIRDQuerying.irradiance_names:

            # Create error file for recording problematic dates
            error_dir = Path(TestLISIRDQuerying.error_loc + "/" + name)
            error_dir.mkdir(parents=True, exist_ok=True)
            error_file = error_dir / "problem_dates_test.txt"
            with open (error_file, "w") as problem_file: 
                problem_file.write("PROBLEM DATES\n\n")
            print(f"\nError log to be saved in directory {error_dir}")


            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = TestLISIRDQuerying.irradiance_names[name]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = name + "_" + subset if subset else name
                min_date = max(min_date, TestLISIRDQuerying.retriever.irradiance_datasets[dataset]['min_date'])
                max_date = min(max_date, TestLISIRDQuerying.retriever.irradiance_datasets[dataset]['max_date'])
            
            print(f"\nDataset {name}: minimum date of {min_date}, maximum date of {max_date}")

            total_days = (max_date - min_date).days
            interval = math.floor(total_days / TestLISIRDQuerying.NUM_SAMPLES)
            progress_bar = tqdm(total=total_days, desc=name)
            query_date = min_date
            
            while query_date <= max_date:
                
                # Query
                for subset in subsets:
                    try:
                        df = TestLISIRDQuerying.retriever.retrieve(dataset=name, subset=subset,
                                                                   date = query_date.strftime("%Y-%m-%d"))
                           
                    except requests.exceptions.RequestException: # sometimes SSL behaves weird, or the connection gets closed suddenly/times out: try one more time
                        time.sleep(5) # First, give server some time to breath
                        df = TestLISIRDQuerying.retriever.retrieve(dataset=name, subset=subset,
                                                                   date = query_date.strftime("%Y-%m-%d"))
                    
                    except ValueError: # There was no data available
                        with open (error_file, "a") as problem_file:
                            problem_file.write(f"Value Error: {query_date}\n\n") # Keep tabs on problematic dates
                        query_date += timedelta(interval)
                        progress_bar.update(1)
                        continue
                        
                    # For available dataframes, ensure data is as expected
                    self.assertIsNotNone(df) # Dataframe exists
                    for col in required_cols:
                        self.assertIn(col, df.columns) # required columns are present
                        dtype = df[col].dtype
                        python_dtype = float if np.issubdtype(dtype, np.floating) else None
                        self.assertEqual(python_dtype, float) # All dtypes are float

                # Increment
                query_date += timedelta(interval)
                progress_bar.update(interval)