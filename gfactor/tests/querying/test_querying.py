# Standard library
import unittest
import json

from gfactor.querying.LISIRDQuerying import LISIRDRetriever, NoDataAvailableError
from gfactor.querying.NISTQuerying import NISTRetriever

import shutil
from pathlib import Path

import math
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import date, timedelta


class TestLISIRDQuerying(unittest.TestCase):

    NUM_SAMPLES = 30 # Total samples to fetch per dataset
    FAILED_QUERY_TOLERANCE = 3 # Failed dates
    TIMEOUT = 20 # seconds waiting for response

    # Get the absolute path to the current file (LISIRDQuerying.py)
    current_file = Path(__file__).resolve()

    # Get the package root
    package_root = current_file.parents[2]  # gfactor/

    # Build path to target directory
    TEST_DIR = (package_root / "tests" / "querying" / "spectra").as_posix()

    retriever = LISIRDRetriever()
    irradiance_identifiers = retriever.irradiance_identifiers
    other_identifiers = retriever.other_identifiers
    STATUS_LOG = "./gfactor/tests/querying/spectral_test_log.json"

    def _get_working_identifiers(self):
        """
        Returns a dictionary of dataset identifiers and their working subsets based on the status log.
        Only includes those marked as 'working'.
        """
        if Path(self.STATUS_LOG).exists():
            with open(self.STATUS_LOG, "r") as f:
                status_log = json.load(f)
        else:
            return {**self.irradiance_identifiers, **self.other_identifiers}

        working = {}
        for identifier, value in status_log.items():
            if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values()):
                # Has subsets
                good_subsets = [subset for subset, subval in value.items() if subval.get("status") == "working"]
                if good_subsets:
                    working[identifier] = good_subsets
            else:
                if value.get("status") == "working":
                    working[identifier] = [None]
        return working
    

    def test_retrieve(self):
        """
        Tests LISIRDRetriever.retrieve for all working datasets and subsets, checking data validity, error handling, and timeout logging.
        """

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        print("\nIrradiance Querying Test")
        print(f"Number of samples per dataset: {TestLISIRDQuerying.NUM_SAMPLES}")
 
        identifiers = {**TestLISIRDQuerying.retriever.irradiance_identifiers, **TestLISIRDQuerying.retriever.other_identifiers}
        
        # Load or initialize status log
        if Path(self.STATUS_LOG).exists():
            with open(self.STATUS_LOG, "r") as f:
                status_log = json.load(f)
        else:
            status_log = {}
        
        # Mark all as working initially
        for identifier in identifiers:
            subsets = identifiers[identifier]
            if subsets[-1]:
                status_log[identifier] = {subset: {"status": "working", "bad dates": []} for subset in subsets}
            else:
                status_log[identifier] = {"status": "working", "bad dates": []}
        with open(self.STATUS_LOG, "w") as f:
            json.dump(status_log, f, indent=2)

        # Dataset category (affects DataFrame type checking)
        for identifier in identifiers:
            extended_check = False
            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
                extended_check = True

            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            print(f"\nDataset {identifier}: minimum date of {min_date}, maximum date of {max_date}")

            # Variables
            total_days = (max_date - min_date).days
            interval = math.floor(total_days / TestLISIRDQuerying.NUM_SAMPLES)
            progress_bar = tqdm(total=total_days, desc=identifier)
            query_date = min_date
            
            # Flags to prevent/measure any error buildup
            failed_queries = np.zeros(len(subsets), dtype=int)
            timeout_flags = np.zeros(len(subsets), dtype=int)

            while query_date < max_date:
                # Query
                for idx, subset in enumerate(subsets):
                        if timeout_flags[idx] < 2:
                            try:
                                df = TestLISIRDQuerying.retriever.retrieve(
                                    dataset=identifier,
                                    subset=subset,
                                    query_date=query_date.strftime("%Y-%m-%d"),
                                    timeout=self.TIMEOUT
                                )
                                timeout_flags[idx] = 0
                            except RuntimeError:
                                timeout_flags[idx] += 1
                                if timeout_flags[idx] == 2:
                                    if subset is not None:
                                        print(f"\nWARNING: Dataset '{identifier + "_" + subset}' reported back-to-back timeouts of over {self.TIMEOUT} seconds - it may no longer be available through the LISIRD API.\n")
                                        status_log[identifier][subset]["status"] = "timeout"
                                        status_log[identifier][subset]["timeout"] = self.TIMEOUT

                                    else:
                                        print(f"\nWARNING: Dataset '{identifier}' reported back-to-back timeouts of over {self.TIMEOUT} seconds - it may no longer be available through the LISIRD API.\n")
                                        status_log[identifier]["status"] = "timeout"
                                        status_log[identifier]["timeout"] = self.TIMEOUT
                                else:
                                    if subset is not None:
                                        status_log[identifier][subset]["bad dates"].append(query_date.strftime("%Y-%m-%d"))
                                    else:
                                        status_log[identifier]["bad dates"].append(query_date.strftime("%Y-%m-%d"))
                            
                            except NoDataAvailableError:
                                failed_queries[idx] += 1
                                if subset is not None:
                                    status_log[identifier][subset]["bad dates"].append(query_date.strftime("%Y-%m-%d"))
                                else:
                                    status_log[identifier]["bad dates"].append(query_date.strftime("%Y-%m-%d"))
                                
                        # Break out of both loops to advance to next dataset
                        if np.sum(timeout_flags) == 2*len(subsets):
                            break
                        
                        # For available dataframes, ensure data is as expected
                        self.assertIsNotNone(df)
                        self.assertGreater(len(df.values), 0)
                        if extended_check:
                            for col in required_cols:
                                self.assertIn(col, df.columns)
                                dtype = df[col].dtype
                                python_dtype = float if np.issubdtype(dtype, np.floating) else None
                                self.assertEqual(python_dtype, float)
                
                # Timeout - break out of both loops to advance to next dataset
                if np.sum(timeout_flags) == 2*len(subsets):
                    break
  
                # Update progress
                query_date += timedelta(interval)
                progress_bar.update(interval)
                continue
            
            # Wrap up dataset
            progress_bar.close()
            with open(self.STATUS_LOG, "w") as f:
                json.dump(status_log, f, indent=2)
            
            for failed_query in failed_queries:
                self.assertLess(failed_query, TestLISIRDQuerying.FAILED_QUERY_TOLERANCE)
    

    def test_dataset_validity(self):
        """
        Asserts that all datasets and subsets in the status log have status 'working'.
        """
        if Path(self.STATUS_LOG).exists():
            with open(self.STATUS_LOG, "r") as f:
                status_log = json.load(f)
        else:
            self.fail(f"Status log file {self.STATUS_LOG} does not exist.")

        for dataset, value in status_log.items():
            # If value is a dict of subsets
            if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values()):
                for subset, subval in value.items():
                    status = subval.get("status", None)
                    self.assertEqual(status, "working", f"Dataset '{dataset}', subset '{subset}' has status '{status}' (expected 'working')")
            else:
                status = value.get("status", None)
                self.assertEqual(status, "working", f"Dataset '{dataset}' has status '{status}' (expected 'working')")  


    def test_retrieve_invalid_dataset(self):
        """
        Tests that retrieve raises TypeError for illegal dataset types and ValueError for invalid dataset names.
        """

        illegal_types = [None, 5, 100.7, False]
        illegal_dataset = 'test'

        for illegal_type in illegal_types:
            with self.assertRaises(TypeError):
                TestLISIRDQuerying.retriever.retrieve(dataset=illegal_type, query_date="2012-01-01", subset=None, timeout=self.TIMEOUT)
        
        with self.assertRaises(ValueError):
            TestLISIRDQuerying.retriever.retrieve(dataset=illegal_dataset, query_date="2012-01-01", subset=None, timeout=self.TIMEOUT)

    
    def test_retrieve_invalid_subset(self):
        """
        Tests that retrieve raises errors for invalid subset types or values, and for missing/extra subset arguments.
        """

        illegal_types = [5, 100.7, False]
        illegal_subset = 'test'

        identifiers = self._get_working_identifiers()
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            subsets = identifiers[identifier]

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Variables
            query_date = min_date.strftime("%Y-%m-%d")
            subset = subsets[-1]

            # Subset exists, failure to specify should throw error
            if subset is not None:
                with self.assertRaises(ValueError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=None, timeout=self.TIMEOUT)
            
            # No subset exists, attempting to specify should throw error
            else:
                with self.assertRaises(ValueError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=illegal_subset, timeout=self.TIMEOUT)
            
            # Invalid typing
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=illegal_type, timeout=self.TIMEOUT)
    

    def test_retrieve_invalid_date(self):
        """
        Tests that retrieve raises errors for out-of-range or invalid date arguments and types.
        """

        illegal_types = [None, 5, 100.7, False]
        
        identifiers = self._get_working_identifiers()
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Variables
            early_date = min_date - timedelta(days=10)
            early_date = early_date.strftime("%Y-%m-%d")
            late_date = max_date + timedelta(days=10)
            late_date = late_date.strftime("%Y-%m-%d")

            # Below minimum and above maximum
            for query_date in (early_date, late_date):
                with self.assertRaises(ValueError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=query_date, subset=subset, timeout=self.TIMEOUT)

            # Invalid typing
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    TestLISIRDQuerying.retriever.retrieve(dataset=identifier, query_date=illegal_type, subset=subset, timeout=self.TIMEOUT)
    

    def test_extract(self):
        """
        Tests LISIRDRetriever.extract for all working datasets, checking file creation and data validity for all subsets and dates.
        """

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        # Load or initialize status log
        if Path(self.STATUS_LOG).exists():
            with open(self.STATUS_LOG, "r") as f:
                status_log = json.load(f)
        else:
            raise FileNotFoundError(f"JSON status log for working datasets not found: please run 'test_retrieve' before any additional query testing is performed")
        
        test_dir = Path(TestLISIRDQuerying.TEST_DIR)
        # If a previous test directory exists, remove it and its associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))

        # Dataset names and subsets
        identifiers = self._get_working_identifiers()
        all_identifiers = {**self.irradiance_identifiers, **self.other_identifiers}

        for identifier in identifiers:

            # Ensure that this dataset and ALL of its subsets (if they exist) are operational before testing
            subsets = identifiers[identifier]
            if identifier not in all_identifiers or not len(subsets) == len(all_identifiers[identifier]):
                            continue
            
            # Irradiance or other
            extended_check = False
            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
                extended_check = True

            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            

            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Find bad dates from retrieve testing
            if subsets[-1]:
                bad_dates = set()
                for subset in subsets:
                    bad_dates.update(status_log[identifier][subset]["bad dates"])
            else:
                bad_dates = set(status_log[identifier]["bad dates"])
            
            # Avoid setting start to a faulty date
            start_date = min_date
            while True:
                if start_date.strftime("%Y-%m-%d") in bad_dates:
                    start_date += timedelta(7)
                else:
                    break
            
            # Avoid setting end to a faulty date
            end_date = max_date
            while True:
                if end_date.strftime("%Y-%m-%d") in bad_dates:
                    end_date -= timedelta(7)
                else:
                    break
            
            # Variables
            interval = (end_date - start_date).days
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")

            # Extract
            TestLISIRDQuerying.retriever.extract(
                dataset=identifier,
                subset=None,
                interval=interval,
                save_dir=TestLISIRDQuerying.TEST_DIR,
                start_date=start_date,
                end_date=end_date,
                overwrite=True,
                timeout=self.TIMEOUT
            )
            
            # Check that files exist
            local_dir = Path(TestLISIRDQuerying.TEST_DIR + "/" + identifier)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Subsets
            for subset in subsets:
                for query_date in (start_date, end_date):
                    file = identifier + "_" + subset + ".pickle" if subset else identifier + ".pickle"
                    file = local_dir / query_date / file
                    df = pd.read_pickle(file)

                    # Ensure data is as expected
                    self.assertIsNotNone(df) # Dataframe exists
                    self.assertGreater(len(df.values), 0) # Contains actual data

                    if extended_check: # For irradiance datasets only
                        for col in required_cols:
                            self.assertIn(col, df.columns) # required columns are present
                            dtype = df[col].dtype
                            python_dtype = float if np.issubdtype(dtype, np.floating) else None
                            self.assertEqual(python_dtype, float) # All dtypes are float
            

        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))
            

    def test_extract_specific_subset(self):
        """
        Tests extract for datasets with multiple subsets, ensuring only the specified subset is extracted and files are correct.
        """
        
        test_dir = Path(TestLISIRDQuerying.TEST_DIR)
        # If a previous test directory exists, remove it and its associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))

        required_cols = ["wavelength (nm)", "irradiance (W/m^2/nm)"]

        identifiers = self._get_working_identifiers()

        for identifier in identifiers:

            subsets = identifiers[identifier]
            
            # Case where subsets = [None] is covered by the base test
            if len(subsets) > 1:
            
                extended_check = False
                if identifier in TestLISIRDQuerying.irradiance_identifiers:
                    datasets = TestLISIRDQuerying.retriever.irradiance_datasets
                    extended_check = True

                elif identifier in TestLISIRDQuerying.other_identifiers:
                    datasets = TestLISIRDQuerying.retriever.other_datasets
                
                # Date initialization - these will always be replaced once actual dates are located
                min_date = date(year=1600, month=1, day=1)
                max_date = date(year=2100, month=1, day=1)

                # Only take the date range that works for all subsets
                for subset in subsets:
                    dataset = identifier + "_" + subset if subset else identifier
                    min_date = max(min_date, datasets[dataset]['min_date'])
                    max_date = min(max_date, datasets[dataset]['max_date'])
                
                # Variables
                interval = (max_date - min_date).days - 1
                start_date = min_date.strftime("%Y-%m-%d")
                end_date = min_date + timedelta(days=interval)
                end_date = end_date.strftime("%Y-%m-%d")

                # Use first subset for testing
                TestLISIRDQuerying.retriever.extract(
                    dataset=identifier,
                    subset=subsets[0],
                    interval=interval,
                    save_dir=TestLISIRDQuerying.TEST_DIR,
                    start_date=start_date,
                    end_date=end_date,
                    overwrite=True,
                    timeout=self.TIMEOUT
                )
                
                # Check that files exist
                local_dir = Path(TestLISIRDQuerying.TEST_DIR + "/" + identifier)
                local_dir.mkdir(parents=True, exist_ok=True)

                # File testing
                for query_date in (start_date, end_date):

                    # File for first subset should exist and be usable
                    file = identifier + "_" + subsets[0] + ".pickle"
                    file = local_dir / query_date / file
                    df = pd.read_pickle(file)

                    # For available dataframes, ensure data is as expected
                    self.assertIsNotNone(df) # Dataframe exists
                    self.assertGreater(len(df.values), 0) # Contains actual data

                    if extended_check: # For irradiance datasets only
                        for col in required_cols:
                            self.assertIn(col, df.columns) # required columns are present
                            dtype = df[col].dtype
                            python_dtype = float if np.issubdtype(dtype, np.floating) else None
                            self.assertEqual(python_dtype, float) # All dtypes are float
                    
                    # File for second subset shouldn't exist
                    invalid_file = file = identifier + "_" + subsets[1] + ".pickle"
                    invalid_file = local_dir / query_date / invalid_file
                    with self.assertRaises(FileNotFoundError):
                        df = pd.read_pickle(file)
        
        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))
    
    
    def test_extract_invalid_dataset(self):
        """
        Tests that extract raises TypeError for illegal dataset types and ValueError for invalid dataset names.
        """
        test_dir = Path(TestLISIRDQuerying.TEST_DIR)
        # If a previous test directory exists, remove it and its associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))

        illegal_types = [None, 5, 100.7, False]
        illegal_dataset = 'test'

        for illegal_type in illegal_types:
            with self.assertRaises(TypeError):
                TestLISIRDQuerying.retriever.extract(dataset=illegal_type, start_date=None,
                                      end_date=None, interval=1,
                                      save_dir=self.TEST_DIR,
                                      subset=None, timeout=self.TIMEOUT)

            # Remove test directory and associated files
            if test_dir.exists() and test_dir.is_dir():
                shutil.rmtree(Path(self.TEST_DIR))
            
        with self.assertRaises(ValueError):
            TestLISIRDQuerying.retriever.extract(dataset=illegal_dataset, start_date=None,
                                   end_date=None, interval=1,
                                   save_dir=self.TEST_DIR,
                                   subset=None, timeout=self.TIMEOUT)

            # Remove test directory and associated files
            if test_dir.exists() and test_dir.is_dir():
                shutil.rmtree(Path(self.TEST_DIR))
    

    def test_extract_invalid_subset(self):
        """
        Tests that extract raises errors for invalid subset types or values for all working datasets.
        """

        illegal_types = [5, 100.7, False]
        illegal_subset = 'test'

        identifiers = self._get_working_identifiers()
        
        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Type Checking 
            for illegal_type in illegal_types:
                with self.assertRaises(TypeError):
                    TestLISIRDQuerying.retriever.extract(dataset=identifier, subset=illegal_type,
                                          interval=1, save_dir=self.TEST_DIR,
                                          start_date=min_date,
                                          end_date=max_date,
                                          overwrite=True,
                                          timeout=self.TIMEOUT)
            
            # Illegal Value
            subset = subsets[-1]        
            if subset is not None:
                with self.assertRaises(ValueError):
                    TestLISIRDQuerying.retriever.extract(dataset=identifier, subset=illegal_subset,
                                          interval=1, save_dir=self.TEST_DIR,
                                          start_date=min_date,
                                          end_date=max_date,
                                          overwrite=True,
                                          timeout=self.TIMEOUT)
                    

    def test_extract_invalid_date(self):
        """
        Tests that extract raises errors for out-of-range or invalid start/end date arguments and types.
        """
        test_dir = Path(TestLISIRDQuerying.TEST_DIR)
        # If a previous test directory exists, remove it and its associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(TestLISIRDQuerying.TEST_DIR))

        illegal_types = [None, 5, 100.7, False]
        
        identifiers = self._get_working_identifiers()

        for identifier in identifiers:

            if identifier in TestLISIRDQuerying.irradiance_identifiers:
                datasets = TestLISIRDQuerying.retriever.irradiance_datasets
        
            elif identifier in TestLISIRDQuerying.other_identifiers:
                datasets = TestLISIRDQuerying.retriever.other_datasets
            
            # Date initialization - these will always be replaced once actual dates are located
            min_date = date(year=1600, month=1, day=1)
            max_date = date(year=2100, month=1, day=1)

            # Only take the date range that works for all subsets
            subsets = identifiers[identifier]
            for subset in subsets:
                dataset = identifier + "_" + subset if subset else identifier
                min_date = max(min_date, datasets[dataset]['min_date'])
                max_date = min(max_date, datasets[dataset]['max_date'])
            
            # Variables
            early_date = min_date - timedelta(days=10)
            early_date = early_date.strftime("%Y-%m-%d")
            late_date = max_date + timedelta(days=10)
            late_date = late_date.strftime("%Y-%m-%d")

            # Below minimum and above maximum
            for begin_date in (None, early_date):
                for finish_date in (None, late_date):
                    if begin_date is None and finish_date is None:
                        continue
                    with self.assertRaises(ValueError):
                        TestLISIRDQuerying.retriever.extract(dataset=identifier, start_date=begin_date,
                                              end_date=finish_date, interval=1,
                                              save_dir=self.TEST_DIR,
                                              subset=None, timeout=self.TIMEOUT)

            # Type checking
            for i in range(len(illegal_types)):
                for j in range(len(illegal_types)):
                    if illegal_types[i] is None and illegal_types[j] is None:
                        continue
                    with self.assertRaises(TypeError):
                        TestLISIRDQuerying.retriever.extract(dataset=identifier, start_date=illegal_types[i],
                                              end_date=illegal_types[j], interval=1,
                                              save_dir=self.TEST_DIR,
                                              subset=None, timeout=self.TIMEOUT)
            
        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(self.TEST_DIR))



class TestNISTQuerying(unittest.TestCase):

    NUM_SAMPLES = 3 # Atomic samples in extract

    # Get the absolute path to the current file (LISIRDQuerying.py)
    current_file = Path(__file__).resolve()

    # Get the package root (assuming this file is always in gfactor/querying/)
    package_root = current_file.parents[2]  # gfactor/

    # Build path to target directory
    TEST_DIR = (package_root / "tests" / "querying" / "atomic").as_posix()

    STATUS_LOG = "./gfactor/tests/querying/atomic_test_log.json"
    required_cols = {"obs_wl(A)" : float, 
                    "fik" : float, 
                    "term_k" : str,  
                    "conf_k": str, 
                    "Ek(eV)": float,
                    "J_k": float,
                    "term_i": str, 
                    "conf_i": str, 
                    "Ei(eV)": float,
                    "J_i": float,
                    "Acc" : float, 
                    "Aki(s^-1)" : float, 
                    "element" : str, 
                    "sp_num": float}
    
    optional_cols = {"unc_obs_wl": float}

    elements = [
                'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ar', 'Ca', 
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
                'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
                'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Th', 'Np', 'U', 'Am', 'Pu', 'Cm', 
                'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Bh', 'Sg', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 
                'Lv', 'Ts', 'Og'
                ]

    # Known to fail: 'U' yields ValueError, all other queries proceed but produce invalid DataFrames
    problem_elements = {'Mt', 'At', 'Fm', 'U', 'Rf', 'Cf', 'Pa', 'No', 'Sg', 'Am', 'Pr', 'Nb', 
                        'Lr', 'Rn', 'Lv', 'Zr', 'Os', 'Hs', 'Og', 'Mc', 'Cm', 'Fl', 'Cn', 'Db', 
                        'Bk', 'Re', 'Rg', 'Po', 'Tb', 'Ds', 'Es', 'Th', 'Pm', 'Se', 'Bh', 'Nh', 
                        'Ts', 'Md', 'Pu', 'Np'}

    retriever = NISTRetriever()


    def _map_numpy_dtype_to_python(self, dtype):
        """
        Maps a numpy/pandas dtype to the corresponding Python type (float, int, str, or None).
        Handles pandas extension types like string[python].
        """

        if pd.api.types.is_string_dtype(dtype):
            return str
        elif np.issubdtype(dtype, np.floating):
            return float
        elif np.issubdtype(dtype, np.integer):
            return int
        else:
            return None


    def test_retrieve(self):
        """
        Tests NISTRetriever.retrieve for a random subset of elements, checking data validity and error handling for problematic elements.
        """

        # Loop through random subset of elements (for manageable runtime)
        safe_elements = [el for el in TestNISTQuerying.elements if el not in TestNISTQuerying.problem_elements]
        elements = random.sample(safe_elements, k=TestNISTQuerying.NUM_SAMPLES)
        problem_elements = random.sample(sorted(TestNISTQuerying.problem_elements), k=2)
        
        # Safe Loop
        print("\n\nValidate data content of standard element queries:")
        progress_bar = tqdm(total=len(elements), desc="Elements")
        for el in elements:
            for ionization in (True, False):
                results = TestNISTQuerying.retriever.retrieve(elements=[el], ionized=ionization)
                df = results[el]
                for col in TestNISTQuerying.required_cols:
                    self.assertIn(col, df.columns)
                    dtype = df[col].dtype
                    python_type = self._map_numpy_dtype_to_python(dtype)
                    self.assertIsNotNone(python_type)
                    self.assertEqual(python_type, TestNISTQuerying.required_cols[col])
                for col in TestNISTQuerying.optional_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        python_type = self._map_numpy_dtype_to_python(dtype)
                        self.assertIsNotNone(python_type)
                        self.assertEqual(python_type, TestNISTQuerying.optional_cols[col])
            
            # Update
            progress_bar.update(1)
        
        progress_bar.close()

        # Problem Loop
        print("\n\nTest error handling for elements with bad data:")
        progress_bar = tqdm(total=len(problem_elements), desc="Problematic Elements")
        for el in problem_elements:
            results = TestNISTQuerying.retriever.retrieve(elements=[el], ionized=False)
            df = results[el]
            self.assertIsNone(df) # Elements with bad data are set to None
            progress_bar.update(1)
        
        progress_bar.close()
        

    def test_retrieve_invalid_elements(self):
        """
        Tests that retrieve raises TypeError for illegal element types and ValueError for invalid element names.
        """

        illegal_types = [None, 5, 100.7, False]
        illegal_element = "test"

        for type in illegal_types:
            with self.assertRaises(TypeError):
                TestNISTQuerying.retriever.retrieve(elements=[type], ionized=False)
        
        with self.assertRaises(ValueError):
            TestNISTQuerying.retriever.retrieve(elements=[illegal_element], ionized=False)
    
    
    def test_retrieve_invalid_ionization(self):
        """
        Tests that retrieve raises TypeError for invalid ionized argument types.
        """

        illegal_types = [None, 5, 100.7, "test"]
        for type in illegal_types:
            with self.assertRaises(TypeError):
                TestNISTQuerying.retriever.retrieve(elements=["H"], ionized=type)
    

    def test_extract_all(self):
        """
        Tests NISTRetriever.extract for all elements, checking file creation and data validity for both ionized and neutral forms.
        """

        test_dir = Path(TestNISTQuerying.TEST_DIR)
        TestNISTQuerying.retriever.extract(elements=None, save_dir=TestNISTQuerying.TEST_DIR, overwrite=True)

        for el in TestNISTQuerying.elements:
            if el in TestNISTQuerying.problem_elements:
                continue
            for ionization in (True, False):
                file = test_dir / el / f"{el}_ionized.pickle" if ionization else test_dir / el / f"{el}.pickle"
                df = pd.read_pickle(file)

                # Ensure data is as expected
                self.assertIsNotNone(df) # Dataframe exists
                self.assertGreater(len(df.values), 0) # Contains actual data

                for col in TestNISTQuerying.required_cols:
                    self.assertIn(col, df.columns)
                    dtype = df[col].dtype
                    python_type = self._map_numpy_dtype_to_python(dtype)
                    self.assertIsNotNone(python_type)
                    self.assertEqual(python_type, TestNISTQuerying.required_cols[col])
                for col in TestNISTQuerying.optional_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        python_type = self._map_numpy_dtype_to_python(dtype)
                        self.assertIsNotNone(python_type)
                        self.assertEqual(python_type, TestNISTQuerying.optional_cols[col])
        
        # Move extraction log to test
        shutil.move(test_dir / "extraction_log.json", "./gfactor/tests/querying/atomic_test_log.json")

        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(self.TEST_DIR))
    

    def test_extract_subset(self):
        """
        Tests extract for a random subset of elements, checking file creation and data validity for both ionized and neutral forms.
        """

        # Loop through random subset of elements
        safe_elements = [el for el in TestNISTQuerying.elements if el not in TestNISTQuerying.problem_elements]
        elements = random.sample(safe_elements, k=TestNISTQuerying.NUM_SAMPLES)
        test_dir = Path(TestNISTQuerying.TEST_DIR)
        TestNISTQuerying.retriever.extract(elements=elements, save_dir=TestNISTQuerying.TEST_DIR, overwrite=True)

        for el in elements:
            for ionization in (True, False):
                file = test_dir / el / f"{el}_ionized.pickle" if ionization else test_dir / el / f"{el}.pickle"
                df = pd.read_pickle(file)

                # Ensure data is as expected
                self.assertIsNotNone(df) # Dataframe exists
                self.assertGreater(len(df.values), 0) # Contains actual data

                for col in TestNISTQuerying.required_cols:
                    self.assertIn(col, df.columns)
                    dtype = df[col].dtype
                    python_type = self._map_numpy_dtype_to_python(dtype)
                    self.assertIsNotNone(python_type)
                    self.assertEqual(python_type, TestNISTQuerying.required_cols[col])
                for col in TestNISTQuerying.optional_cols:
                    if col in df.columns:
                        dtype = df[col].dtype
                        python_type = self._map_numpy_dtype_to_python(dtype)
                        self.assertIsNotNone(python_type)
                        self.assertEqual(python_type, TestNISTQuerying.optional_cols[col])

        # Remove test directory and associated files
        if test_dir.exists() and test_dir.is_dir():
            shutil.rmtree(Path(self.TEST_DIR))
    

    def test_extract_subset_invalid_elements(self):
        """
        Tests that extract raises TypeError for illegal element types and ValueError for invalid element names.
        """

        illegal_types = [None, 5, 100.7, False]
        illegal_element = "test"

        with self.assertRaises(TypeError):
            TestNISTQuerying.retriever.extract(elements=illegal_types, save_dir=TestNISTQuerying.TEST_DIR, overwrite=True)
        
        with self.assertRaises(ValueError):
            TestNISTQuerying.retriever.extract(elements=illegal_element, save_dir=TestNISTQuerying.TEST_DIR, overwrite=True)







