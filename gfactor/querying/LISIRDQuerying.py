
# Standard libraries
import requests
import io
from datetime import datetime as dt
from datetime import date, timedelta
from typing import List

import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm
import math
import time


class LISIRDRetriever():
    
    """ Class for fetching data from LISIRD within the gfactor framework."""
    def __url_build(self, ds):
        url = self.__base_url + ds + "." + "csv"
        return url

    def __param_build(self, prjns, slctns, optns):
        # Initialize variables
        no_ampersand = True
        params = ""
        # Unpack projections
        if prjns is not None:
            prjn_comb = ""
            for prjn in prjns:
                prjn_comb += ',' + prjn
            prjn_comb = prjn_comb.replace(',', '', 1)
            params += prjn_comb
            no_ampersand = False  # Need an ampersand as a segway in the url
        # Unpack selections
        if slctns is not None:
            slctn_comb = ""
            for slctn in slctns:
                slctn_comb += '&' + slctn
            if no_ampersand:
                slctn_comb = slctn_comb.replace('&', '', 1)
            params += slctn_comb
            no_ampersand = False  # Need an ampersand as a segway in the url
        # Unpack operations
        if optns is not None:
            optn_comb = ""
            for optn in optns:
                optn_comb += '&' + optn
            if no_ampersand:
                optn_comb = optn_comb.replace('&', '', 1)
            params += optn_comb

        return params
    

    def __query(self, ds: str, prjn = None, slctn = None, optn = None) -> pd.DataFrame:
        
        """
        Retrieves specific data from LISIRD based off of information provided by the user. Automatically returns a
        Pandas dataframe storing the results, and saves to an external file if requested.

        @param ds: string, The dataset to request (see Available Datasets in LATIS documentation)

        @param prjn: Optional, string list of desired variables: for instance, to target the time, wavelength, and
        irradiance variables, set prjn = ["time", "wavelength", "irradiance"]. If this parameter is not specified,
        then all categories will be included. For more information on valid variable identifiers, see the
        'Available Datasets' section of the LATIS documenation.

        @param slctn: Optional, string list of variable constraints: for instance, ["irradiance>1360"] will remove any
        irradiance values less than 1360. Note that, if seeking to specify time periods with more than just the
        year, the formatting is YYYY-MM-DDTHH:MM. For example, to find data in a specific time range, slctn would
        look something like ["time>=2005-05-05T12:00','time<2006-05-05T12:00"]. If this parameter is not specified,
        then no constraints will be applied.

        @param optn: Optional, string list of operations to be applied to the dataset as a whole. For instance,
        ["replace_missing(NaN)"] will replace any missing values with NaN. If this parameter is not specified,
        then the "last()" operation will be applied in order to pull the most recent data sample. This can be overridden
        by setting optn = None or to some other set of operations. However, if no time constraints were previously
        applied in slctn or a large sample size (n) is requested, then the request may be time-consuming - and
        could potentially cause an overload error. For more information on valid operations, see the 'Operation Options'
        section of the LATIS documentation.

        @param save_results: Boolean, indicates whether the dataset will be saved to an external file

        @return: df: resultant dataframe from API request
        """
        
        # Build full url
        init_url = self.__url_build(ds)
        params = self.__param_build(prjn, slctn, optn)
        url = init_url + "?" + params
        
        # Retrieve data
        response = requests.get(url)
        if not response:
            print(response.text)
            return None
        
        # Return data
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        return df
    

    def __init__(self):
      
        self.__base_url = "https://lasp.colorado.edu/lisird/latis/dap/"

        self.datasets = {"TIMED": {"name": "timed_see_ssi_l3", "min_date": date(2002, 2, 8), "max_date": date(2023, 8, 30)},
                        "SORCE": {"name": "sorce_ssi_l3", "min_date": date(2003, 2, 25), "max_date": date(2020, 2, 25)},
                        "GOES_18": {"name": "noaa_goes18_euvs_1d", "min_date": date(2022, 9, 9), "max_date": date(2025, 1, 27)},
                        "NNL_low_res": {"name": "nnl_ssi_P1D", "min_date": date(1874, 5, 9), "max_date": date(2023, 12, 31)},
                        "NNL_high_res": {"name": "nnl_hires_ssi_P1D", "min_date": date(1874, 5, 9), "max_date": date(2023, 12, 31)}
                        }
    

    def retrieve(self, dataset:str, date=None, wavelength_bounds=None, max_queries=10) -> pd.DataFrame:
        
        """ Takes a dataset name to query, as well as optional date and wavelength specifications, and returns a pandas
        dataframe recording the irradiance and uncertainty at each wavelength and time.
        
        @param dataset: dataset to query, can be found in 'self.datasets'
        @param date: optional, date to query in 'YYYY-MM-DD' format
        @param wavelength_bounds: optional, format is [lower_bound, upper_bound] in Angstroms. If only one bound is desired, set the 
        other element to None
        @param max_queries: optional, integer to specify the maximum number of queries to make before raising an error
        
        @return df: results from API request
        """
        
        # Create dictionary mapping all dataset names to their subsets
        names = {}
        for key in list(self.datasets.keys()):
            parts = key.split('_', 1)
            if parts[0] in names:
                names[parts[0]].append(parts[1])
            else:
                if len(parts) == 2:
                    names[parts[0]] = [parts[1]]
                else:
                    names[parts[0]] = [None]

        # Generic dataset query
        if dataset.upper() in names:
            subset = names[dataset.upper()][-1] # Picks most recent subset available (eg. 'high_res' for NNL, '18' for GOES, etc.)
            dataset = dataset.upper()
            if subset:
                dataset += '_' + subset
        
        # Not in names - check for specific subset
        elif dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not recognized: available datasets are: {list(names.keys())}")
        
        ds = self.datasets[dataset]["name"]
        
        # Date selection
        if date is None:
            raise ValueError(f"""Query date is required. Time range for the selected dataset '{dataset}' is:
                             {self.datasets[dataset]['min_date']} through {self.datasets[dataset]['max_date']}.""")
     
        init_date = dt.strptime(date, "%Y-%m-%d").date()
        
        # Add date selection to query
        slctn = ["time>=" + init_date.strftime("%Y-%m-%d")]
        upper_bound = init_date + timedelta(days=1)
        slctn.append("time<" + upper_bound.strftime("%Y-%m-%d"))
        
        # Keep track of upper and lower date bounds
        min_date = self.datasets[dataset]["min_date"]
        max_date = self.datasets[dataset]["max_date"]
        
        # Ensure target date is within the bounds
        if init_date < min_date or init_date > max_date:
            raise ValueError("Chosen date is out of bounds: must fall between " + min_date.strftime("%Y-%m-%d") +
                             " and " + max_date.strftime("%Y-%m-%d"))
        
        # Check if wavelength bounds were provided - otherwise, all wavelengths will be included
        if wavelength_bounds is not None:
            if wavelength_bounds[0] is not None:
                wavelength_bounds[0] = wavelength_bounds[0] / 10 # Convert from Angstrom to nm
                min_wave = str(wavelength_bounds[0])
                slctn.append("wavelength>=" + min_wave)
            if wavelength_bounds[1] is not None:
                wavelength_bounds[1] = wavelength_bounds[1] / 10 # Convert from Angstrom to nm
                max_wave = str(wavelength_bounds[1])
                slctn.append("wavelength<=" + max_wave)
            if wavelength_bounds[0] is not None and wavelength_bounds[1] is not None:
                if wavelength_bounds[0] >= wavelength_bounds[1]:
                    raise ValueError("Invalid wavelength bounds: lower bound must be less than upper bound")
        
        # Search variables
        cur_date = init_date
        cur_query = 0
        upwards = True # If query is unsuccessful, search at more recent dates
        
        # Query loop
        while cur_query < max_queries:
            
            df = self.__query(ds=ds, slctn=slctn)
            
            # Check for successful query
            if len(df.values > 0):
                break
            
            # Incrementing date, looking for new data
            if upwards:
                new_date = upper_bound
                upper_bound = upper_bound + timedelta(days=1)
                
                # Max date reached without any results, search at older dates instead
                if upper_bound >= max_date:
                    new_date = init_date - timedelta(days=1)
                    upper_bound = init_date
                    upwards = False
                        
            # Decrementing date, looking for old data
            else:
                new_date = cur_date - timedelta(days=1)
                upper_bound = cur_date
                
                # No more data to query
                if new_date <= min_date:
                    raise ValueError("Querying unsuccessful within the known date range: it may be that this dataset is no longer available")
            
            # Update query parameters
            slctn[0] = "time>=" + new_date.strftime("%Y-%m-%d")
            slctn[1] = "time<" + upper_bound.strftime("%Y-%m-%d")
            cur_date = new_date
            cur_query += 1
        
        # Check if max number of queries was made without a valid result
        if cur_query == max_queries:
            raise ValueError("Max queries reached without success")

        # Successful query - modify columns
        else:
            for col in df.columns:
                if "time" in col:
                    df.drop(col, axis=1, inplace=True)
                    df["date (YYYY-MM-DD)"] = [cur_date]*len(df)
            
            df["Dataset"] = dataset
                    
            return df
    
    
    def extract(self, dataset="NNL", all_subsets=False, start_date:str=None, end_date:str=None, 
                interval:int=1, save_dir="./spectra", overwrite=False):

        # Dataset Identification
        names = {}
        for key in list(self.datasets.keys()):
            parts = key.split('_', 1)
            if parts[0] in names: # Already seen, must have subsets ('low_res', 'high_res', etc.)
                names[parts[0]].append(parts[1])
            else:
                if len(parts) == 2: # Has a subset
                    names[parts[0]] = [parts[1]]
                else:
                    names[parts[0]] = [None] # No subset

        default_subset = None
        if dataset in self.datasets:
            parts = dataset.split('_', 1)
            if all_subsets:
                raise ValueError(f"'{dataset}' is already a subset of {parts[0]}: please set 'all_subsets' to false to ensure expected behavior.")
            dataset = parts[0]

            if len(parts) == 2:
                default_subset = parts[1]
        else:
            dataset = dataset.upper()
            if dataset not in names:
                raise ValueError(f"Dataset {dataset} not recognized: available datasets are: {list(names.keys())}")

        # Create save directory
        save_dir = Path(save_dir + "/" + dataset)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Min and max dates - for subset extraction, will be bounded by strictest time window
        default_start_date = date(1000, 1, 1)
        default_end_date = date(3000, 1, 1)
        if all_subsets:
            for subset in names[dataset]:
                full_name = dataset
                if subset:
                    full_name += '_' + subset
                default_start_date = max(default_start_date, self.datasets[full_name]["min_date"])
                default_end_date = min(default_end_date, self.datasets[full_name]["max_date"])
        else:
            full_name = dataset
            subset = default_subset if default_subset else names[dataset][-1]
            if subset:
                full_name += '_' + subset
            default_start_date = max(default_start_date, self.datasets[full_name]["min_date"])
            default_end_date = min(default_end_date, self.datasets[full_name]["max_date"])
        
        # User start and end dates - check validity
        if start_date is not None:
            test_date = dt.strptime(start_date, "%Y-%m-%d").date()
            if test_date < default_start_date:
                raise ValueError(f"chosen start date of {start_date} precedes the minimum" \
                                 f"start date of{default_start_date}.")
            start_date = test_date
        else:
            start_date = default_start_date
        
        if end_date is not None:
            test_date = dt.strptime(end_date, "%Y-%m-%d").date()
            if test_date > default_end_date:
                raise ValueError(f"chosen end date of {end_date} excedes the maximum" \
                                 f"end date of{default_end_date}.")
            end_date = test_date
        else:
            end_date = default_end_date

        # Query variable initialization
        query_date = start_date
        total_days = math.floor((end_date - default_start_date).days / interval)
        progress_bar = tqdm(total=total_days, desc="Querying")
        progress_bar.update((start_date - default_start_date).days) # Indicate progress relative to min date
        
        # Setup files for recording problem dates and queried dates
        with open ("./gfactor/querying/problem_dates.txt", "w") as problem_file: 
            problem_file.write("PROBLEM DATES\n\n")
        
        # Loop
        while query_date <= end_date:

            # Subdirectory
            cur_dir = Path(save_dir / query_date.strftime("%Y-%m-%d"))

            # Unless command is given to overwrite, continue if directory exists
            if cur_dir.exists() and not overwrite:
                
                # Write current date to file
                with open("./gfactor/querying/queried_dates.txt", "w") as query_file:
                    query_file.write(query_date.strftime('%Y-%m-%d'))

                # Update progress bar
                query_date += timedelta(interval)
                progress_bar.update(1)

                continue
            
            # Otherwise, make the directory
            cur_dir.mkdir(parents=True, exist_ok=True)
            
            # Query
            try:
                data = {}
                if all_subsets:
                    for subset in names[dataset]:
                        full_name = dataset
                        if subset:
                            full_name += '_' + subset
                        data[subset] = self.retrieve(dataset=full_name, date=query_date.strftime("%Y-%m-%d"), max_queries=1)
                else:
                    full_name = dataset
                    subset = default_subset if default_subset else names[dataset][-1]
                    if subset:
                        full_name += '_' + subset
                    data[subset] = self.retrieve(dataset=full_name, date=query_date.strftime("%Y-%m-%d"), max_queries=1)
            
            except requests.exceptions.RequestException: # sometimes SSL behaves weird, or the connection gets closed suddenly/times out: try one more time
                
                time.sleep(10) # First, give server some time to breath
                
                # Query again
                try:
                    data = {}
                    if all_subsets:
                        for subset in names[dataset]:
                            full_name = dataset
                            if subset:
                                full_name += '_' + subset
                            data[subset] = self.retrieve(dataset=full_name, date=query_date.strftime("%Y-%m-%d"), max_queries=1)
                    else:
                        full_name = dataset
                        subset = default_subset if default_subset else names[dataset][-1]
                        if subset:
                            full_name += '_' + subset
                        data[subset] = self.retrieve(dataset=full_name, date=query_date.strftime("%Y-%m-%d"), max_queries=1)
                
                except requests.exceptions.RequestException as e: # Back-to-back failures probably isn't a coincidence
                    with open ("./gfactor/querying/problem_dates.txt", "a") as problem_file:
                        problem_file.write(f"Requests Error: {query_date}, {e}\n\n") # Keep tabs on problematic dates
                    query_date += timedelta(interval)
                    progress_bar.update(1)
                    continue
            
            except ValueError: # There was no data available, or else something (that isn't an SSL or connection issue) didn't go right
                with open ("./gfactor/querying/problem_dates.txt", "a") as problem_file:
                    problem_file.write(f"Value Error: {query_date}\n\n") # Keep tabs on problematic dates
                query_date += timedelta(interval)
                progress_bar.update(1)
                continue
            
            # Save dataframes
            for subset in list(data.keys()):
                filename = subset if subset else dataset
                pd.to_pickle(data[subset], cur_dir / f"{filename}.pickle")
            
            # Write current date to file
            with open("./gfactor/querying/queried_dates.txt", "w") as query_file:
                query_file.write(query_date.strftime('%Y-%m-%d'))
        
            # Update progress bar
            query_date += timedelta(interval)
            progress_bar.update(1)



if __name__ == "__main__":
    retriever = LISIRDRetriever()
    data = retriever.retrieve("nnl", "2015-05-22")
    print(data)
    