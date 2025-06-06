# Standard libraries
import requests
import io
import argparse

from datetime import datetime as dt
from datetime import date, timedelta

import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm
import math
import time

import json

from typing import List, Dict


class NoDataAvailableError(Exception):
    """Raised when a query is valid but no data is available for the requested parameters."""
    pass


class LISIRDRetriever():
    
    """ Class for fetching spectral data from LISIRD within the gfactor framework: for more details
    on LISIRD, see https://lasp.colorado.edu/lisird/"""
    
    def __url_build(self, ds: str) -> str:
        """
        Constructs the first half of the URL from components.

        Parameters
        ----------
        ds : str
            The dataset to query.

        Returns
        -------
        str
            The constructed URL.
        """
        url = self.__base_url + ds + "." + "csv"
        return url

    def __param_build(self, prjns: List[str], slctns: List[str], optns: List[str]) -> str:
        """
        Constructs specifications to be applied on requested data.

        Parameters
        ----------
        prjns : list of str
            Variables to return - defaults to all variables.
        slctns : list of str
            Variable constraints.
        optns : list of str
            Operations to be applied.

        Returns
        -------
        str
            URL component containing specification instructions.
        """

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
    
    
    def __query(self, ds: str, prjn: List[str] = None, slctn: List[str] = None, optn: List[str] = None, timeout=10) -> pd.DataFrame:
        """
        Retrieves specific data from LISIRD using the LATIS framework.

        Parameters
        ----------
        ds : str
            The dataset to request.
        prjn : list of str, optional
            Desired variables, e.g., ["time", "wavelength", "irradiance"]. Defaults to all categories.
        slctn : list of str, optional
            Variable constraints, e.g., ["irradiance>1360"].
        optn : list of str, optional
            Operations to apply, e.g., ["replace_missing(NaN)"].
        timeout : int or float, optional
            Timeout in seconds for the network request. Default is 10.

        Returns
        -------
        pd.DataFrame
            Resultant DataFrame from the API request.
        """
        
        # Build full url
        init_url = self.__url_build(ds)
        params = self.__param_build(prjn, slctn, optn)
        url = init_url + "?" + params
        try:
            response = requests.get(url, timeout=timeout)
        except requests.exceptions.Timeout as e:
            raise requests.exceptions.Timeout(f"Request to {url} timed out after {timeout} seconds.") from e
        if not response:
            print(response.text)
            return None
        
        # Return data
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        return df
    

    def __init__(self):
        """
        LISIRDRetriever object constructor.

        Initializes the base URL and dataset metadata for querying. 
        
        For an overview of available datasets, subsets, and date ranges, simply call 'print' on a LISIRDRetriever object.

        Attributes
        ----------
        __base_url : str
            Base URL for LISIRD API.
        irradiance_datasets : dict
            Nested dictionary containing dataset metadata.
        other_datasets : dict
            Nested dictionary containing alternate dataset metadata.
        """
      
        self.__base_url = "https://lasp.colorado.edu/lisird/latis/dap/"

        self.irradiance_datasets = {"TIMED": {"name": "timed_see_ssi_l3", "min_date": date(2002, 2, 8), "max_date": date(2023, 8, 30)},
                        "SORCE_low_res": {"name": "sorce_ssi_l3", "min_date": date(2003, 2, 25), "max_date": date(2020, 2, 25)},
                        "SORCE_high_res": {"name": "sorce_solstice_ssi_high_res", "min_date": date(2003, 3, 5), "max_date": date(2020, 2, 25)},
                        "NNL_low_res": {"name": "nnl_ssi_P1D", "min_date": date(1874, 5, 9), "max_date": date(2023, 12, 31)},
                        "NNL_high_res": {"name": "nnl_hires_ssi_P1D", "min_date": date(1874, 5, 9), "max_date": date(2023, 12, 31)}
                        }
        
        self.other_datasets = {"GOES_18": {"name": "noaa_goes18_euvs_1d", "min_date": date(2022, 9, 9), "max_date": date(2025, 1, 27)}}
    

    def __str__(self):
        info = ["\n-------------- Available Datasets & Subsets ------------\n"]
        info.append("----- Irradiance (SSI) -----\n")
        for k, v in self.irradiance_identifiers.items():
            if v[-1] is None:
                # No subsets
                min_date = self.irradiance_datasets[k]["min_date"].strftime("%Y-%m-%d")
                max_date = self.irradiance_datasets[k]["max_date"].strftime("%Y-%m-%d")
                info.append(f"  {k}\n    {min_date} through {max_date}\n")
            else:
                info.append(f"  {k}")
                for subset in v:
                    ds_key = k + "_" + subset
                    min_date = self.irradiance_datasets[ds_key]["min_date"].strftime("%Y-%m-%d")
                    max_date = self.irradiance_datasets[ds_key]["max_date"].strftime("%Y-%m-%d")
                    info.append(f"    {subset}: {min_date} through {max_date}")
                info.append(" ")
        info.append("----- Other datasets (TSI, bandpass, etc.) -----\n")
        for k, v in self.other_identifiers.items():
            if v[-1] is None:
                min_date = self.other_datasets[k]["min_date"].strftime("%Y-%m-%d")
                max_date = self.other_datasets[k]["max_date"].strftime("%Y-%m-%d")
                info.append(f"  {k}\n   {min_date} through {max_date}\n")
            else:
                info.append(f"  {k}")
                for subset in v:
                    ds_key = k + "_" + subset
                    min_date = self.other_datasets[ds_key]["min_date"].strftime("%Y-%m-%d")
                    max_date = self.other_datasets[ds_key]["max_date"].strftime("%Y-%m-%d")
                    info.append(f"    {subset}: {min_date} through {max_date}")
                info.append(" ")
        info.append(" ")
        return "\n".join(info)
    

    @property
    def irradiance_identifiers(self) -> Dict[str, List[str]]:
        """
        Maps names of SSI (Solar Spectral Irradiance) datasets to their respective subsets.

        Returns
        -------
        dict
            Dictionary mapping dataset names to subset lists [subset 1, subset 2, etc.]. If no subsets exist, maps to [None].
        """
        # Dataset Identification
        identifiers = {}
        for key in list(self.irradiance_datasets.keys()):
            parts = key.split('_', 1)
            if parts[0] in identifiers: # Already seen, must have subsets ('low_res', 'high_res', etc.)
                identifiers[parts[0]].append(parts[1])
            else:
                if len(parts) == 2: # Has a subset
                    identifiers[parts[0]] = [parts[1]]
                else:
                    identifiers[parts[0]] = [None] # No subset
        
        return identifiers


    @property
    def other_identifiers(self) -> Dict[str, List[str]]:
        """
        Maps names of alternative datasets (TSI, bandpass measurements, etc.) to their respective subsets.

        Returns
        -------
        dict
            Dictionary mapping dataset names to subset lists [subset 1, subset 2, etc.]. If no subsets exist, maps to [None].
            
            For example, the 'NNL' dataset maps to ['low_res', 'high_res'].
        """
        # Dataset Identification
        identifiers = {}
        for key in list(self.other_datasets.keys()):
            parts = key.split('_', 1)
            if parts[0] in identifiers: # Already seen, must have subsets ('low_res', 'high_res', etc.)
                identifiers[parts[0]].append(parts[1])
            else:
                if len(parts) == 2: # Has a subset
                    identifiers[parts[0]] = [parts[1]]
                else:
                    identifiers[parts[0]] = [None] # No subset
        
        return identifiers
    

    def retrieve(self, dataset:str, query_date:str, subset:str = None, timeout=10) -> pd.DataFrame:
        """
        Queries a dataset and returns a DataFrame of irradiance and uncertainty at each wavelength and time.

        Parameters
        ----------
        dataset : str
            Dataset to query, found in 'self.datasets'.
        query_date : str
            Date to query in 'YYYY-MM-DD' format. Required.
        subset : str, optional
            Subset of the dataset to query. Defaults to the most recent subset.
        timeout : int or float, optional
            Timeout in seconds for the network request. Default is 10.

        Returns
        -------
        pd.DataFrame
            Results from the API request.
        """
        
        if not isinstance(dataset, str):
            raise TypeError(f"Expected type 'str' for 'dataset', got type '{type(dataset)}' instead." 
                                    f"\nAvailable irradiance (SSI) datasets are: {list(self.irradiance_identifiers.keys())}."
                                    f"\nAvailable alternate datasets (TSI, bandpass, etc.) are: {list(self.other_identifiers.keys())}.")

        dataset = dataset.upper()
        identifiers = {**self.irradiance_identifiers, **self.other_identifiers}

        if dataset in self.irradiance_identifiers:
            datasets = self.irradiance_datasets
        
        elif dataset in self.other_identifiers:
            datasets = self.other_datasets
        
        else:
            raise ValueError(f"Dataset '{dataset}' not recognized." 
                           f"\nAvailable irradiance (SSI) datasets are: {list(self.irradiance_identifiers.keys())}."
                           f"\nAvailable alternate datasets (TSI, bandpass, etc.) are: {list(self.other_identifiers.keys())}.")

        subsets = identifiers[dataset] # Any type of sub-designation for the dataset, e.g. [low_res, high_res]
        
        if subset is not None:
            
            if not isinstance(subset, str):
                raise TypeError(f"Expected type 'str' for 'subset', got type '{type(subset)}' instead."
                                f"\nAvailable subsets are {subsets}.")
            
            if not subsets[-1]:
                raise ValueError(f"'{dataset}' has no subsets, but a specific subset was specified. Please set 'subset=None' if you wish to query this dataset.")
            
            elif subset not in subsets:
                raise ValueError(f"subset '{subset}' not recognized for dataset '{dataset}'. Available subsets are {subsets}.")
        
        elif subsets[-1] is not None:
            raise ValueError(f"dataset '{dataset}' consists of subsets {subsets}, but 'subsets' was set to None. Please specify which ")

        dataset = dataset + "_" + subset if subset else dataset
        ds = datasets[dataset]["name"]
        
        # Date selection
        if not isinstance(query_date, str):
            raise TypeError(f"Expected type 'str' for 'date', got {type(query_date)}. Note that the time range for the selected dataset '{dataset}'"
                             f" is: {datasets[dataset]['min_date']} through {datasets[dataset]['max_date']}.")
        
        query_date = dt.strptime(query_date, "%Y-%m-%d").date()
        
        # Add date selection to query
        slctn = ["time>=" + query_date.strftime("%Y-%m-%d")]
        upper_bound = query_date + timedelta(days=1)
        slctn.append("time<" + upper_bound.strftime("%Y-%m-%d"))
        
        # Keep track of upper and lower date bounds
        min_date = datasets[dataset]["min_date"]
        max_date = datasets[dataset]["max_date"]
        
        # Ensure target date is within the bounds
        if query_date < min_date or query_date > max_date:
            raise ValueError("Chosen date is out of bounds: must fall between " + min_date.strftime("%Y-%m-%d") +
                             " and " + max_date.strftime("%Y-%m-%d"))
        
        try:
            df = self.__query(ds=ds, slctn=slctn, timeout=timeout)
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Timeout occurred while querying dataset '{dataset}' on {query_date.strftime('%Y-%m-%d')}: you can check data availability from LISIRD directly"
                               "\nat https://lasp.colorado.edu/lisird/. In some cases, data may appear to be available on LISIRD, but is no longer accessible through the API.") from e
            
        # Check for successful query
        if not len(df.values > 0):
            raise NoDataAvailableError(f"No data available fo dataset {dataset} on date '{query_date.strftime('%Y-%m-%d')}': you can check data availability from LISIRD directly"
                                       "\nat https://lasp.colorado.edu/lisird/. In some cases, data may appear to be available on LISIRD, but is no longer accessible through the API.")
        
        # Successful query - modify columns
        else:
            for col in df.columns:
                if "time" in col:
                    df.drop(col, axis=1, inplace=True)
                    df["date (YYYY-MM-DD)"] = [query_date]*len(df)
            
            df["Dataset"] = dataset
                    
        return df
    

    def extract(self, dataset: str = "NNL", subset: str = None, start_date: str = None, end_date: str = None, 
                timeout=10, interval: int = 1, save_dir: str = "../data/spectra", overwrite: bool = False):
        """
        Extracts spectral data for a given dataset and saves it locally.

        Parameters
        ----------
        dataset : str, optional
            Dataset to query. Default is "NNL".
        subset : str, optional
            Subset of the dataset to query. Queries all subsets if None.
        start_date : str, optional
            Start date for querying in 'YYYY-MM-DD' format. Defaults to dataset's minimum date.
        end_date : str, optional
            End date for querying in 'YYYY-MM-DD' format. Defaults to dataset's maximum date.
        interval : int, optional
            Interval in days for querying. Default is 1.
        save_dir : str, optional
            Directory to save results. Default is '../data/spectra'.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        timeout : int or float, optional
            Timeout in seconds for the network request. Default is 10.

        Returns
        -------
        None
        """

         # Dataset Identification
        if not isinstance(dataset, str):
            raise TypeError(f"Expected type 'str' for 'dataset', got type '{type(dataset)}' instead." 
                            f"\nAvailable irradiance (SSI) datasets are: {list(self.irradiance_identifiers.keys())}."
                            f"\nAvailable alternate datasets (TSI, bandpass, etc.) are: {list(self.other_identifiers.keys())}.")
        
        dataset = dataset.upper()
        identifiers = {**self.irradiance_identifiers, **self.other_identifiers}
        
        if dataset in self.irradiance_identifiers:
            datasets = self.irradiance_datasets
        
        elif dataset in self.other_identifiers:
            datasets = self.other_datasets
        
        else:
            raise ValueError(f"Dataset '{dataset}' not recognized." 
                             f"\nAvailable irradiance (SSI) datasets are: {list(self.irradiance_identifiers.keys())}."
                             f"\nAvailable alternate datasets (TSI, bandpass, etc.) are: {list(self.other_identifiers.keys())}.")

        subsets = identifiers[dataset] # Any type of sub-designation for the dataset, e.g. [low_res, high_res]

        # Create save directory
        dataset_dir = Path(save_dir + "/" + dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults to be saved in directory {dataset_dir}")

        # Create json file for logging
        save_dir = Path(save_dir)
        log_dir = save_dir / "extraction_log.json"
        if Path(log_dir).exists():
            with open(log_dir, "r") as f:
                status_log = json.load(f)
        else:
            status_log = {}
        
        # Dictionary nesting for json file
        if subsets[-1] is not None:
            if overwrite or dataset not in status_log:
                status_log[dataset] = {}
                for subset in subsets:
                    status_log[dataset][subset] = {"last_date_queried":None, "bad_dates": [], "status":"working"}
        else:
            if overwrite or dataset not in status_log:
                status_log[dataset] = {}
                for subset in subsets:
                    status_log[dataset] = {"last_date_queried":None, "bad_dates": [], "status":"working"}
    
        with open(log_dir, "w") as f:
                json.dump(status_log, f, indent=2)
        
        # Check user-specified subset validity
        if subset is not None:
            
            if not isinstance(subset, str):
                raise TypeError(f"Expected type 'str' for 'subset', got type '{type(subset)}' instead."
                                "\nCall 'print' on a LISIRDretriever object to view all available datasets, subsets, and date ranges.")
            
            if not subsets[-1]:
                raise ValueError(f"'{dataset}' has no subsets, but a specific subset was specified. Please set 'subset=None' if you wish to query this dataset.")
            
            elif subset not in subsets:
                raise ValueError(f"subset '{subset}' not recognized for dataset '{dataset}'. Available subsets are {subsets}.")
            
            else:
                print(f"\n------------- Beginning query of {dataset}, subset: {subset} --------------\n")
            subs = [subset]
        
        else:
            print(f"\n------------- Beginning query of {dataset}, subsets: {identifiers[dataset]} --------------\n")
            subs = identifiers[dataset]
        
        
        # Min and max dates - for subset extraction, will be bounded by strictest time window
        default_start_date = date(1000, 1, 1)
        default_end_date = date(3000, 1, 1)
        if not subset:
            for sub in identifiers[dataset]:
                full_name = dataset + '_' + sub if sub else dataset
                default_start_date = max(default_start_date, datasets[full_name]["min_date"])
                default_end_date = min(default_end_date, datasets[full_name]["max_date"])
        else:
            full_name = dataset + '_' + subset
            default_start_date = max(default_start_date, datasets[full_name]["min_date"])
            default_end_date = min(default_end_date, datasets[full_name]["max_date"])
        
        # Check user start and end date validity
        if start_date is not None:
            # Typing
            if not isinstance(start_date, str):
                raise TypeError(f"Expected type 'str' for 'start_date', got {type(start_date)}. Note that the time range for the" 
                                f" selected dataset '{dataset}'"
                                f"is: {default_start_date} through {default_end_date}.")
            
            test_date = dt.strptime(start_date, "%Y-%m-%d").date()
            
            # Bounds
            if test_date < default_start_date:
                raise ValueError(f"chosen start date of {start_date} precedes the minimum" \
                                 f"start date of{default_start_date}.")
            start_date = test_date
        else:
            start_date = default_start_date
        
        if end_date is not None:
            # Typing
            if not isinstance(end_date, str):
                raise TypeError(f"Expected type 'str' for 'end_date', got {type(end_date)}. Note that the time range for the" 
                                    f" selected dataset '{dataset}'"
                                    f"is: {default_start_date} through {default_end_date}.")
            
            test_date = dt.strptime(end_date, "%Y-%m-%d").date()
            
            # Bounds
            if test_date > default_end_date:
                raise ValueError(f"chosen end date of {end_date} excedes the maximum" \
                                 f"end date of{default_end_date}.")
            end_date = test_date
        else:
            end_date = default_end_date
        
        print(f"\nStart date of {start_date.strftime('%Y-%m-%d')}")
        print(f"End date of {end_date.strftime('%Y-%m-%d')}")
        print(f"Step size of {interval} day(s)\n")

        # Query variable initialization
        query_date = start_date
        total_days = math.floor((end_date - default_start_date).days / interval)
        progress_bar = tqdm(total=total_days, desc="Querying")
        progress_bar.update((start_date - default_start_date).days) # Indicate progress relative to min date
        
        # Query loop
        while query_date <= end_date:

            # Subdirectory
            cur_dir = Path(dataset_dir / query_date.strftime("%Y-%m-%d"))

            # Unless command is given to overwrite, continue if directory exists
            if cur_dir.exists() and not overwrite:
                
                # Write current date to file
                with open(log_dir, "w") as query_file:
                    query_file.write(query_date.strftime('%Y-%m-%d'))

                # Update progress bar
                query_date += timedelta(interval)
                progress_bar.update(1)

                continue
            
            # Otherwise, make the directory
            cur_dir.mkdir(parents=True, exist_ok=True)
            
            # Subset loop
            data = {}
            for sub in subs:
                try:
                    data[sub] = self.retrieve(dataset=dataset, subset=sub, 
                                                query_date=query_date.strftime("%Y-%m-%d"), timeout=timeout)
                    if sub is not None:
                        status_log[dataset][sub]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                    else:
                        status_log[dataset]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                    with open (log_dir, "w") as f:
                        json.dump(status_log, f, indent=2)
                except (requests.exceptions.Timeout, NoDataAvailableError) as e:  # Single indicate dataset is offline
                    if sub is not None:
                        status_log[dataset][sub]["bad_dates"].append((query_date.strftime("%Y-%m-%d"), e))
                        status_log[dataset][sub]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                    else:
                        status_log[dataset]["bad_dates"].append((query_date.strftime("%Y-%m-%d"), e))
                        status_log[dataset]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                    with open (log_dir, "w") as f:
                        json.dump(status_log, f, indent=2)
                    query_date += timedelta(interval)
                    progress_bar.update(1)
                    continue
                except requests.exceptions.RequestException: # sometimes SSL closes connections which have been open for too long: try one more time
                    time.sleep(10) # First, give server some time to breath
                    try: # Try one more time
                        data[sub] = self.retrieve(dataset=dataset, subset=sub, 
                                                    query_date=query_date.strftime("%Y-%m-%d"), timeout=timeout)
                        if sub is not None:
                            status_log[dataset][sub]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                        else:
                            status_log[dataset]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                        with open (log_dir, "w") as f:
                            json.dump(status_log, f, indent=2)
                    except (requests.exceptions.Timeout, requests.exceptions.RequestException, NoDataAvailableError) as e: # Back-to-back failures isn't a coincidence
                        if sub is not None:
                            status_log[dataset][sub]["bad_dates"].append((query_date.strftime("%Y-%m-%d"), e))
                            status_log[dataset][sub]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                        else:
                            status_log[dataset]["bad_dates"].append((query_date.strftime("%Y-%m-%d"), e))
                            status_log[dataset]["last_date_queried"] = query_date.strftime("%Y-%m-%d")
                        with open (log_dir, "w") as f:
                            json.dump(status_log, f, indent=2)
                        query_date += timedelta(interval)
                        progress_bar.update(1)
                        continue

            # Save dataframes
            for sub in list(data.keys()):
                filename = dataset + "_" + sub if sub else dataset
                pd.to_pickle(data[sub], cur_dir / f"{filename}.pickle")
        
            # Update progress bar
            query_date += timedelta(interval)
            progress_bar.update(1)
    
 
def parse_args():
        p = argparse.ArgumentParser("Extract spectral data from LISIRD")
        p.add_argument("--dataset", '-ds', type=str, default="nnl",
                       help="LISIRD dataset to query from")
        p.add_argument("--subset", '-sub', type=str, default=None,
                       help="Queries every subset (e.g. high-res and low-res) for a given dataset")
        p.add_argument("--interval", "-i", type=int, default=365,
                       help="Queries every ith date from min to max")
        p.add_argument("--start-date", '-start', type=str, default=None,
                       help="Start date for querying, in 'YYYY-MM-DD' format")
        p.add_argument("--end-date", "-end", type=str, default=None,
                       help="End date for querying, in 'YYYY-MM-DD' format")
        p.add_argument("--timeout", "-t", type=int, default=10,
                       help="Time (in seconds) allowed for request to process before forcing timeout")
        p.add_argument("--save-dir", '-save', type=str, default="./data/spectra",
                    help="Save directory for spectral pickle files")
        p.add_argument("--overwrite", "-o", type=bool, default=False,
                    help="Forcibly ovewrite existing files if true")
        
        return p.parse_args()


def main():
    args = parse_args()
    retriever = LISIRDRetriever()
    retriever.extract(dataset=args.dataset,
                      subset=args.subset,
                      start_date=args.start_date,
                      end_date=args.end_date,
                      timeout=args.timeout,
                      interval=args.interval,
                      save_dir=args.save_dir,
                      overwrite=args.overwrite)

if __name__ == "__main__":
    main()