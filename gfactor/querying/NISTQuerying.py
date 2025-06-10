# Standard libraries
import requests
import argparse
import io
import warnings
from pandas.errors import ParserWarning
from pandas import StringDtype

from pathlib import Path
from typing import List
import time
from tqdm import tqdm

import pandas as pd
import numpy as np
import json

class NoDataAvailableError(Exception):
    """Raised when a query is valid but no data is available for the requested parameters."""
    pass

class NISTRetriever:
    
    """
    Custom class for retrieving data from the NIST Atomic Database. 
    Sample link: https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H+I%3B+C+I%3B+O+I%3B+S+I&limits_type=0&low_w=800&upp_w=6000&unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    Non-ionized: https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H%3B+C%3B+O%3B+S&limits_type=0&low_w=800&upp_w=6000&unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    With Conf: "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
    Without Conf: "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
    """
    
    def __init__(self):

        """
        NIST retriever object constructor.

        'elements_by_mass' provides a built-in sorting mechanism for the querying
        order when performing data extraction.  
        """

        self._base_url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra="
           
        self.elements_by_mass = {
            'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 
            'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.305, 
            'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098, 
            'Ar': 39.948, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 
            'Mn': 54.938, 'Fe': 55.845, 'Ni': 58.693, 'Co': 58.933, 'Cu': 63.546, 'Zn': 65.38, 
            'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 
            'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95, 
            'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 
            'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'I': 126.9, 'Te': 127.6, 'Xe': 131.29, 
            'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 
            'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.5, 
            'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49, 
            'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 
            'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 
            'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Pa': 231.04, 
            'Th': 232.04, 'Np': 237.0, 'U': 238.03, 'Am': 243.0, 'Pu': 244.0, 'Cm': 247.0, 
            'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 
            'Lr': 262.0, 'Rf': 267.0, 'Db': 270.0, 'Bh': 270.0, 'Sg': 271.0, 'Hs': 277.0, 
            'Mt': 278.0, 'Ds': 281.0, 'Rg': 282.0, 'Cn': 285.0, 'Nh': 286.0, 'Fl': 289.0, 
            'Mc': 290.0, 'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0
        }

    @staticmethod
    def __float_func(frac_str):
        """
        Removes the extraneous characters from "questionable" levels in a NIST generated Pandas dataframe
        for the relevant columns and converts half-integer momenta, e.g. 3/2, to decimals (1.5) for use in the g-factor
        script.

        Known questionable flags (to date) are: 'a', brackets i.e. '[ ]', '*', and '?'

        Parameters
        ----------
        frac_str : str
            String to be converted to a float.

        Returns
        -------
        float
            Float value of frac_str.
        """

        if type(frac_str) is float or type(frac_str) is np.float64:
            return frac_str
        
        try:
            number_string = frac_str.split('?')[0] 
        except AttributeError:
            return frac_str

        try:
            number_string = number_string.replace('[', '')
            number_string = number_string.replace(']', '')
            number_string = number_string.replace('*', '')
            number_string = number_string.replace('a', '')
            return float(number_string)
        except ValueError:
            
            try:
                top, bottom = number_string.split('/')
                try:
                    # If >1:
                    leading, top = top.split(' ')   
                except ValueError:  # if <1 handle thusly:
                    whole = 0
                    frac = float(top) / float(bottom)
                    if whole < 0:
                        return whole - frac
                    else:
                        return whole + frac      
            except ValueError:
                pass


    @staticmethod
    def __acc_swap(val):
        """
        Swaps the qualitative accuracies of the oscillator strengths from NIST data for the quantitative values,
        as a percentage of the oscillator strength. Note that for ratings of E the error is typically greater than 50%,
        so here we assign 70%.

        Parameters
        ----------
        val : str
            Qualitative oscillator strength accuracy.

        Returns
        -------
        acc: float
            Quantitative oscillator strength accuracy.
        """

        acc = 0
        if 'A' in val:
            if 'AAA' in val:
                acc = 0.003
            elif 'AA' in val:
                acc = 0.01
            elif 'A+' in val:
                acc = 0.02
            elif 'A' in val:
                acc = 0.03
        elif 'B' in val:
            if 'B+' in val:
                acc = 0.07
            elif 'B' in val:
                acc = 0.1
        elif 'C' in val:
            if 'C+' in val:
                acc = 0.18
            elif 'C' in val:
                acc = 0.25
        elif 'D' in val:
            if 'D+' in val:
                acc = 0.4
            elif 'D' in val:
                acc = 0.5
        elif 'E' in val:
            acc = 0.7
        else:
            acc = np.nan
        return acc
    

    def __url_build(self, atoms:List[str], ionized:bool):

        """
        Constructs final URL from components.

        Parameters
        ----------
        atoms : List[str]
            List of atoms to query for.
        ionized : bool
            Indicates whether or not ionized transitions are included.

        Returns
        -------
        url : str
            Constructed URL.
        """
        
        atom_comb = ""
        for atom in atoms:
            if ionized:
                atom_comb += "%3B+" + atom
            else:
                atom_comb += "%3B+" + atom + "+I"
                
        atom_comb = atom_comb.replace("%3B+", '', 1)  # Remove unneeded %3B+ at the beginning
        segway = "&limits_type=0&"
        
        # Hardcoded variables
        remaining = "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
        url = self._base_url + atom_comb + segway + remaining
        return url
    

    def __clean(self, df:pd.DataFrame, element:str):

        """
        Sequence of operations to tidy up column notation, standardize data types, etc.

        Parameters
        ----------
        df : pd.DataFrame
            Resultant DataFrame from successful query.
        elements : str
            Atomic species targeted in the query.

        Returns
        -------
        df : pd.DataFrame
            Cleaned DataFrame.
        """
        
        # Check dataframe validity
        core_cols = ['Ei(eV)', 'Ek(eV)', 'fik', 'J_i', 'J_k', 'Acc', 'conf_i', 'term_i', 'conf_k', 'term_k']
        wavelength_cols = ['obs_wl_vac(A)', 'obs_wl_air(A)']
        drop_cols = ['Unnamed: 0', 'intens', 'Type', 'line_ref', 'Unnamed: 17']
        missing_cores = [col for col in core_cols if col not in df.columns]
        missing_wavelengths = [col for col in wavelength_cols if col not in df.columns]
        
        if len(df) == 0 or len(missing_cores) > 0 or len(missing_wavelengths) == 2:
            raise NoDataAvailableError(f"No data available for element '{element}'")
        
        # Bring all wavelength data under the same header
        if "obs_wl_air(A)" in df.columns:
            df = df.rename(columns={"obs_wl_air(A)": "obs_wl(A)"})
        
        if "obs_wl_vac(A)" in df.columns:
            df = df.rename(columns={"obs_wl_vac(A)": "obs_wl(A)"})
        
        # Remove headers erroneously placed in the data
        df = df[df["intens"] != "intens"]
        df = df[df["obs_wl(A)"] != "obs_wl_air(A)"]
        df = df[df["obs_wl(A)"] != "obs_wl_vac(A)"]

        # Drop unneeded columns
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Adjust data type and content
        df['J_k'] = df['J_k'].apply(np.nan_to_num)
        
        for col in ['Ei(eV)', 'Ek(eV)', 'fik', 'J_i', 'J_k']:
            df[col] = df[col].apply(self.__float_func)
        
        dtype_adjust = {"obs_wl(A)": float, 
                        "unc_obs_wl":float,
                        "Aki(s^-1)":float,
                        'Acc': str, 
                        "sp_num": float, 
                        "conf_k": str, 
                        "term_k": str,
                        "conf_i": str, 
                        "term_i": str}
        
        if "unc_obs_wl" not in df.columns:
            del dtype_adjust["unc_obs_wl"]
        if "sp_num" not in df.columns:
            del dtype_adjust["sp_num"]

        for col, dtype in list(dtype_adjust.items()):
            if dtype == str:
                df[col] = df[col].fillna("").astype(StringDtype())
            else:
                df[col] = df[col].astype(float)

        df['Acc'] = df['Acc'].apply(self.__acc_swap)
        df['Acc'] = df['Acc'].astype(float)
        
        # For whatever reason, hydrogen isn't included if queried individually
        if 'element' not in df.columns:
            df['element'] = element
            df['sp_num'] = 1.0

        return df
    

    def retrieve(self, elements: List[str], ionized=False, timeout=10) -> dict[str, pd.DataFrame]:
        
        """
        Retrieve data from NIST Atomic Database.

        Parameters
        ----------
        elements : List[str]
            Atomic species to be queried for.
        ionized : bool, optional
            Indicates whether or not ionized transitions will be included.
        overwrite : bool, optional
            If True (and save_dir is not None), forcibly overwrite existing files.

        Returns
        -------
        dataframes : dict[str, pd.DataFrame]
            Maps elements to their dataframes - any elements with bad data will be mapped to None
        """

        dataframes = {}

        for el in elements:

            # Type checking
            if not isinstance(el, str):
                raise TypeError(f"Expected 'elements' to contain items of type 'str', found type '{type(el)}' instead.")

            if not isinstance(ionized, bool):
                raise TypeError(f"Expected 'ionization' to be of type 'bool', found type '{type(ionized)}' instead.")

            # Confirm element data available
            if el not in self.elements_by_mass:
                raise ValueError(f"{el} is not a recognizeable element")
            
            # Construct URL
            url = self.__url_build(elements, ionized)

            # Retrieve data
            try:
                response = requests.get(url, timeout=timeout)
            except requests.exceptions.Timeout as e:
                raise RuntimeError(f"Request to {url} timed out after {timeout} seconds.") from e
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=ParserWarning)
                df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), delimiter="\t", 
                                index_col=False, dtype=str)
                # Check if any warnings were captured
                if len(w) != 0:
                    dataframes[el] = None
                else:
                    df = self.__clean(df, el)
                    dataframes[el] = df
                    
        return dataframes


    def extract(self, elements:List[str] = None, save_dir:str = "../data/atomic", overwrite=False):

        """
        By default sweeps through 'elements_by_mass' object dictionary, querying all available
        elements 1-by-1 and saving both non-ionized and ionized transition CSV files to
        the chosen directory. If 'elements' is provided, then saves to the 

        Parameters
        ----------
        save_dir : str
            Save directory for atomic data.
        error_dir : str
            Directory to record error logs in.
        overwrite : bool, optional
            If True, forcibly overwrite existing files.

        Returns
        -------
        None
        """

        # Core directory
        dir = Path(save_dir)
        dir.mkdir(parents=True, exist_ok=True)

        # Elements List
        if elements:
            for el in elements:
                if not isinstance(el, str):
                    raise TypeError(f"Expected 'elements' to contain items of type 'str', found type '{type(el)}' instead.")

                # Confirm element data available
                if el not in self.elements_by_mass:
                    raise ValueError(f"{el} is not a recognizeable element")
        else:
            elements = list(self.elements_by_mass.keys())
        elements.sort(key = lambda x: self.elements_by_mass[x])

        # Progress Bar
        progress_bar = tqdm(total=len(elements), desc="Querying")

        # Setup log file
        log_dir = dir / "extraction_log.json"
        if Path(log_dir).exists():
            with open(log_dir, "r") as f:
                status_log = json.load(f)
        else:
            status_log = {}

        # Loop through all elements
        for el in elements:

            path = dir / el
            path.mkdir(parents=True, exist_ok=True)
            
            # Default for status log, if necessary
            if overwrite or el not in status_log:
                status_log[el] = {"standard": None, "ionized": None}

            for ionization in (False, True):
                status = "available"
                file = dir / el / f"{el}_ionized.pickle" if ionization else dir / el / f"{el}.pickle"

                # Move on to next element if data already exists 
                if file.exists() and not overwrite:
                    break
                
                # Retrieval
                try:
                    results = self.retrieve(elements=[el], ionized=ionization)
                    df = results[el]
                    pd.to_pickle(df, file)
                    # Request processes, but no data available
                    if df is None:
                        status = "bad data"

                # Results for this element are offline or unavailable
                except (requests.exceptions.Timeout, NoDataAvailableError):
                    status = "unavailable"

                # Sometimes SSL closes connections which have been open for too long: try one more time
                except requests.exceptions.RequestException:
                        
                        # First, give server some time to breath
                        time.sleep(10)  

                        # Attempt same query again
                        try:
                            results = self.retrieve(elements=[el], ionized=ionization)
                            df = results[el]
                            pd.to_pickle(df, file)
                            # Request processes, but no data available
                            if df is None:
                                status = "bad data"
                        
                        # Results for this element are offline or unavailable
                        except (requests.exceptions.Timeout, NoDataAvailableError, requests.exceptions.RequestException):
                            status = "unavailable"

                            
                # Write to log file
                if ionization:
                    status_log[el]["ionized"] = {"status":status}
                else:
                    status_log[el]["standard"] = {"status":status}
                
                # Write error data to JSON file
                with open(log_dir, "w") as f:
                    json.dump(status_log, f, indent=2)

            # Next element
            progress_bar.update(1)

        
        

def parse_args():
        
        # Get the absolute path to the current file (LISIRDQuerying.py)
        current_file = Path(__file__).resolve()

        # Get the package root (assuming this file is always in gfactor/querying/)
        package_root = current_file.parents[2]  # gfactor/

        # Build path to target directory
        atomic_dir = (package_root / "data" / "atomic").as_posix()

        p = argparse.ArgumentParser("Extract Atomic Species Data from NIST Atomic Spectral Database")
        p.add_argument("--elements", '-e', type=List[str], default=None, help="list of specific elements to extract")
        p.add_argument("--save-dir", '-sd', type=str, default=atomic_dir,
                    help="Save directory for elemental csv files")
        p.add_argument("--overwrite", "-o", type=bool, default=True,
                       help="Forcibly ovewrite existing files if true")
        
        return p.parse_args()


def main():
    args = parse_args()
    retriever = NISTRetriever()
    retriever.extract(elements=['H', 'C', 'O', 'S'],
                      save_dir=args.save_dir,
                      overwrite=args.overwrite)
      

if __name__ == "__main__":
    main()