# Standard libraries
import requests
import argparse
import io
import warnings
from pandas.errors import ParserWarning

from pathlib import Path
from typing import List
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

class NISTRetriever:
    
    """
    Custom class for retrieving data from the NIST Atomic Database. Note that, while not in the imports list,
    openpyxl is necessary for saving to excel files.
    Sample link: https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H+I%3B+C+I%3B+O+I%3B+S+I&limits_type=0&low_w=800&upp_w=6000&unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    Non-ionized: https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H%3B+C%3B+O%3B+S&limits_type=0&low_w=800&upp_w=6000&unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    With Conf: "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
    Without Conf: "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
    """
    
    def __init__(self):

        self._base_url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra="
        self.df = None  # Pandas dataframe, filled out by the data_retrieval function
           
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
        
        """Removes the extraneous characters from "questionable" levels in a NIST generated Pandas dataframe
        for the relevant columns and converts half-integer momenta, e.g. 3/2, to decimals (1.5) for use in the g-factor
        script.

        Known questionable flags (to date) are: 'a', brackets i.e. '[ ]', '*', and '?'

        @param frac_str: string to be converted to a float
        @return: float value of frac_str
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
        """Swaps the qualitative accuracies of the oscillator strengths from NIST data for the quantitative values,
        as a percentage of the oscillator strength. Note that for ratings of E the error is typically greater than 50%,
        so here we assign 70%.

        @param val: qualitative oscillator strength accuracy
        @return: quantitative oscillator strength accuracy
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
            acc = val
        return acc
    

    def __url_build(self, atoms, ionized):
        
        atom_comb = ""
        for atom in atoms:
            if ionized:
                atom_comb += "%3B+" + atom
            else:
                atom_comb += "%3B+" + atom + "+I"
                
        atom_comb = atom_comb.replace("%3B+", '', 1)  # Remove unneeded %3B+ at the beginning
        segway = "&limits_type=0&"
        
        # Hardcoded variables
        everything_else = "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
        url = self._base_url + atom_comb + segway + everything_else
        return url
    

    def __clean(self, df, elements):
        
        # Check dataframe validity
        core_cols = ['Ei(eV)', 'Ek(eV)', 'fik', 'J_i', 'J_k', 'Acc', 'conf_i', 'term_i', 'conf_k', 'term_k']
        wavelength_cols = ['obs_wl_vac(A)', 'obs_wl_air(A)']
        missing_cores = [col for col in core_cols if col not in df.columns]
        missing_wavelengths = [col for col in wavelength_cols if col not in df.columns]
        
        if len(df) == 0 or len(missing_cores) > 0 or len(missing_wavelengths) == 2:
            raise ValueError(f"No data available for these parameters: atoms -> {elements}")
        
        # Bring all wavelength data under the same header
        if "obs_wl_air(A)" in df.columns:
            df = df.rename(columns={"obs_wl_air(A)": "obs_wl(A)"})
        
        if "obs_wl_vac(A)" in df.columns:
            df = df.rename(columns={"obs_wl_vac(A)": "obs_wl(A)"})
        
        # Remove headers erroneously placed in the data
        df = df[df["intens"] != "intens"]
        df = df[df["obs_wl(A)"] != "obs_wl_air(A)"]
        df = df[df["obs_wl(A)"] != "obs_wl_vac(A)"]
        
        # Adjust data type and content
        df['J_k'] = df['J_k'].apply(np.nan_to_num)
        
        for col in ['Ei(eV)', 'Ek(eV)', 'fik', 'J_i', 'J_k']:
            df[col] = df[col].apply(self.__float_func)
            
        df = df.astype({"obs_wl(A)": float, "fik": float, "term_k": str, "Acc": str, "unc_obs_wl":float, "Aki(s^-1)":float})
        df['Acc'] = df['Acc'].apply(self.__acc_swap)
        
        # Add columns for element and species number, if not already present
        if 'element' not in df.columns:
            df['element'] = elements[0]
            df['sp_num'] = 1

        return df
    

    def retrieve(self, elements: List[str], ionized=False, save_dir=None, overwrite=False) -> pd.DataFrame:
        
        """
        Function to retrieve data from NIST within the gfactor framework. 
        
        @param wavelength_bounds: lower and upper wavelength bounds, in Angstroms
        @param elements: atomic species to be considered
        @param ionized: Optional, indicates whether or not ionized transitions will be included
        @save_dir: Optional, saves csv to this directory if provided
        @param overwite: Optional, if saving results, forcibly overwrite existing files
        @return: df: results from API request
        
        """

        # Save to an external file
        if save_dir:

            # Make path
            dir = Path(save_dir)
            dir.mkdir(parents=True, exist_ok=True)
            elements_str = " ".join(elements)

            if ionized:
                save_file = dir / f"{elements_str}_ionized.csv"
            else:
                save_file = dir / f"{elements_str}.csv"
            
            if not overwrite and save_file.exists():
                return None # Data already exists

        # Construct URL
        url = self.__url_build(elements, ionized)

        # Retrieve data
        response = requests.get(url)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=ParserWarning)
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), delimiter="\t", 
                            index_col=False, dtype=str)
            # Check if any warnings were captured
            if len(w) != 0:
                return None # Faulty dataframe
            
        # Clean, save, return data
        df = self.__clean(df, elements)
        df.to_csv(save_file)

        return df
    

    def extract(self, save_dir, error_dir, overwrite):

        # Core directory
        dir = Path(save_dir)
        dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        ionized_dir = dir / "ionized"
        standard_dir = dir / "standard"

        # Elements List
        elements = list(self.elements_by_mass.keys())
        elements.sort(key = lambda x: self.elements_by_mass[x])

        # Progress Bar
        progress_bar = tqdm(total=len(elements), desc="Querying")

        # Setup file for recording problem dates and queried dates
        error_dir = Path(error_dir)
        error_dir.mkdir(parents=True, exist_ok=True)
        error_file = error_dir / "problem_elements.txt"
        with open (error_file, "w") as problem_file: 
            problem_file.write("PROBLEM ELEMENTS\n\n")

        for element in elements:
            for cur_dir, ionization in [(standard_dir, False), (ionized_dir, True)]:
                try:
                    self.retrieve(elements=[element], ionized=ionization, 
                                  save_dir=cur_dir, overwrite=overwrite)
                except Exception as e:
                    if isinstance(e, ValueError):
                        with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                            problem_file.write(f"\nRequests Error: elements={[element]}," \
                                               f"ionization={ionization}, error = {e}\n\n") # Keep tabs on problematic dates
                    elif isinstance(e, requests.exceptions.RequestException):
                        time.sleep(5)
                        try:
                            self.retrieve(elements=[element], ionized=ionization, 
                                          save_dir=cur_dir, overwrite=overwrite)
                        except requests.exceptions.RequestException as e:
                            with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                problem_file.write(f"\nRequests Error: elements={[element]}," \
                                               f"ionization={ionization}, error = {e}\n\n") # Keep tabs on problematic dates

            progress_bar.update(1)


def parse_args():
        p = argparse.ArgumentParser("Extract Atomic Species Data from NIST Atomic Spectral Database")
        p.add_argument("--save-dir", '-sd', type=str, default="./data/atomic",
                    help="Save directory for elemental csv files")
        p.add_argument("--error-dir", '-ed', type=str, default="./data/errors",
                       help="Logs errors with specific queries, if any")
        p.add_argument("--overwrite", "-o", type=bool, default=False,
                       help="Forcibly ovewrite existing files if true")
        
        return p.parse_args()


def main():
    args = parse_args()
    retriever = NISTRetriever()
    retriever.extract(save_dir=args.save_dir,
                      error_dir = args.error_dir,
                      overwrite=args.overwrite)
      

if __name__ == "__main__":
    main()