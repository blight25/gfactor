# Standard libraries
import requests
import io
from pathlib import Path
from typing import List
import time
from tqdm import tqdm

# Third-party libraries
import pandas as pd
import numpy as np

from itertools import combinations

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


    def __url_build(self, limits, atoms, ionized):
        
        atom_comb = ""
        for atom in atoms:
            if ionized:
                atom_comb += "%3B+" + atom
            else:
                atom_comb += "%3B+" + atom + "+I"
                
        atom_comb = atom_comb.replace("%3B+", '', 1)  # Remove unneeded %3B+ at the beginning
        segway = "&limits_type=0&"
        limits_comb = f"low_w={limits[0]}&upp_w={limits[1]}&"
        
        # Hardcoded variables
        everything_else = "unit=0&de=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&bibrefs=1&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data"
        url = self._base_url + atom_comb + segway + limits_comb + everything_else
        return url
    

    def __clean(self, df, wavelength_bounds, elements):
        
        # Check dataframe validity
        core_cols = ['Ei(eV)', 'Ek(eV)', 'fik', 'J_i', 'J_k', 'Acc', 'conf_i', 'term_i', 'conf_k', 'term_k']
        wavelength_cols = ['obs_wl_vac(A)', 'obs_wl_air(A)']
        missing_cores = [col for col in core_cols if col not in df.columns]
        missing_wavelengths = [col for col in wavelength_cols if col not in df.columns]
        
        if len(df) == 0 or len(missing_cores) > 0 or len(missing_wavelengths) == 2:
            raise ValueError(f"No data available for these parameters: atoms -> {elements}, wavelength bounds -> {wavelength_bounds}")
        
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
            df[col] = df[col].apply(self.float_func)
            
        df = df.astype({"obs_wl(A)": float, "fik": float, "term_k": str, "Acc": str, "unc_obs_wl":float, "Aki(s^-1)":float})
        df['Acc'] = df['Acc'].apply(self.acc_swap)
        
        # Add columns for element and species number, if not already present
        if 'element' not in df.columns:
            df['element'] = elements[0]
            df['sp_num'] = 1

        return df
    

    def retrieve(self, wavelength_bounds: List[int], elements: List[str], ionized=False, save_file=None) -> pd.DataFrame:
        
        """
        Function to retrieve data from NIST within the gfactor framework. 
        
        @param limits: lower and upper wavelength bounds, in Angstroms
        @param elements: atomic species to be considered
        @param ionized: Optional, indicates whether or not ionized transitions will be included - default is false
        @param save_results: Optional, indicates whether or not the retrieved data will be save to an external file
        @return: df: results from API request
        
        """
        
        # Construct URL
        url = self.__url_build(wavelength_bounds, elements, ionized)
        
        # Retrieve data
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), delimiter="\t", index_col=False, dtype=str)
        df = self.__clean(df, wavelength_bounds, elements)
            
        # Save to an external file
        if save_file:
            file_path = Path(save_file)
            if not file_path.exists():
                df.to_csv(save_file)
        return df
    

    def extract(self, wavelength_bounds: List[int], elements: List[str], save_dir="./atomic_data", ionized=True, overwrite=False):

        def find_subsets(input_list):
            subsets = []
            for r in range(1, len(input_list) + 1):
                subsets.extend(combinations(input_list, r))
            return subsets

        # Core directory
        dir = Path(save_dir)
        dir.mkdir(parents=True, exist_ok=True)
        element_combs = find_subsets(elements)

        # Progress Bar
        progress_bar = tqdm(total=len(element_combs), desc="Querying")

        # Setup files for recording problem dates and queried dates
        with open ("./gfactor/querying/problem_elements.txt", "w") as problem_file: 
            problem_file.write("PROBLEM ELEMENTS\n\n")

        for comb in element_combs:
            comb = [*comb]
            comb.sort(key = lambda x: self.elements_by_mass[x]) # Sort string
            str_comb = ''.join(comb) # Make string query

            # Ionized
            if ionized:
                cur_dir = dir / "ionized"
                save_file = cur_dir / f"{str_comb}_{wavelength_bounds[0]}-{wavelength_bounds[1]}_ionized.csv"
                if not save_file.exists() or overwrite:
                    cur_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        self.retrieve(wavelength_bounds=wavelength_bounds, elements=comb, ionized=True, save_file=save_file)
                    except Exception as e:
                        if isinstance(e, ValueError):
                            with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                problem_file.write(f"\nRequests Error: elements={comb}, wavelength_lims={wavelength_bounds}, error = {e}\n\n, ionized=True") # Keep tabs on problematic dates
                        elif isinstance(e, requests.exceptions.RequestException):
                            time.sleep(10)
                            try:
                                self.retrieve(wavelength_bounds=wavelength_bounds, elements=comb, ionized=True, save_file=save_file)
                            except requests.exceptions.RequestException as e:
                                # Setup files for recording problem dates and queried dates
                                with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                    problem_file.write(f"\nRequests Error: elements={comb}, wavelength_lims={wavelength_bounds}, error = {e}\n\n, ionized=True") # Keep tabs on problematic dates

            # Standard
            cur_dir = dir / "standard"
            save_file = cur_dir / f"{str_comb}_{wavelength_bounds[0]}-{wavelength_bounds[1]}.csv"
            if not save_file.exists() or overwrite:
                cur_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self.retrieve(wavelength_bounds=wavelength_bounds, elements=comb, ionized=False, save_file=save_file)
                except Exception as e:
                    if isinstance(e, ValueError):
                        with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                            problem_file.write(f"\nRequests Error: elements={comb}, wavelength_lims={wavelength_bounds}, error = {e}\n\n") # Keep tabs on problematic dates
                    elif isinstance(e, requests.exceptions.RequestException):
                        time.sleep(10)
                        try:
                            self.retrieve(wavelength_bounds=wavelength_bounds, elements=comb, ionized=True, save_file=save_file)
                        except requests.exceptions.RequestException as e:
                            # Setup files for recording problem dates and queried dates
                            with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                problem_file.write(f"\nRequests Error: elements={comb}, wavelength_lims={wavelength_bounds}, error = {e}\n\n") # Keep tabs on problematic dates

            progress_bar.update(1)
    

    def extract_single(self, wavelength_bounds: List[int], elements: List[str], save_dir="./atomic_data", ionized=True, overwrite=False):

        # Core directory
        dir = Path(save_dir)
        dir.mkdir(parents=True, exist_ok=True)

        # Progress Bar
        progress_bar = tqdm(total=len(elements), desc="Querying")

        # Setup files for recording problem dates and queried dates
        with open ("./gfactor/querying/problem_elements.txt", "w") as problem_file: 
            problem_file.write("PROBLEM ELEMENTS\n\n")

        for element in elements:

            # Ionized
            if ionized:
                cur_dir = dir / "ionized"
                save_file = cur_dir / f"{element}_{wavelength_bounds[0]}-{wavelength_bounds[1]}_ionized.csv"
                if not save_file.exists() or overwrite:
                    cur_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        self.retrieve(wavelength_bounds=wavelength_bounds, elements=[element], ionized=True, save_file=save_file)
                    except Exception as e:
                        if isinstance(e, ValueError):
                            with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                problem_file.write(f"\nRequests Error: elements={element}, wavelength_lims={wavelength_bounds}, error = {e}\n\n, ionized=True") # Keep tabs on problematic dates
                        elif isinstance(e, requests.exceptions.RequestException):
                            time.sleep(10)
                            try:
                                self.retrieve(wavelength_bounds=wavelength_bounds, elements=[element], ionized=True, save_file=save_file)
                            except requests.exceptions.RequestException as e:
                                # Setup files for recording problem dates and queried dates
                                with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                    problem_file.write(f"\nRequests Error: elements={element}, wavelength_lims={wavelength_bounds}, error = {e}\n\n, ionized=True") # Keep tabs on problematic dates

            # Standard
            cur_dir = dir / "standard"
            save_file = cur_dir / f"{element}_{wavelength_bounds[0]}-{wavelength_bounds[1]}.csv"
            if not save_file.exists() or overwrite:
                cur_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self.retrieve(wavelength_bounds=wavelength_bounds, elements=[element], ionized=False, save_file=save_file)
                except Exception as e:
                    if isinstance(e, ValueError):
                        with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                            problem_file.write(f"\nRequests Error: elements={element}, wavelength_lims={wavelength_bounds}, error = {e}\n\n") # Keep tabs on problematic dates
                    elif isinstance(e, requests.exceptions.RequestException):
                        time.sleep(10)
                        try:
                            self.retrieve(wavelength_bounds=wavelength_bounds, elements=[element], ionized=True, save_file=save_file)
                        except requests.exceptions.RequestException as e:
                            # Setup files for recording problem dates and queried dates
                            with open ("./gfactor/querying/problem_elements.txt", "a") as problem_file:
                                problem_file.write(f"\nRequests Error: elements={element}, wavelength_lims={wavelength_bounds}, error = {e}\n\n") # Keep tabs on problematic dates

            progress_bar.update(1)

            
    @staticmethod
    def float_func(frac_str):
        
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
    def acc_swap(val):
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


if __name__ == "__main__":

    nist = NISTRetriever()
    # elements = [element for element in list(nist.elements_by_mass.keys()) if nist.elements_by_mass[element] <= 56]
    # nist.extract(wavelength_bounds=[800, 7000], elements=elements, save_dir="./atomic_data")
    elements = list(nist.elements_by_mass.keys())
    nist.extract_single(wavelength_bounds=[800, 7000], elements=elements, save_dir="./atomic_data")