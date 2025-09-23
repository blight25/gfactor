**gfactor, the up-to-date atomic fluorescence model**

# gfactor

`gfactor` is a Python package for computing atomic fluorescence efficiencies in a solar radiation field for astrophysical calculations. To produce the most accurate values the code uses API calls to both the National Institute of Standards and Technology's Atomic Spectral Database and the Laboratory for Atmospheric and Space Physics LiSIRD solar spectral database to obtain the atomic constants and solar irradiance data needed to calculate the fluorescence efficiency. This allows the user to specify specific dates of interest to obtain fluorescence efficiencies; early tests have shown that over the course of full solar cycles the fluorescence efficiencies can vary by up to 30%. 

---

## Installation

You can install `gfactor` directly from GitHub:

```bash
pip install git+https://github.com/username/gfactor.git
Alternatively, if you want to work with the latest development version:

bash
Copy code
git clone https://github.com/blight25/gfactor.git
cd gfactor
pip install -e .

Requirements
-Python 3.12+
-requests
-tqdm 
-pandas
-matplotlib
-scipy
-specutils
-astropy 
-lmfit
-pytest


All required dependencies will be installed automatically with pip.

Usage
Here’s a minimal example:

python
Copy code
import gfactor

# Example function call
result = gfactor.compute("H2O", wavelength=550)
print(result)
For more examples, see the documentation or the examples/ directory.

Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
