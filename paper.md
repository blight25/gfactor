# Summary

A major componenent within observational and theoretical astrophysics centers around determining how efficient a particular atom, molecule, or ion is at producing photons. For planetary science in particular, determing how efficient our Sun's radiation is at being absorbed and re-radiated by both neutral and ion species is critical to our ability to derive quantitative abundances, examine excitation processes, and probe beyond our own atmosphere. Even within our own solar system there are thousands of unique use cases for determining these fluorescence efficiencies, sometimes referred to as _g_-factors, each dependent on the relevant solar activity, heliocentric distance, heliocentric velocity, and gas temperature.

While there are many published articles that contain tabulated fluorescence efficiencies, to date there has not been a software package that allows for the user to customize the parameters and query the high-quality curated datasets that contain atomic parameters and solar spectral irradiance data. Standard practice in the field has been to pick either a solar minimum or solar maximum representative spectrum, whichever best represents the Sun's activity level on the day of observation, and calculate the _g_-factor for the atomic species of interest with that spectrum. However, due to the large volume of cometary data taken in the last 50 years with a range of UV assets, where the change in solar flux can drastically change the fluorescence efficiency, we have decided to implement an API call to the NASA/NOAA/LASP LiSiRD Database, where solar spectral data can be retrieved directly. This allows the user of _gfactor_ to input a specific date, heliocentric distance and velocity, and retrieve a relevant solar spectrum for the day of observation that can be used to calculate fluorescence efficiencies. 

Our atomic fluorescence calculator, _gfactor_, makes use of publicly available APIs provided by NASA, NOAA, and NIST to calculate the photon emission efficiency in a solar radiation field, colloquially known as a _g_-factor, given a particular heliocentric distance and velocity by the user. This Python package was designed for use in cometary science, where the wide range of applicable heliocentric distances and velocities can drastically affect the fluorescence efficiency via the Swings effect (Swings 1941). However, there are many astrophysical applications where accurate determination of the fluorescence efficiency is required (Mercury and Moon exospheres, thin atmospheres on the Galilean satellites, etc.) and we hope that our Python packages empowers more early career researchers to probe the variability in these rates, even when the literature has historically kept them static. 

*describe the scaling methodology*

*describe the astropy units*

_gfactor_ is broken into three main classes, the _gfactorcalc_, _gfactoratomic_, and _gfactorsolar_, which are designed to perform the calculations, retrieve the queried atomic constants from the NIST Atomic Spectra Database (<https://www.nist.gov/pml/atomic-spectra-database>), and retrieve, trim, and scale the solar spectral irradiance data from LiSIRD (<https://lasp.colorado.edu/lisird/>). Benchmark tests have been carried out against the $g$-factors published in Killen et al. 2022, which were done for Mercury's exospheric species and orbital properties. This publication has the most transparent methodology and solar spectral dataset in the literature, allowing us to test our spectral scaling methodology; most comet literature does not report the scaling method used to adjust a solar spectral atlas dataset to a given day, making it necessary for us to use Killen et al. 2022's approach of implementing the solar spectrum for a specific day from the LiSIRD database.  

# Acknowledements

This work was supported by funds from Space Telescope Science Institute grant AR-17031 and the Auburn University College of Science and Mathematics Undergraduate Research Fellowship. 

# References

Killen, Rosemary M., Ronald J. Vervack Jr, and Matthew H. Burger. "Updated photon scattering coefficients (g_values) for Mercury’s exospheric species." The Astrophysical Journal Supplement Series 263.2 (2022): 37.

Swings, Polydore. "Complex structure of cometary bands tentatively ascribed to the contour of the solar spectrum." Lick observatory bulletin 19 (1941).

