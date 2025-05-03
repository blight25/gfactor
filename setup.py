from setuptools import setup, find_packages

# List of requirements
requirements = ['requests',
                'tqdm', 
                'pandas',
                'matplotlib',
                'scipy',
                'specutils', 
                'astropy', 
                'lmfit',
                'pytest']


print(find_packages())

# Package (minimal) configuration
setup(
    name="gfactor",
    version="1.0.0",
    description="Accurate calculation of atomic fluorescence efficiencies using up-to-date atomic and solar data.",
    author="John Noonan, Ben Lightfoot, Steve Bromley, etc.",
    author_email="jnoonan@auburn.edu, bcl0025@auburn.edu",
    packages=find_packages(),  # __init__.py folders search
    install_requires=requirements
)