# Import libraries
import pandas as pd
import numpy as np
import zipfile
import gdown

# Download and extract the dataset
url = 'https://drive.google.com/uc?id=1X3IpoYxAJHIBlyG6QQ_rWhSGiF5E1aL8'
output = '/content/dataset-ml-25m.zip'
gdown.download(url, output, quiet=False)

with zipfile.ZipFile('/content/dataset-ml-25m.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')