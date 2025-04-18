import urllib.request
import zipfile

url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
zip_path = "nature_12K.zip"
extract_dir = "./inaturalist_12K"

# Download
urllib.request.urlretrieve(url, zip_path)

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)