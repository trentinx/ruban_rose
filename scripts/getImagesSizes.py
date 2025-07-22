import zipfile
from PIL import Image
from io import BytesIO
import multiprocessing
import csv

ZIP_PATH = "data/BHI.zip"

def process_image(name):
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            with zf.open(name) as img_file:
                img = Image.open(BytesIO(img_file.read()))
                img.load()  # force le chargement
                width, height = img.size
                return (name, width, height)
    except Exception as e:
        return (name, "error", str(e))

def main(zip_path, output_csv=None, folder_filter="", num_workers=8):
    # Obtenir la liste des images
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [name for name in zf.namelist()
                 if name.lower().endswith(('.jpg', '.jpeg', '.png')) and name.startswith(folder_filter)]
    
    print(f"{len(names)} images à traiter avec {num_workers} workers...")

    # Multiprocessing pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(pool.imap_unordered(process_image, names, chunksize=100))

    # Enregistrement dans CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "width", "height"])
            for row in results:
                writer.writerow(row)
    else:
        for row in results:
            print(row)

if __name__ == "__main__":
    main(ZIP_PATH, output_csv="tailles_images.csv", folder_filter="", num_workers=8)
