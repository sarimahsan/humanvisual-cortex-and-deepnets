"""
Dataset Downloader and Setup Helper for Algonauts 2023 / NSD.

Provides direct instructions and automated utilities to download Subject data:
1. From the Algonauts 2023 Official Challenge repository.
2. Or generate a local quick-start dataset for immediate laptop testing.
"""

import os
import sys
import argparse
import urllib.request
import zipfile

def setup_quickstart_data(output_dir: str = "data/raw/algonauts_2023", subject_id: str = "subj01", n_images: int = 150):
    """Generates a functional local dataset with real PNGs and fMRI response arrays."""
    from data.mock_data import generate_synthetic_dataset
    print(f"\n[QuickStart] Generating functional {subject_id} dataset in {output_dir}...")
    generate_synthetic_dataset(
        output_dir=output_dir,
        subject_id=subject_id,
        n_images=n_images,
        n_lh_vertices=1000,
        n_rh_vertices=1000,
        seed=42,
    )
    print(f"\n[Success] {subject_id} created with {n_images} images and 2,000 cortical vertices!")
    print(f"You can now run feature extraction and encoding models on it directly.\n")


def print_official_download_guide(subject_id: str = "subj01"):
    """Prints step-by-step instructions for downloading the full 7T Algonauts 2023 NSD data."""
    print("=" * 70)
    print("📥 HOW TO DOWNLOAD REAL ALGONAUTS 2023 (NSD) DATA")
    print("=" * 70)
    print(f"""
The official Algonauts 2023 dataset is hosted on the Challenge Portal & AWS S3.

Option A: Download via Official Algonauts Challenge Drive
---------------------------------------------------------
1. Visit the challenge portal:
   http://algonautsproject.csail.mit.edu/download.html
2. Download '{subject_id}.zip' (approx 2.2 GB).
3. Extract '{subject_id}.zip' directly into:
   data/raw/algonauts_2023/

   Your folder structure should look like:
   d:\\human-visual-cortex-and-deepnets\\data\\raw\\algonauts_2023\\{subject_id}\\
   ├── training_split/
   │   ├── training_images/
   │   │   ├── train-0001_nsd-00001.png
   │   │   └── ...
   │   └── training_fmri/
   │       ├── lh_training_fmri.npy
   │       └── rh_training_fmri.npy
   └── roi_masks/
       ├── lh.V1.npy
       ├── rh.V1.npy
       └── ...

Option B: Instant Local QuickStart (2 seconds)
-----------------------------------------------
If you want to run the full pipeline right now on your laptop without
waiting for a 2.2 GB download, run:

   python -m data.download --quickstart --subject {subject_id}
""")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or setup Algonauts 2023 data")
    parser.add_argument("--quickstart", action="store_true", help="Generate instant functional dataset")
    parser.add_argument("--subject", type=str, default="subj01", help="Subject ID (subj01-subj08)")
    parser.add_argument("--output_dir", type=str, default="data/raw/algonauts_2023", help="Destination folder")
    args = parser.parse_args()

    if args.quickstart:
        setup_quickstart_data(output_dir=args.output_dir, subject_id=args.subject)
    else:
        print_official_download_guide(subject_id=args.subject)
