"""
Dataset Downloader and Setup Helper for Algonauts 2023 / Natural Scenes Dataset (NSD).

Provides official download instructions and setup instructions for Subjects 1-8.
"""

import os
import sys
import argparse


def print_official_download_guide(subject_id: str = "subj01"):
    """Prints step-by-step instructions for downloading the full 7T Algonauts 2023 NSD data."""
    print("=" * 70)
    print("📥 HOW TO DOWNLOAD REAL ALGONAUTS 2023 (NSD) DATA")
    print("=" * 70)
    print(f"""
The official Algonauts 2023 dataset is hosted on the Challenge Portal & AWS S3.

Download via Official Algonauts Challenge Portal:
--------------------------------------------------
1. Visit the challenge portal:
   http://algonautsproject.csail.mit.edu/download.html
2. Download '{subject_id}.zip' (approx 2.2 GB).
3. Extract '{subject_id}.zip' directly into:
   data/raw/algonauts_2023/

   Your folder structure should look like:
   data/raw/algonauts_2023/{subject_id}/
   ├── training_split/
   │   ├── training_images/
   │   │   ├── train-0001_nsd-00001.png
   │   │   └── ... (9,841 stimuli images)
   │   └── training_fmri/
   │       ├── lh_training_fmri.npy
   │       └── rh_training_fmri.npy
   └── roi_masks/
       ├── lh.V1.npy
       ├── rh.V1.npy
       └── ...

If running on Kaggle:
---------------------
Attach the Algonauts 2023 dataset directly as an Input:
- Input path: /kaggle/input/algonauts-2023-challenge/{subject_id}
""")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or setup Algonauts 2023 data")
    parser.add_argument("--subject", type=str, default="subj01", help="Subject ID (subj01-subj08)")
    args = parser.parse_args()

    print_official_download_guide(subject_id=args.subject)
