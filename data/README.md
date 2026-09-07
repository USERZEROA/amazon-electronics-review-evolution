# Data

The raw and processed datasets are not included in this repository because they are too large for GitHub.

The project uses the Electronics category from the Amazon Reviews'23 dataset:

https://amazon-reviews-2023.github.io/

Download the review file and place it at:

`data/raw/Electronics.jsonl.gz`

Then run:

```powershell
python preprocess.py