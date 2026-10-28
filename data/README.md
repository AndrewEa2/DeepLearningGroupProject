# Dataset

Expected file: `enron_spam_data.csv` in this directory.

This repository uses Marcel Wiechmann's cleaned Enron-Spam CSV release: <https://github.com/MWiechmann/enron_spam_data>. Download `enron_spam_data.zip`, extract the CSV here, and keep the original file unchanged. The release describes `Subject`, `Message`, `Spam/Ham`, and `Date` columns, with `spam` / `ham` labels. Review the source README for provenance and its GPL-3.0 license.

The CSV is excluded from Git because it is source data. Store derived datasets and model outputs outside version control as well. Document any preprocessing and keep the original labels available for audit.
