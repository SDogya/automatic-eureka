"""Download every output file of a Kaggle kernel, page by page, with retries (run with the Python that has `kaggle`).

    python3 scripts/kaggle_pull_pages.py <owner/kernel> <dest> [regex]

The CLI's `kernels output` returns one page (20 files by default) unless given page tokens, and a transient SSL error
aborts it; this loops over pages of 200, retries each page, and skips files already downloaded (force=False), so a
re-run resumes. Default regex: analysis files only (csv, json, log, png); pass '.*' to include checkpoints.
"""

import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

kernel, dest = sys.argv[1], sys.argv[2]
pattern = sys.argv[3] if len(sys.argv) > 3 else r"\.(csv|json|log|png)$"
api = KaggleApi()
api.authenticate()
token, pages, files = None, 0, 0
while True:
    for attempt in range(6):
        try:
            got, token = api.kernels_output(kernel, dest, file_pattern=pattern, force=False, quiet=True,
                                            page_token=token, page_size=200)
            break
        except Exception as error:  # noqa: BLE001  (network: retry the same page)
            print(f"page {pages} attempt {attempt}: {type(error).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"giving up on page {pages}")
    pages += 1
    files += len(got or [])
    if not token:
        break
print(f"{kernel}: {pages} pages, {files} files matched '{pattern}'")
