# GitHub Upload Checklist

Before uploading or immediately after creating the GitHub repository:

- Replace `USERNAME/REPOSITORY` in `CITATION.cff` with the real GitHub URL.
- Decide whether to add an open-source license.
- Run `python3 scripts/smoke_test.py`.
- Run `python3 scripts/generate_figures.py`.
- Confirm `results/summary.txt` matches the manuscript values.
- Create a GitHub release after acceptance or final submission.
- Archive the release with Zenodo if a permanent DOI is needed.
- Insert the GitHub or Zenodo URL into the manuscript data/code statement.
