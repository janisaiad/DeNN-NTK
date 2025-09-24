# Deeper or wider ?

## Description
We investigate wether deeper or wider is better for NNs optimization under the NTK's perspective
In parallel, other experiments with the same amount of content related to low rank weights NNs are available at [github.com/janisaiad/MMNN](https://github.com/janisaiad/MMNN).


## Report and contributions

A complete report has been done relating the theory and why all of those experiments have been done, especially getting the result page 62.

You can read it in refs/report/report.pdf

If there is an issue you can compile again the report, first run the following command:
```bash
pdflatex -interaction=nonstopmode -output-directory=refs/wk14-report-and-transformers/stage_3a_le_bon_template/ refs/wk14-report-and-transformers/stage_3a_le_bon_template/main3.tex
```

Then, open the report in the `refs/wk14-report-and-transformers/stage_3a_le_bon_template/main.pdf` file.

The work is currently in progress (and will remain during a long amout of time because this topic is very rich and new) and daily/weekly commits are done.

Feel free to contact me for any information.

## Installation

To install dependencies using uv, follow these steps:

1. Install uv:
   
   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, you can install uv using:
   - pipx (recommended): `pipx install uv`
   - pip: `pip install uv`
   - Homebrew: `brew install uv`
   - WinGet: `winget install --id=astral-sh.uv -e`
   - Scoop: `scoop install main/uv`

2. Using uv in this project:

   - Initialize a new virtual environment:
   ```bash
   uv venv
   ```

   - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

   - Install dependencies from requirements.txt:
   ```bash
   uv add -r requirements.txt
   ```


   - Add a new package:
   ```bash
   uv add package_name
   ```

   - Remove a package:
   ```bash
   uv remove package_name
   ```

   - Update a package:
   ```bash
   uv pip install --upgrade package_name
   ```

   - Generate requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

   - List installed packages:
   ```bash
   uv pip list
   ```

## Warning

If you're using macOS or Python 3, replace `pip` with `pip3` in line 1 of ```launch.sh```

Replace with your project folder name (which means the name of the library you are deving) in :```tests/test_env.py: ```


