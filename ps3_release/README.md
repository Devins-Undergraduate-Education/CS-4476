# PS3
Projection Matrix and Fundamental Matrix Estimation

# Setup
- Note: Refer to `PS3.pdf` for complete instructions.
- Install <a href="https://conda.io/miniconda.html">Miniconda</a> (python 3.10).
- Create a conda environment, using the appropriate command. On Windows, open the installed "Conda prompt" to run this command. On MacOS and Linux, you can just use a terminal window to run the command. Modify the command based on your OS ('linux', 'mac', or 'win'): `mamba env create -f ps3_env_<OS>.yml`
- This should create an environment named `ps3`. Activate it using the following Windows command: `activate ps3` or the following MacOS / Linux command: `mamba activate ps3`.
- Run `pip install -e .`
- Run the notebook using: `jupyter notebook ./ps3_code/ps3.ipynb`
- Generate the submission once you're finished using `python zip_submission.py --gt_username <your_username>`