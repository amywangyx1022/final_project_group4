"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ, getcwd, path
from pathlib import Path

from colorama import Fore, Style, init

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter

from settings import config

try:
    in_slurm = environ["SLURM_JOB_ID"] is not None
except:
    in_slurm = False


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        doit_mark = stuff.split(" ")[0].ljust(2)
        task = " ".join(stuff.split(" ")[1:]).strip() + "\n"
        output = (
            Fore.GREEN
            + doit_mark
            + f" {path.basename(getcwd())}: "
            + task
            + Style.RESET_ALL
        )
        self.outstream.write(output)


if not in_slurm:
    DOIT_CONFIG = {
        "reporter": GreenReporter,
        # other config here...
        # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
        "backend": "sqlite3",
        "dep_file": "./.doit-db.sqlite",
    }
else:
    DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
init(autoreset=True)


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
PUBLISH_DIR = config("PUBLISH_DIR")


## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --log-level WARN --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --log-level WARN --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --log-level WARN --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull_bloomberg():
    """ """
    file_dep = [
        "./src/settings.py",
        "./src/pull_bloomberg.py"
    ]
    targets = [
        DATA_DIR / "dividend_data.parquet",
        DATA_DIR / "index_data.parquet",
        DATA_DIR / "dividend_futures_data.parquet"
    ]

    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_bloomberg.py"
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }


def task_clean_bloomberg_data():
    """ """
    file_dep = [
        "./src/settings.py",
        "./src/clean_data.py"
    ]
    targets = [
        DATA_DIR / "cleaned_current_index_data.parquet",
        DATA_DIR / "cleaned_current_dividend_data.parquet",
        DATA_DIR / "calculated_current_yields.parquet",
        DATA_DIR / "combined_current_data.parquet",
       
    ]

    return {
        "actions": [
            "ipython ./src/clean_data.py"
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_replicate_figure_1():
    """ """
    file_dep = [
        "./src/settings.py",
        "./src/figure1_replicate.py",
        DATA_DIR/"clean"/"index_data_clean.parquet"
    ]
    targets = [
         OUTPUT_DIR / "figures" / "figure_1.png"
    ]

    return {
        "actions": [
            "ipython ./src/figure1_replicate.py"
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }
    

def task_replicate_table_1():
    """ """
    file_dep = [
        "./src/settings.py",
        "./src/TABLE1_replication.py",
         DATA_DIR / "clean" / "merged_dividend_data_quarterly.parquet",
        
    ]
    targets = [
        OUTPUT_DIR / "tables" / "table1_results.tex"
    ]

    return {
        "actions": [
            "ipython ./src/TABLE1_replication.py"
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_replicate_figure_5():
    """ """
    file_dep = [
        "./src/settings.py",
        "./src/figure5_replicate.py",
         "./_output/figures/forecast_paper_dividend_growth.parquet",
        "./_output/figures/forecast_updated_dividend_growth.parquet",
       
    ]
    targets = [
        OUTPUT_DIR / "figures" / "paper_figure5_panel_a.png",
        OUTPUT_DIR / "figures" / "paper_figure5_panel_b.png",
        OUTPUT_DIR / "figures" / "updated_figure5_combined.png"
    ]

    return {
        "actions": [
            "ipython ./src/figure5_replicate.py"
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }
    



def task_summary_stats():
    """ """
    file_dep = ["./src/additional_stats_table.py"]
    file_output = [
        OUTPUT_DIR/"tables"/"additional_stats.tex",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "ipython ./src/additional_stats_table.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    } 



notebook_tasks = {

    "brief_tour.ipynb": {
        "file_dep": [
            "./src/pull_bloomberg.py",
             "./src/clean_data.py",
            "./src/calc_functions.py",
            "./src/figure1_replicate.py",
            "./src/TABLE1_replication.py",
            "./src/figure5_replicate.py",
            "./src/additional_stats_table.py",
            OUTPUT_DIR / "figures" /"paper_figure5_panel_a.png",
            OUTPUT_DIR / "figures" /"paper_figure5_panel_b.png",
            OUTPUT_DIR / "figures"/"updated_figure5_combined.png",
        ],
        "targets": [
        ],
    },
}


def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./src") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_file(
                    Path("./src") / f"{notebook_name}.ipynb",
                    OUTPUT_DIR / f"{notebook_name}.ipynb",
                    mkdir=True,
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                OUTPUT_DIR / f"{notebook_name}.ipynb",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on


# ###############################################################
# ## Task below is for LaTeX compilation
# ###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/project_report.tex",
        "./_output/tables/table1_results.tex",
        "./_output/tables/additional_stats.tex",
        "./reports/my_article_header.sty",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
    ]
    targets = [
        "./reports/project_report.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/project_report.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/project_report.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }
