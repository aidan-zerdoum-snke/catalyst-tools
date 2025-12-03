# SnkeOS Internal
#
# Copyright (C) 2024-2025 Snke OS GmbH, Germany. All rights reserved.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import importlib
import inspect

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import pathlib
import traceback

sys.path.insert(0, os.path.abspath(".."))


def get_line_number(function):
    return inspect.getsourcelines(function)[1]


def get_python_object_line_number(module: str, attribute: str) -> int:
    """Returns the Python object corresponding to the given module and
    attribute."""
    line_number = 1
    try:
        obj = importlib.import_module(module)

        for attribute_name in attribute.split("."):
            obj = getattr(obj, attribute_name)
            try:
                line_number = inspect.getsourcelines(obj)[1]
            except Exception:
                return line_number

    except AttributeError:
        pass
    return line_number


def linkcode_resolve(domain, info):
    repo_name = str(pathlib.Path(__file__).parents[1].name)
    if domain != "py":
        return None
    if not info["module"]:
        return None
    try:
        line_number = get_python_object_line_number(info["module"], info["fullname"])
    except Exception:
        print(info)
        traceback.print_exc()
        exit()
    filename = info["module"].replace(".", "/")
    return f"https://github.com/snkeos/{repo_name}/blob/main/src/{filename}.py#L{line_number}"


project = str(pathlib.Path(__file__).parents[1].name)
author = "SnkeOS GmbH"
release = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
]


autosummary_generate = True
autoclass_content = "both"
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "pydata_sphinx_theme"
html_favicon = "_images/favicon.ico"
html_theme_options = {
    "logo": {
        "text": project,
        "image_light": "_images/logo_light.png",
        "image_dark": "_images/logo_dark.png",
    }
}
