import os
import re
import setuptools

NAME = "flow_matching"
DESCRIPTION = "Flow Matching for Generative Modeling"
REQUIRES_PYTHON = ">=3.9.0"

# for line in open("flow_matching/__init__.py"):
#    line = line.strip()
#    if "__version__" in line:
#        context = {}
#        exec(line, context)
#        VERSION = context["__version__"]

VERSION = None
with open("flow_matching/__init__.py") as f:
    for line in f:
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', line)
        if match:
            VERSION = match.group(1)
            break

if VERSION is None:
    raise RuntimeError("Unable to find version string in flow_matching/__init__.py")

readme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")

try:
    with open(readme_path) as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    packages=setuptools.find_packages(),
    extras_require={
        "dev": [
            "pre-commit",
            "black==22.6.0",
            "usort==1.0.4",
            "ufmt==2.3.0",
            "flake8==7.0.0",
            "pydoclint",
        ],
    },
    install_requires=["numpy", "torch", "torchdiffeq"],
    license="CC-by-NC",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
