# # Make sure to set the below in your .bashrc
# PYENV=<path to your environment>  # e.g. PYENV="/d/Anaconda3/envs/ws"
# export PATH="$PYENV/Scripts":"$PYENV/bin":$PATH

# FORMAT ---------------------------------------------------------------------------------------------------------------
docformatter:
	docformatter -r . --in-place --wrap-summaries=120 --wrap-descriptions=120

isort:
	isort numfun/ tests/ -m 4 -l 120

fmt: docformatter isort

# LINT -----------------------------------------------------------------------------------------------------------------
docformatter-check:
	docformatter -r . --check --wrap-summaries=120 --wrap-descriptions=120 || \
	docformatter -r . --wrap-summaries=120 --wrap-descriptions=120  # Show diff if check failed

isort-check:
	isort --check-only numfun/ tests/ -m 4 -l 120 || \
	isort --diff --color numfun/ tests/ -m 4 -l 120  # Show diff if check failed

flake8:
	flake8 . --config=build-support/.flake8

pylint:
	pylint numfun/ tests/ --rcfile=build-support/.pylintrc

lint: flake8 docformatter-check isort-check # pylint

# TYPE CHECK -----------------------------------------------------------------------------------------------------------
mypy:
	mypy numfun/ tests/ --config-file build-support/mypy.ini

# CLEAN ----------------------------------------------------------------------------------------------------------------
clean-pyc:
	find . -name *.pyc | xargs rm -f && find . -name *.pyo | xargs rm -f;

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info


# OTHERS  --------------------------------------------------------------------------------------------------------------
pre-commit: mypy flake8 isort docformatter

check-all: mypy lint
