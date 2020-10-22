# # Make sure to set the below in your .bashrc
# PYENV=<path to your environment>  # e.g. PYENV="/d/Anaconda3/envs/ws"
# export PATH="$PYENV/Scripts":"$PYENV/bin":$PATH

# FORMAT ---------------------------------------------------------------------------------------------------------------
docformatter:
	docformatter -r . --in-place --wrap-summaries=120 --wrap-descriptions=120

isort:
	isort -rc numfun/ tests/ -m 4 -l 120

fmt: docformatter isort

# LINT -----------------------------------------------------------------------------------------------------------------
docformatter-check:
	docformatter -r . --check --wrap-summaries=120 --wrap-descriptions=120

isort-check:
	isort --check-only -rc numfun/ tests/ -m 4 -l 120

flake8:
	flake8 . --config=build-support/.flake8

bandit:
	bandit -r . --configfile build-support/.bandit.yml

pylint:
	pylint numfun/ tests/ --rcfile=build-support/.pylintrc

lint: flake8 bandit pylint docformatter-check

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
