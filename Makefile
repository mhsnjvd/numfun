# FORMAT ---------------------------------------------------------------------------------------------------------------
docformatter:
	docformatter numfun/ tests/ --wrap-summaries=120 --wrap-descriptions=120

isort:
	isort numfun/ tests/ -m 4 -l 120

fmt: docformatter isort

# LINT -----------------------------------------------------------------------------------------------------------------
docformatter-check:
	docformatter --check numfun/ tests/ --wrap-summaries=120 --wrap-descriptions=120

isort-check:
	isort --check-only numfun/ tests/ -m 4 -l 120

flake8:
	flake8 . --config=build-support/.flake8

bandit:
	bandit -r . --configfile build-support/.bandit.yml

pylint:
	pylint numfun/ tests/ --rcfile=build-support/.pylintrc

lint: flake8 bandit pylint

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
pre-commit: mypy flake8

check-all: mypy lint
