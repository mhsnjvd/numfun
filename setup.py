import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numfun", 
    version="0.0.1",
    author="Mohsin Javed",
    author_email="mhsnjvd@gmail.com",
    description="Numerical computing with functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mhsnjvd/funpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
