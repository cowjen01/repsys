import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="repsys",
    version="0.1.0",
    author="Jan Safarik",
    author_email="safarj10@fit.cvut.cz",
    description="A package for modeling and testing of recommendation systems.",
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
    packages=["repsys"],
    license="Apache License 2.0",
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "repsys=repsys.__main__:main",
        ],
    },
    install_requires=[
        "click==8.0.3",
        "sanic==21.9.3",
        "coloredlogs==15.0.1",
        "numpy==1.20.3",
        "scipy==1.7.3",
        "matplotlib==3.5.1",
        "pandas==1.3.5",
        "jax[cpu]==0.2.26",
        "pymde==0.1.14",
        "bidict==0.21.4"
    ],
    extras_require={
        "dev": ["black", "tox", "flake8", "wheel", "setuptools", "pytest"],
    },
    zip_safe=False,
)
