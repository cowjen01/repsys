import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="repsys",
    version="0.1.0",
    author="Jan Safarik",
    author_email="safarj10@fit.cvut.cz",
    description="A package for modeling and testing recommendation systems.",
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
        "click",
        "sanic",
        "coloredlogs",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "jax[cpu]",
    ],
    extras_require={
        "dev": ["black"],
        "test": ["flake8", "pytest-cov", "pytest-sanic", "pytest"],
    },
)
