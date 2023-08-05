import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="repsys-framework",
    version="0.4.1",
    author="Jan Safarik",
    author_email="safarj10@fit.cvut.cz",
    description="Framework for developing and analyzing recommender systems.",
    url="https://github.com/cowjen01/repsys",
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
    packages=["repsys"],
    license="GPLv3",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "repsys=repsys.__main__:main",
        ],
    },
    install_requires=[
        "click==8.1.6",
        "sanic==22.6.2",
        "coloredlogs==15.0.1",
        "numpy==1.24.4",
        "scipy==1.10.1",
        "pandas==2.0.3",
        "bidict==0.22.1",
        "scikit-learn==1.3.0",
        "umap-learn==0.5.3",
        "websockets==10.4",
    ],
    extras_require={
        # pynndescent needs a fixed version 0.5.8 due to the issue:
        # https://github.com/lmcinnes/pynndescent/issues/218
        "pymde": ["pymde==0.1.18", "pynndescent==0.5.8"],
    },
    zip_safe=False,
)
