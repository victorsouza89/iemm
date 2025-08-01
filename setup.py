from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="iemm",
    version="0.1.0",
    author="Victor Souza",
    author_email="victorflosouza@gmail.com",
    description="Iterative Evidential Mistakeness Minimization - Explainable Evidential Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/victorsouza89/iemm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "schemdraw>=0.11",
    ],
    package_data={
        "iemm": ["py.typed"],
    },
    keywords="belief functions, machine learning, uncertainty, explainable AI, XAI, evidential clustering",
)
