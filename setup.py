from setuptools import setup, find_packages

setup(
    name="imprinting",
    version="0.1.0",
    packages=find_packages(),
    description="Code for paper 'Robust Weight Imprinting: Insights from Neural Collapse and Proxy-Based Aggregation'",
    python_requires=">=3.8",
    install_requires=[
        # Requirements will be read from requirements.txt by the Dockerfile
    ],
)
