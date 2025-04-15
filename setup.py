from setuptools import setup, find_packages

setup(
    name="multi-imprinting",
    version="0.1.0",
    packages=find_packages(),
    description="Multi-imprinting project for learning embeddings",
    python_requires=">=3.8",
    install_requires=[
        # Requirements will be read from requirements.txt by the Dockerfile
    ],
)
