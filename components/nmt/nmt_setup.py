from setuptools import setup

setup(
    name="nmt",
    version="0.0.1",
    description="Nonparameteric Machine Teacher",
    author="S.T.S Luo, Chen Zhang, Jason Chun Lok Li",
    author_email="stevents.luo@mail.utoronto.ca",
    packages=["nmt"],
    install_requires=[
        "torch",
        "numpy",
        "pillow"
    ]
)