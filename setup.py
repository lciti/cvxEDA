from setuptools import setup

setup(
    name="cvxEDA",
    version="0.1.0",
    description="cvxEDA python implemention, by Greco et al.",
    url="https://github.com/LeonardoAlchieri/cvxEDA",
    author="Leonardo Alchieri",
    author_email="leonardo@alchieri.eu",
    license="GPL-3.0 License",
    packages=["cvxEDA"],
    install_requires=["numpy", "cvxopt"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
