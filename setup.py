import setuptools

setuptools.setup(
    name="dsvae",
    version="0.0.1",
    author="Max Kraan",
    description="VAE-based drum machine sequence engine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
