import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="rich_logger",
        version="0.1.3",
        author="Perceval Wajsburt",
        author_email="perceval.wajsburt@sorbonne-universite.fr",
        license='BSD 3-Clause',
        description="Table logger using Rich, aimed at Pytorch Lightning logging",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/percevalw/rich_logger",
        packages=setuptools.find_packages(),
        package_data={},
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires=[
            'rich~=9.10.0',
        ]
    )
