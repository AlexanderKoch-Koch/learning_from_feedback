from setuptools import find_packages, setup

# Required dependencies
required = [
    # Please keep alphabetized
    'torch',
    'numpy',
    'ray'
]

setup(
    name='learning_from_feedback',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
