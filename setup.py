from setuptools import find_packages, setup

setup(
    name='your_project_name',
    version='0.1.0',
    description='A short description of your project',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        'numpy',
        'pandas',
        'xgboost',
        'optuna',
        'seaborn',
        'scikit-learn',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
