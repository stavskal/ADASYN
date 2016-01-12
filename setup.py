from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn',
    ]

setup(name='ADASYN',
      version='0.1',
      description='Python module for synthetic oversampling of skewed datasets',
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          ],
      author='Stavrianos Skalidis',
      author_email='stavrianos_@hotmail.com',
      url='https://github.com/stavskal/ADASYN',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )