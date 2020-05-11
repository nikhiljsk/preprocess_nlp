import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='preprocess_nlp',  
     version='0.1',
     scripts=['preprocess_nlp'] ,
     author="Nikhil JSK",
     author_email="nikhiljsk98@gmail.com",
     description="A fast framework for pre-processing  (Cleaning text, Reduction of vocabulary, Feature extraction and Vectorization). Implemented with parallel processing using custom number of processes.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/nikhiljsk/preprocess_nlp",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )