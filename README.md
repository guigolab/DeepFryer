# DeepFryer

DeepFryer (Deep Learning Framework for your Expression RNA-seq data) is a package built in Python 3.6 (Van Rossum et al., 2009) that orbitates around Google’s Tensorflow backend (Abadi et al., 2016) and Keras API for the implementation of Deep Learning algorithms (Chollet et al., 2015). Among its functionalities it integrates Pandas, Pyarrow, Numpy, Scipy, Scikit-learn, Matplotlib (McKinney, 2011; Lowe, 2017; Walt et al., 2011; Jones et al., 2014; Pedregosa et al., 2011; Hunter, 2007) at the central core for the different steps during the analysis. This framework is organized in four big modules that cover data processing, analysis and correction of batch effects or unwanted variation, deep learning modeling and relevant information extraction from models with gene ontology analysis.

## Installation

```
pip install git+https://github.com/guigolab/DeepFryer.git
```

### Pre-requisites
You need to have installed Python 3.x.

In order to install DeepFryer you should have prepared an enviroment with *Keras >= 2.1.6* and *Tensorflow >= 1.7*.

To install Keras follow instructions in https://keras.io/#installation
To install Tensorflow follow instructions in https://www.tensorflow.org/install/

