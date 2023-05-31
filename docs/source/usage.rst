Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install causal-imputation-test

Test the performance of causal missing-data imputation
----------------

Two test

.. function:: oneshot.test(Z, X, M, Y, G, L = 10000, verbose = False)



.. function:: retrain.test(Z, X, M, Y, G, L = 10000, verbose = False)

For example:

>>> import oneshot
>>> import simulation
>>> from sklearn.experimental import enable_iterative_imputer
>>> from sklearn.impute import IterativeImputer
>>> import xgboost as xgb
>>> 
>>> DataGen = simulation.DataGenerator(N = 100)
>>> Z, X, Y, M, S = DataGen.GenerateData()
>>> G = IterativeImputer(estimator = xgb.XGBRegressor())
>>> print(oneshot.test(Z, X, Y, M, G))

