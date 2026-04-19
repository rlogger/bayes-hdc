HDC Learning
============

After learning about representing and manipulating information in hyperspace, we can implement our first HDC classification model! This tutorial follows a structure similar to the `TorchHD classification guide <https://torchhd.readthedocs.io/en/latest/classification.html>`_.

We use a synthetic dataset to demonstrate the full pipeline: **Dataset** → **Encoding** → **Training** → **Testing**. For real datasets (e.g. MNIST), you would replace the data loading step with your dataset of choice.

Configuration and Imports
-------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from bayes_hdc import MAP
   from bayes_hdc.embeddings import RandomEncoder
   from bayes_hdc.models import CentroidClassifier

   # Configuration
   DIMENSIONS = 10000
   NUM_FEATURES = 20
   NUM_VALUES = 10
   NUM_CLASSES = 5
   N_TRAIN = 800
   N_TEST = 200

Datasets
--------

Generate or load your data. Here we use synthetic discrete features; for images (e.g. MNIST), you would use :class:`~bayes_hdc.embeddings.LevelEncoder` or :class:`~bayes_hdc.embeddings.ProjectionEncoder` instead of :class:`~bayes_hdc.embeddings.RandomEncoder`.

.. code-block:: python

   def generate_data(key, n_samples, n_features, n_classes):
       data = jax.random.randint(key, (n_samples, n_features), 0, NUM_VALUES)
       feature_sum = jnp.sum(data[:, :3], axis=1)
       labels = feature_sum % n_classes
       return data, labels

   key = jax.random.PRNGKey(42)
   k1, k2, k3 = jax.random.split(key, 3)

   train_data, train_labels = generate_data(k2, N_TRAIN, NUM_FEATURES, NUM_CLASSES)
   test_data, test_labels = generate_data(k3, N_TEST, NUM_FEATURES, NUM_CLASSES)

Encoding
--------

Define how to map raw data into hypervectors. For discrete features we use :class:`~bayes_hdc.embeddings.RandomEncoder`; for continuous values use :class:`~bayes_hdc.embeddings.LevelEncoder`, and for images use :class:`~bayes_hdc.embeddings.ProjectionEncoder`.

.. code-block:: python

   model = MAP.create(dimensions=DIMENSIONS)
   encoder = RandomEncoder.create(
       num_features=NUM_FEATURES,
       num_values=NUM_VALUES,
       dimensions=DIMENSIONS,
       vsa_model=model,
       key=k1,
   )

   # Encode training and test data into hypervectors
   train_hvs = encoder.encode_batch(train_data)
   test_hvs = encoder.encode_batch(test_data)

Training
--------

Create the classifier and fit it on encoded hypervectors. The :class:`~bayes_hdc.models.CentroidClassifier` performs single-pass learning by computing class centroids—no iterations or backpropagation.

.. code-block:: python

   classifier = CentroidClassifier.create(
       num_classes=NUM_CLASSES,
       dimensions=DIMENSIONS,
       vsa_model=model,
   )

   classifier = classifier.fit(train_hvs, train_labels)

Testing
-------

Evaluate on the test set by encoding samples and comparing to class prototypes via similarity (e.g. cosine similarity for MAP).

.. code-block:: python

   predictions = classifier.predict(test_hvs)
   accuracy = classifier.score(test_hvs, test_labels)

   print(f"Test accuracy: {accuracy:.2%}")

Running the Full Example
------------------------

A complete runnable script with synthetic data is in ``examples/classification_simple.py``:

.. code-block:: bash

   pip install -e ".[examples]"
   python examples/classification_simple.py

For MNIST-style image classification, use :class:`~bayes_hdc.embeddings.LevelEncoder` to encode pixel values and :class:`~bayes_hdc.embeddings.RandomEncoder` or position encodings for spatial structure—similar to the TorchHD encoder combining ``Random`` (position) and ``Level`` (value).
