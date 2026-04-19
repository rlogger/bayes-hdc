Models Module
=============

The ``bayes_hdc.models`` module provides classification and learning algorithms.

CentroidClassifier
------------------

.. autoclass:: bayes_hdc.models.CentroidClassifier
   :members:
   :undoc-members:

Example::

   from bayes_hdc import MAP, CentroidClassifier
   import jax
   import jax.numpy as jnp

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   # Create classifier
   classifier = CentroidClassifier.create(
       num_classes=10,
       dimensions=10000,
       vsa_model=model
   )

   # Train
   train_hvs = model.random(key, (100, 10000))
   train_labels = jax.random.randint(key, (100,), 0, 10)
   classifier = classifier.fit(train_hvs, train_labels)

   # Predict
   test_hvs = model.random(key, (20, 10000))
   predictions = classifier.predict(test_hvs)

   # Evaluate
   test_labels = jax.random.randint(key, (20,), 0, 10)
   accuracy = classifier.score(test_hvs, test_labels)

LVQClassifier
-------------

.. autoclass:: bayes_hdc.models.LVQClassifier
   :members:
   :undoc-members:

RegularizedLSClassifier
----------------------

.. autoclass:: bayes_hdc.models.RegularizedLSClassifier
   :members:
   :undoc-members:

AdaptiveHDC
-----------

.. autoclass:: bayes_hdc.models.AdaptiveHDC
   :members:
   :undoc-members:

Example::

   from bayes_hdc import AdaptiveHDC

   classifier = AdaptiveHDC.create(
       num_classes=10,
       dimensions=10000,
       vsa_model=model
   )

   # Iterative training
   classifier = classifier.fit(
       train_hvs,
       train_labels,
       epochs=10,
       learning_rate=0.1
   )
