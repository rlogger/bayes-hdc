Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from bayes_hdc import MAP

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)
   k1, k2 = jax.random.split(key)

   x = model.random(k1, (10000,))
   y = model.random(k2, (10000,))

   bound = model.bind(x, y)
   bundled = model.bundle(jnp.stack([x, y]), axis=0)
   sim = model.similarity(x, y)

Classification Pipeline
-----------------------

.. code-block:: python

   import jax
   from bayes_hdc import MAP, RandomEncoder, CentroidClassifier

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   encoder = RandomEncoder.create(
       num_features=20, num_values=10, dimensions=10000,
       vsa_model=model, key=key,
   )
   classifier = CentroidClassifier.create(
       num_classes=5, dimensions=10000, vsa_model=model,
   )

   data = jax.random.randint(key, (100, 20), 0, 10)
   labels = jax.random.randint(key, (100,), 0, 5)
   encoded = encoder.encode_batch(data)
   classifier = classifier.fit(encoded, labels)
   accuracy = classifier.score(encoded, labels)

Key Concepts
------------

* **Hypervectors**: High-dimensional vectors (typically 10,000 dimensions)
* **Binding**: Combines two hypervectors into a dissimilar result
* **Bundling**: Superposes multiple hypervectors into a similar result
* **Similarity**: Measures relatedness between hypervectors
