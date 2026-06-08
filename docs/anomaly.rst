Anomaly detection
=================

The ``bayes_hdc.anomaly`` module provides calibrated one-class anomaly
detection. You fit on "normal" data and get back split-conformal
p-values: under exchangeability the false-positive rate is controlled
at a target :math:`\alpha` with a finite-sample guarantee, no
distributional assumptions and no hand-tuned threshold.

The nonconformity score is a distance to the normal region in
hypervector space (``1 - cosine`` to the centroid by default, a k-NN
mean, or Hamming distance for BSC). The conformal layer turns that
score into a p-value via the standard
:math:`(1 + \#\{s_\text{cal} \ge s_\text{query}\}) / (n + 1)`
construction (Laxhammar 2014; Bates, Candès, Lei & Romano 2023).

Both classes are registered JAX pytrees, so ``score`` / ``pvalue`` /
``predict`` compose under ``jit`` and ``vmap``, and ``score`` is
differentiable.

Scorer
------

.. autoclass:: bayes_hdc.anomaly.HDCAnomalyScorer
   :members:
   :undoc-members:
   :member-order: bysource

Detector
--------

.. autoclass:: bayes_hdc.anomaly.ConformalAnomalyDetector
   :members:
   :undoc-members:
   :member-order: bysource

End-to-end pipeline
-------------------

.. autofunction:: bayes_hdc.anomaly.fit_anomaly_pipeline

See also
--------

- :doc:`tutorials` — ``02_anomaly_detection.py`` walks through the
  coverage guarantee, the conformal-vs-naive-threshold comparison,
  multi-VSA mode, and a tabular fraud-style demo.
- ``examples/anomaly_detection_intrusion.py`` and
  ``examples/anomaly_detection_sensors.py`` — applied demos.
