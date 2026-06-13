Tutorials
=========

A read-in-order tour of the library. Each tutorial is a runnable
``.py`` file with copy-pasteable sections — no notebooks required, but
every file opens cleanly in `Google Colab
<https://colab.research.google.com/github/rlogger/bayes-hdc/blob/main/tutorials/01_quickstart.py>`_
too.

Available now
-------------

``01_quickstart.py``
    Installation to first prediction in about ninety seconds: a first
    Gaussian hypervector, an iris classifier, temperature scaling and
    conformal prediction, and anomaly detection in five lines.

``02_anomaly_detection.py``
    Calibrated one-shot anomaly detection from first principles. Shows
    the coverage guarantee empirically over 200 splits, the drift of a
    naive threshold versus the conformal detector, multi-VSA mode
    (MAP / BSC / HRR), a streaming twist, and a tabular fraud-style
    demo. This is the one to read after the quickstart.

``03_sequences.py``
    Sequence encoding from first principles. Builds an item codebook,
    encodes and retrieves with the flat ``Sequence`` and the chunked
    ``HierarchicalSequence``, then sweeps sequence length to show why
    the hierarchical variant stays near-perfect at ``T = 800`` where
    flat permute-bundle collapses to about 31 % retrieval.

In progress
-----------

Files land as they are written; the numbering reserves reading order.

- ``03_calibration_and_coverage.py`` — ECE / MCE, reliability curves,
  split-conformal classification and regression with coverage audits.
- ``04_resonator_factorisation.py`` — deterministic and probabilistic
  resonator networks for decoding bound compositions.
- ``05_real_data_eeg.py`` — seizure detection on a real EEG benchmark,
  end-to-end with calibrated probabilities.

Tutorials live under ``tutorials/`` in the repository. The
``examples/`` directory is a complementary cookbook: each file solves
one applied problem and is meant to be read in isolation.
