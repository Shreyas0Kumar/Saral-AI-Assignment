"""Version stamps for every artifact this repo emits.

These are compared at runtime (``GET /health`` refuses to report healthy if the
signals in the database were written by a different ``SIGNALS_VERSION``) and are
written into the run manifest so a metrics number can be traced back to the code
that produced it.
"""

SIGNALS_VERSION = "1.0.0"
SCORING_VERSION = "1.0.0"
LEXICON_VERSION = "1.0.0"
REASON_VOCAB_VERSION = "1.0.0"
