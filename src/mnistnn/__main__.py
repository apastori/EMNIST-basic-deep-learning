"""Run EMNIST ensemble training and evaluation with ``python -m mnistnn``."""

import logging

from mnistnn.ensemble import train_ensemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)

train_ensemble()
