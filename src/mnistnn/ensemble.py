"""Train and evaluate an ensemble of EMNIST MLP members."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

from mnistnn.mnist_neural_network import MNISTNeuralNetwork
from mnistnn.train_mnist import _normalize_images, load_emnist_arrays, train_mnist
from mnistnn.utils.images import shift_images

logger = logging.getLogger(__name__)

# EMNIST Balanced mapping: digits, uppercase, then 11 distinct lowercase letters.
EMNIST_BALANCED_CHARS: str = (
    '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
)
MODELS_DIR: Path = Path(__file__).resolve().parents[2] / 'models'
DEFAULT_MEMBERS: int = 5
DEFAULT_BASE_SEED: int = 0
EVAL_BATCH_SIZE: int = 256
SHIFT_VIEWS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def _member_weight_path(member_number: int) -> Path:
    """Return the weight-file path for one ensemble member (1-based)."""
    return Path(MODELS_DIR, f'member_{member_number}.npz')


def _class_char(label: int) -> str:
    """Map a class index to its Balanced EMNIST character."""
    num_classes: int = len(EMNIST_BALANCED_CHARS)
    label_is_valid: bool = (label >= 0) and (label < num_classes)
    if label_is_valid:
        return EMNIST_BALANCED_CHARS[label]
    return str(label)


def _predict_proba_normalized(
    network: MNISTNeuralNetwork, images: np.ndarray, batch_size: int
) -> np.ndarray:
    """Return class probabilities for already-normalized flattened images."""
    probabilities: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch: np.ndarray = images[start : start + batch_size]
        probabilities.append(network.predict_proba(batch))
    return np.concatenate(probabilities, axis=0)


def _predict_proba_batches(
    network: MNISTNeuralNetwork, images: np.ndarray, batch_size: int
) -> np.ndarray:
    """Return class probabilities for a raw flattened image array."""
    return _predict_proba_normalized(
        network, _normalize_images(images), batch_size
    )


def _predict_proba_with_shifts(
    network: MNISTNeuralNetwork, images: np.ndarray, batch_size: int
) -> np.ndarray:
    """Average probabilities over the original image and four one-pixel shifts."""
    normalized: np.ndarray = _normalize_images(images)
    accumulated: np.ndarray | None = None
    for shift_y, shift_x in SHIFT_VIEWS:
        shifted: np.ndarray = shift_images(normalized, shift_y, shift_x)
        view_probs: np.ndarray = _predict_proba_normalized(
            network, shifted, batch_size
        )
        if accumulated is None:
            accumulated = view_probs
        else:
            # Element-wise sum of (N, 47) probability matrices across shift views.
            accumulated = np.add(accumulated, view_probs)
    if accumulated is None:
        raise RuntimeError('No shift views were evaluated.')
    # Average probabilities across shift views.
    return accumulated / float(len(SHIFT_VIEWS))


def evaluate_ensemble(
    weight_paths: list[Path],
    batch_size: int = EVAL_BATCH_SIZE,
) -> float:
    """Evaluate members and their averaged predictions on the test set."""
    if not weight_paths:
        raise ValueError('weight_paths must not be empty')
    _train_images: np.ndarray
    _train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    (
        _train_images,
        _train_labels,
        test_images,
        test_labels,
    ) = load_emnist_arrays()

    member_probs: list[np.ndarray] = []
    member_shift_probs: list[np.ndarray] = []
    for index, weight_path in enumerate(weight_paths):
        network: MNISTNeuralNetwork = MNISTNeuralNetwork()
        network.load_weights(weight_path)
        probs: np.ndarray = _predict_proba_batches(
            network, test_images, batch_size
        )
        shift_probs: np.ndarray = _predict_proba_with_shifts(
            network, test_images, batch_size
        )
        member_probs.append(probs)
        member_shift_probs.append(shift_probs)
        # Percentage of correctly predicted labels.
        member_accuracy: float = float(
            np.mean(np.argmax(probs, axis=1) == test_labels)
        )
        logger.info(
            'Member %d accuracy: %.2f%% (%s)',
            index + 1,
            member_accuracy * 100,
            weight_path.name,
        )

    # Average Probabilities as the ensemble prediction incrementally
    # show the benefits of ensemble
    for count in range(1, len(member_probs) + 1):
        # soft votes of the ensemble members by accumulating the probabilities
        averaged: np.ndarray = np.mean(member_probs[:count], axis=0)
        accuracy: float = float(
            np.mean(np.argmax(averaged, axis=1) == test_labels)
        )
        logger.info(
            'Ensemble accuracy without shift averaging (%d members): %.2f%%',
            count,
            accuracy * 100,
        )
    # Generate the final ensemble prediction average of five networks
    ensemble_shift_probs: np.ndarray = np.mean(member_shift_probs, axis=0)
    ensemble_predictions: np.ndarray = np.argmax(ensemble_shift_probs, axis=1)
    headline_accuracy: float = float(
        # Compare the ensemble prediction with the test labels
        np.mean(ensemble_predictions == test_labels)
    )
    logger.info(
        'Ensemble accuracy with shift averaging (%d members): %.2f%%',
        len(member_shift_probs),
        headline_accuracy * 100,
    )

    confusion_counts: Counter[tuple[int, int]] = Counter()
    num_examples: int = len(test_labels)
    if len(ensemble_predictions) != num_examples:
        raise ValueError(
            'test_labels and ensemble_predictions must have the same length'
        )
    for index in range(num_examples):
        true_label: int = int(test_labels[index])
        predicted_label: int = int(ensemble_predictions[index])
        if true_label != predicted_label:
            confusion_counts[(true_label, predicted_label)] += 1

    logger.info('Top 15 confused true/predicted pairs:')
    top_confusions: list[tuple[tuple[int, int], int]] = (
        confusion_counts.most_common(15)
    )
    for confusion_entry in top_confusions:
        label_pair: tuple[int, int] = confusion_entry[0]
        count: int = confusion_entry[1]
        true_label: int = label_pair[0]
        predicted_label: int = label_pair[1]
        logger.info(
            '  %s -> %s: %d',
            _class_char(true_label),
            _class_char(predicted_label),
            count,
        )
    return headline_accuracy


def train_ensemble(
    members: int = DEFAULT_MEMBERS,
    base_seed: int = DEFAULT_BASE_SEED,
    visualize: bool = True,
) -> float:
    """Train missing ensemble members, then evaluate the full ensemble."""
    if members <= 0:
        raise ValueError('members must be greater than zero')

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weight_paths: list[Path] = []
    trained_in_session: bool = False

    for member_number in range(1, members + 1):
        weight_path: Path = _member_weight_path(member_number)
        weight_paths.append(weight_path)
        if weight_path.exists():
            logger.info(
                'Skipping member %d; weights already exist at %s',
                member_number,
                weight_path,
            )
            continue

        # Keep seeds aligned with the original 0-based training order.
        member_seed: int = base_seed + (member_number - 1)
        show_plots: bool = visualize and not trained_in_session
        logger.info(
            'Training ensemble member %d/%d with seed=%d',
            member_number,
            members,
            member_seed,
        )
        network: MNISTNeuralNetwork = train_mnist(
            seed=member_seed,
            visualize=show_plots,
            append_logs=True,
        )
        network.save_weights(weight_path)
        trained_in_session = True
        logger.info('Saved member %d weights to %s', member_number, weight_path)

    return evaluate_ensemble(weight_paths)


__all__ = ['evaluate_ensemble', 'train_ensemble']
