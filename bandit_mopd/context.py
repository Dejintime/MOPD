"""Operational definitions for the seven coordinates of manuscript Eq.9.

The manuscript does not specify a domain estimator, advantage, agreement, or
cost unit. These definitions are explicit, shared by both scoring backends, and
saved with each run. No labels from the current example enter predict().
"""
from collections import Counter
import hashlib
import re

import numpy as np


CONTEXT_FEATURES = (
    'domain_probability', 'teacher_mean_log_probability', 'student_teacher_kl',
    'teacher_entropy', 'teacher_student_advantage', 'teacher_agreement',
    'teacher_cost',
)


def configure_context(cfg):
    settings = cfg.setdefault('context', {})
    defaults = dict(schema='paper_eq9_v1', domain_estimator='hashed_multinomial_nb',
                    hash_bins=32768, smoothing=1.0, domain_prior='uniform',
                    advantage='mean_log_probability_difference',
                    agreement='mean_pairwise_bhattacharyya',
                    cost='measured_prefill_seconds')
    for key, value in defaults.items():
        settings.setdefault(key, value)
        if key not in ('hash_bins', 'smoothing', 'cost') and settings[key] != value:
            raise ValueError(f'Unsupported context.{key}: {settings[key]}')
    if settings['cost'] not in ('measured_prefill_seconds', 'relative_excess_prefill'):
        raise ValueError(f"Unsupported context.cost: {settings['cost']}")
    if (not isinstance(settings['hash_bins'], int) or isinstance(settings['hash_bins'], bool)
            or settings['hash_bins'] < 2):
        raise ValueError('context.hash_bins must be an integer >= 2')
    if not np.isfinite(settings['smoothing']) or settings['smoothing'] <= 0:
        raise ValueError('context.smoothing must be positive and finite')
    return settings


class PromptDomainModel:
    """Frozen, train-split-only text classifier with a uniform domain prior.

    Hashed multinomial Naive Bayes is an implementation choice, not a classifier
    prescribed by the paper. Missing training domains have probability zero and
    are reported; this does NOT exclude their teachers from bandit selection.
    """
    def __init__(self, domains, hash_bins=32768, smoothing=1.0):
        self.domains = tuple(sorted(set(domains)))
        if not self.domains or hash_bins < 2 or not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError('Invalid domain estimator settings')
        self.hash_bins, self.smoothing = hash_bins, smoothing

    def _features(self, messages):
        # Read prompt text only: never answers, verifier metadata, or row.domain.
        text = '\n'.join(m['content'] for m in messages)
        tokens = re.findall(r'\w+|[^\w\s]', text.casefold())
        return Counter(int.from_bytes(hashlib.blake2b(t.encode('utf-8'), digest_size=8).digest(),
                                      'little') % self.hash_bins for t in tokens)

    def fit(self, rows):
        counts = np.full((len(self.domains), self.hash_bins), self.smoothing, dtype=np.float64)
        examples = np.zeros(len(self.domains), dtype=np.int64)
        indices = {d: i for i, d in enumerate(self.domains)}
        fingerprint = hashlib.sha256()
        for row in rows:
            i = indices.get(row['domain'])
            if i is None:
                continue
            if row.get('usage') == 'final_evaluation_only':
                raise ValueError('Final evaluation data cannot train the domain estimator')
            features = self._features(row['messages'])
            for token, count in features.items():
                counts[i, token] += count
            examples[i] += 1
            fingerprint.update(repr((row['domain'], row['messages'])).encode('utf-8'))
        if not examples.any():
            raise ValueError('No training prompts for the teacher domains')
        self.log_likelihood = np.log(counts) - np.log(counts.sum(axis=1, keepdims=True))
        self.examples = examples
        self.training_fingerprint = fingerprint.hexdigest()
        return self

    def predict(self, messages, teachers):
        if not hasattr(self, 'log_likelihood'):
            raise ValueError('Fit the domain estimator before routing')
        scores = np.zeros(len(self.domains))
        features = self._features(messages)
        if features:
            ids, counts = zip(*features.items())
            scores = self.log_likelihood[:, ids] @ np.asarray(counts)
        scores[self.examples == 0] = -np.inf
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        values = dict(zip(self.domains, probs))
        return np.asarray([values[t['domain']] for t in teachers])

    def manifest(self):
        return dict(estimator='hashed_multinomial_nb', prior='uniform_over_observed_domains',
                    hash_bins=self.hash_bins, smoothing=self.smoothing,
                    training_examples=dict(zip(self.domains, self.examples.tolist())),
                    missing_domains=[d for d, n in zip(self.domains, self.examples) if not n],
                    missing_domain_probability=0.0, input='prompt_messages_only',
                    training_fingerprint=self.training_fingerprint,
                    probability_calibration='not_calibrated')

    def save(self, path):
        np.savez_compressed(path, domains=np.asarray(self.domains), examples=self.examples,
                            log_likelihood=self.log_likelihood, hash_bins=self.hash_bins,
                            smoothing=self.smoothing, training_fingerprint=self.training_fingerprint)


def measured_costs(costs):
    """Teacher scoring seconds, excluding checkpoint loading and probe evaluation."""
    result = np.asarray([c['prefill_seconds'] for c in costs], dtype=np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid measured teacher prefill cost')
    return result


def routing_costs(seconds, cfg):
    """Declare the cost unit explicitly; shared hardware slowdown cancels out.

    Relative excess is (t_i - min_j t_j) / max_j t_j in [0, 1].
    It is zero for equally expensive teachers, even on a slow machine.
    Normalize each trajectory's candidate pool before averaging the batch.
    """
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Teacher scoring seconds must be a nonempty finite nonnegative vector')
    if configure_context(cfg)['cost'] == 'measured_prefill_seconds':
        return values.copy()
    maximum = values.max()
    if maximum == 0:
        return np.zeros_like(values)
    return (values-values.min())/maximum


def paper_context(probabilities, observed, kls, entropies, observed_student, similarity, costs):
    """Eq.9 in its stated order; cost units are declared by context.cost.

    A_i = mean(log pi_i(y_t) - log pi_student(y_t)).
    C_i = mean_{j != i} sim(pi_i, pi_j); zero for a singleton pool.
    """
    probabilities, observed, kls, entropies, costs = (
        np.asarray(x, dtype=np.float64) for x in (probabilities, observed, kls, entropies, costs))
    n = len(observed)
    similarity = np.asarray(similarity, dtype=np.float64)
    if any(x.shape != (n,) for x in (probabilities, kls, entropies, costs)) or similarity.shape != (n, n):
        raise ValueError('Teacher statistics must align with the teacher pool')
    if (probabilities < 0).any() or (probabilities > 1).any() or (costs < 0).any():
        raise ValueError('Invalid domain probabilities or costs')
    agreement = (similarity.sum(axis=1)-np.diag(similarity))/(n-1) if n > 1 else np.zeros(n)
    contexts = np.column_stack((probabilities, observed, kls, entropies,
                               observed-observed_student, agreement, costs))
    if not np.isfinite(contexts).all():
        raise ValueError('Nonfinite Eq.9 context')
    return contexts
