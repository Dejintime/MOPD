"""Equations 11, 17, 28-33, 37-38 of bandit-MOPD.pdf.

Selection uses detached, current-rollout features. All candidate prefills are
charged to measured runtime; this reference backend does not claim sparse cost.
"""
import itertools
import numpy as np
import torch


def sequence_weights(means, chosen, beta):
    """Eq.12: s_i is Eq.27's predicted utility, not Eq.28's UCB bonus."""
    if not chosen or not np.isfinite(beta) or beta < 0:
        raise ValueError('A nonempty team and finite nonnegative beta are required')
    values = np.asarray(means, dtype=np.float64)[chosen] * beta
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite teacher utility')
    weights = np.exp(values-values.max())
    return weights/weights.sum()


def team_objective(chosen, scores, similarity, costs, diversity, cost):
    """Eq.29 with Div(S) = sum_{i<j in S}(1-sim_ij), Cost(S)=sum_i cost_i."""
    return (sum(scores[i] for i in chosen)
            + diversity*sum(1-similarity[i, j] for i, j in itertools.combinations(chosen, 2))
            - cost*sum(costs[i] for i in chosen))


def mixture_log_probs(teacher_log_probs, weights, student_log_probs=None, epsilon=0.0):
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0, 1)")
    w = torch.as_tensor(weights, device=teacher_log_probs.device, dtype=torch.float32)
    if w.ndim != 1 or len(w) != len(teacher_log_probs) or (w <= 0).any():
        raise ValueError("positive sequence-level mixture weights required")
    w = w / w.sum()
    q = torch.logsumexp(teacher_log_probs.float() + w.log()[:, None, None], dim=0)
    if epsilon:
        if student_log_probs is None:
            raise ValueError("student distribution required for regularization")
        q = torch.logaddexp(q + np.log1p(-epsilon), student_log_probs.detach() + np.log(epsilon))
    return q.detach()


def reverse_kl_loss(student_log_probs, target_log_probs, teacher_log_probs, top_k=0):
    """Full vocab normalization is retained; no renormalization of the union.

    Every teacher contributes its actual probability at EVERY union token,
    including tokens outside that teacher's own top-k. Duplicate IDs count once.
    """
    p_log = student_log_probs.float()
    if top_k:
        k = min(top_k, p_log.shape[-1])
        ids = teacher_log_probs.topk(k, dim=-1).indices
        ids = ids.permute(1,0,2).reshape(p_log.shape[0],-1).sort(-1).values
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:,1:] = ids[:,1:] != ids[:,:-1]
        # Gather before arithmetic to avoid several T x vocabulary GPU buffers.
        q_log = target_log_probs.detach().gather(1,ids.to(target_log_probs.device)).to(p_log.device)
        p_selected = p_log.gather(1,ids.to(p_log.device))
        terms = p_selected.exp() * (p_selected-q_log) - p_selected.exp() + q_log.exp()
        return (terms * valid.to(p_log.device)).sum(-1).mean()
    q_log = target_log_probs.detach().to(p_log.device)
    return (p_log.exp() * (p_log - q_log)).sum(-1).mean()


class CombinatorialLinUCB:
    def __init__(self, n, dim, alpha=0.5, rho=0.99, ridge=1.0, diversity=0.1,
                 cost=0.01, domains=None):
        if not 0 < rho <= 1 or ridge <= 0:
            raise ValueError("invalid discount/ridge")
        base_A = np.repeat((np.eye(dim) * ridge)[None], n, axis=0)
        self.domains = tuple(domains or ())
        if len(set(self.domains)) != len(self.domains) or any(
                not isinstance(domain, str) or not domain for domain in self.domains):
            raise ValueError('domains must be unique nonempty strings')
        self.domain_indices = {domain: i for i, domain in enumerate(self.domains)}
        if self.domains:
            self.A = np.repeat(base_A[None], len(self.domains), axis=0)
            self.b = np.zeros((len(self.domains), n, dim))
        else:
            self.A = base_A
            self.b = np.zeros((n, dim))
        self.update_counts = np.zeros(self.A.shape[:-2], dtype=np.int64)
        self.reward_credit = np.zeros(self.A.shape[:-2], dtype=np.float64)
        self.alpha, self.rho = alpha, rho
        self.diversity, self.cost = diversity, cost

    def _account(self, domain):
        if not self.domains:
            if domain is not None:
                raise ValueError('This bandit was initialized without domain accounts')
            return self.A, self.b, self.update_counts, self.reward_credit
        try:
            index = self.domain_indices[domain]
        except KeyError as error:
            raise ValueError(f'Unknown bandit domain: {domain!r}') from error
        return (self.A[index], self.b[index], self.update_counts[index],
                self.reward_credit[index])

    def scores(self, contexts, domain=None):
        x = np.asarray(contexts, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite contexts")
        account_A, account_b, _, _ = self._account(domain)
        means, scores = [], []
        for A, b, c in zip(account_A, account_b, x):
            mean = c @ np.linalg.solve(A, b)
            uncertainty = max(0.0, c @ np.linalg.solve(A, c)) ** 0.5
            means.append(mean)
            scores.append(mean + self.alpha * uncertainty)
        return np.array(means), np.array(scores)

    def select(self, contexts, k, eligible, similarity, costs, domain=None):
        means, scores = self.scores(contexts, domain)
        chosen = []
        for _ in range(min(k, len(scores))):
            candidates = [i for i in range(len(scores)) if eligible[i] and i not in chosen]
            if not candidates:
                break
            baseline = team_objective(chosen, scores, similarity, costs, self.diversity, self.cost)
            gains = {i: team_objective(chosen+[i], scores, similarity, costs,
                                      self.diversity, self.cost)-baseline for i in candidates}
            best = max(candidates, key=lambda i: (gains[i], -i))
            if gains[best] < 0:
                break
            chosen.append(best)
        # The caller logs/skips empty teams with a bounded consecutive-skip limit.
        return chosen, means, scores

    def update(self, chosen, contexts, reward, weights, domain=None):
        if not np.isfinite(reward):
            raise ValueError("nonfinite reward")
        weights = np.asarray(weights) / np.sum(weights)
        account_A, account_b, update_counts, reward_credit = self._account(domain)
        for i, weight in zip(chosen, weights):
            c = np.asarray(contexts[i], dtype=np.float64)
            credit = reward * weight
            account_A[i] = self.rho * account_A[i] + np.outer(c, c)
            account_b[i] = self.rho * account_b[i] + c * credit
            update_counts[i] += 1
            reward_credit[i] += credit

    def account_summary(self, domain=None):
        _, _, update_counts, reward_credit = self._account(domain)
        return {'domain': domain, 'update_counts': update_counts.tolist(),
                'reward_credit': reward_credit.tolist()}

    def state_dict(self):
        return {"accounting": "teacher_by_domain" if self.domains else "teacher_global",
                "domains": list(self.domains), "A": self.A.tolist(), "b": self.b.tolist(),
                "update_counts": self.update_counts.tolist(),
                "reward_credit": self.reward_credit.tolist()}


def distribution_similarity(teachers):
    """Mean Bhattacharyya coefficient: normalized, symmetric and in [0,1]."""
    result = np.eye(len(teachers))
    for i, j in itertools.combinations(range(len(teachers)), 2):
        result[i, j] = result[j, i] = float(((teachers[i] + teachers[j]) * 0.5).exp().sum(-1).mean())
    return result
