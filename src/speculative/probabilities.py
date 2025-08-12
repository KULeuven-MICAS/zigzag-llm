from typing import List

import numpy as np


def geometric_acceptance_pdf(p: float, draft_len: int) -> List[float]:
    """
    Returns the probability mass function (PDF) for the number of accepted draft tokens
    before rejection under a constant acceptance probability model.

    Parameters:
        p (float): constant acceptance probability per token
        draft_len (int): number of tokens proposed by the draft model

    Returns:
        List[float]: probability of accepting exactly k tokens, for k in [0, ..., draft_len]
    """
    pdf = [p**k * (1 - p) for k in range(draft_len)]
    pdf.append(p**draft_len)  # Probability all tokens are accepted
    return pdf


def exp_decay_acceptance_pdf(
    draft_len: int, p0: float = 0.1, lam: float = 0.9
) -> List[float]:
    """
    Returns the PDF for the number of accepted draft tokens before rejection
    under an exponentially decaying acceptance probability model.

    Parameters:
        draft_len (int): number of tokens proposed by the draft model
        p0 (float): initial acceptance probability
        lam (float): decay rate (lambda)

    Returns:
        List[float]: probability of accepting exactly k tokens, for k in [0, ..., draft_len]
    """
    p_accept = p0 * np.exp(
        -lam * np.arange(draft_len)
    )  # per-token accept probabilities
    pdf = []
    for k in range(draft_len):
        prefix_prob = np.prod(p_accept[:k]) if k > 0 else 1.0
        pdf.append(prefix_prob * (1 - p_accept[k]))
    pdf.append(np.prod(p_accept))  # All tokens accepted
    return pdf
