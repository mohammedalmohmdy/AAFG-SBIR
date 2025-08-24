import torch, torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss on L2-normalized embeddings.
    emb: (B,D), labels: (B,)
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, emb, labels):
        # cosine distance <-> euclidean on normalized vectors: d^2 = 2 - 2*cos
        sim = emb @ emb.t()  # B,B
        dist = (2 - 2*sim).clamp(min=0)

        labels = labels.view(-1,1)
        mask_pos = labels.eq(labels.t())              # positives mask incl. diagonal
        mask_neg = ~mask_pos                          # negatives mask

        # ignore self-comparisons in positives
        eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
        mask_pos = mask_pos & ~eye

        # hardest positive: max dist among positives
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = float('-inf')
        hardest_pos, _ = dist_pos.max(dim=1)
        hardest_pos[hardest_pos == float('-inf')] = 0.0  # handle classes with single sample

        # hardest negative: min dist among negatives
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float('inf')
        hardest_neg, _ = dist_neg.min(dim=1)
        hardest_neg[hardest_neg == float('inf')] = 0.0  # fallback

        loss = self.relu(hardest_pos - hardest_neg + self.margin).mean()
        return loss
