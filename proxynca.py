
from similarity import pairwise_distance
import torch
import torch.nn.functional as F


def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T


class ProxyNCAUnstable(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, smoothing_const = 0.0, 
            exclude_positive = False):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.exclude_positive = exclude_positive
        self.smoothing_const = smoothing_const
        
    def forward_single(self, X, T, i):

        P = self.proxies
        nb_classes = len(P)
        sz_batch = len(X)

        P = 3 * F.normalize(P, p = 2, dim = -1)
        X = 3 * F.normalize(X, p = 2, dim = -1)

        y_label = T[i].long().cuda()
        Z_labels = torch.arange(nb_classes).long().cuda()
        if self.exclude_positive:
            # all classes/proxies except of t
            Z_labels = Z_labels[Z_labels != y_label].long()
            assert Z_labels.size(0) == nb_classes - 1

        # necessary to calc distances for label smoothing
        # if label smoothing ommitted, one can simply use p_dist = D[i][y_label]
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        # use all positives, for probabilities, else p_dist = D[i][y_label]
        p_dist = D[i]
        n_dist = D[i][Z_labels]

        return torch.log(torch.exp(p_dist) / torch.sum(torch.exp(n_dist)))
    
    def forward(self, X, T):
        # log with exp results in unstable calculations, could use max to fix it
        out = torch.stack(
            [self.forward_single(X, T, i) for i in range(len(X))]
        )
        T = binarize_and_smooth_labels(
            T = T, 
            nb_classes = len(self.proxies), 
            smoothing_const = self.smoothing_const
        )
        # if D not calculated (pdist), then smoothing only for positive possible
        loss = (- T * out).sum(-1).mean()
        return loss
        
        
class ProxyNCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, smoothing_const = 0.0, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.smoothing_const = smoothing_const
        
    def forward(self, X, T):
        
        P = self.proxies
        P = 3 * F.normalize(P, p = 2, dim = -1)
        X = 3 * F.normalize(X, p = 2, dim = -1)
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = self.smoothing_const
        )

        # cross entropy with distances as logits, one hot labels
        # note that compared to proxy nca, positive not excluded in denominator
        loss = torch.sum(- T * F.log_softmax(D, -1), -1)

        return loss.mean()
        

if __name__ == '__main__':
    import random
    nb_classes = 100
    sz_batch = 32
    sz_embed = 64
    X = torch.randn(sz_batch, sz_embed).cuda()
    P = torch.randn(nb_classes, sz_embed).cuda()
    T = torch.arange(
        0, nb_classes
    ).repeat(sz_batch)[torch.randperm(nb_classes * sz_batch)[:sz_batch]].cuda()
    
    pnca = ProxyNCA(nb_classes, sz_embed).cuda()
    pnca_unst = ProxyNCAUnstable(
        nb_classes, sz_embed, exclude_positive= False
    ).cuda()
    pnca_unst.proxies.data = pnca.proxies.data.clone()
    
    print(pnca(X, T.view(sz_batch)))
    print(pnca_unst(X, T.view(sz_batch, 1)))

