import torch
from   torch import nn
from   torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from   torchvision import transforms

def proxy_nca_loss(proxies, xs, ts, i):
    
    nb_classes, sz_embed = [int(k) for k in proxies.data.shape]
    # print(nb_classes, sz_embed)
    # print(type(nb_classes), type(sz_embed))

    proxies_normed = F.normalize(proxies, p = 2, dim = 1)
    ys = torch.stack([proxies_normed[k] for k in ts.data])

    Z_i = torch.index_select(
        proxies_normed,
        0,
        Variable(
            torch.LongTensor(
                np.reshape(
                    np.setdiff1d(
                        np.arange(0, nb_classes), 
                        # ts.data.numpy()[i]
                        ts.data.cpu().numpy()[i]
                    ),
                    (nb_classes - 1)
                )
            )
        ).cuda()
    )

    p_dist = torch.exp(-torch.sum(torch.pow(ys[i] - xs[i], 2)))
    n_dist = torch.sum(torch.exp(-torch.sum(torch.pow(Z_i - xs[i], 2), dim = 1)))

    return -torch.log(p_dist / n_dist)


class ProxyNCALayer(nn.Module):
    def __init__(self, sz_embed, nb_classes, sz_batch):
        super().__init__() # adapt python3 if possible, i.e. super().__init__()
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.sz_batch = sz_batch
        self.proxies  = nn.Parameter(torch.randn(nb_classes, sz_embed))
        torch.nn.init.xavier_uniform(self.proxies)
        
    def forward(self, xs, ts):
        return torch.mean(
            torch.stack(
                [proxy_nca_loss(self.proxies, xs, ts, i) for i in range(self.sz_batch)]
            )
        )


class ProxyNCA(nn.Module):
    def __init__(self, no_top_model, sz_embed, nb_classes, sz_batch):
        super().__init__()
        # for inception
        self.no_top_model = no_top_model
        self.no_top_model.fc = nn.Linear(2048, sz_embed)
        torch.nn.init.xavier_uniform(self.no_top_model.fc.weight)
        self.proxy_nca_layer = ProxyNCALayer(sz_embed, nb_classes, sz_batch)
    
    def forward(self, xs, ts):
        tuples = self.no_top_model(xs)
        xs = tuples[0] # select very last part of inception model
        xs = F.normalize(xs)
        return self.proxy_nca_layer(xs, ts)


def remove_top_layer(model): # assumption torchvision.model, last module: `classifier`
    # vgg16
    model.classifier = nn.Sequential(*[i for i in list(model.classifier)[:-1]])
    return model


def outshape_last_layer(m):
    layers = list(filter(lambda x : x.__class__ == nn.Linear, list(m)))
    return layers[-1].__getattribute__("out_features")


if __name__ == "__main__":

    no_top_model = models.inception_v3(pretrained=True)
    SZ_BATCH = 20
    SZ_EMBED = 64
    NB_CLASS = 100

    pnca = ProxyNCA(no_top_model, ProxyNCALayer(SZ_EMBED, NB_CLASS, SZ_BATCH))
    xs_tr = Variable(torch.zeros(SZ_BATCH, 3, 224, 224))
    ts_tr = Variable(torch.LongTensor(SZ_BATCH, 1).zero_())

    print(pnca(xs_tr, ts_tr))
