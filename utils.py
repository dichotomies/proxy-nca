import evaluation
import numpy as np
import torch
import logging

def predict_batchwise(model, dataloader):
    with torch.no_grad():
        X, Y = zip(*[
            [x, y] for X, Y in dataloader
                for x, y in zip(
                    model(X.cuda()).cpu(), 
                    Y
                )
        ])
    return torch.stack(X), torch.stack(Y)


def evaluate(model, dataloader, nb_classes):
    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels (non-batch-wise)
    X, T = predict_batchwise(model, dataloader)

    # calculate NMI with kmeans clustering
    nmi = evaluation.calc_normalized_mutual_information(
        T, 
        evaluation.cluster_by_kmeans(
            X, nb_classes
        )
    )
    logging.info("NMI: {:.3f}".format(nmi * 100))

    # get predictions by assigning nearest 8 neighbors with euclidian
    Y = evaluation.assign_by_euclidian_at_k(X, T, 8)
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 2, 4, 8]:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
    model.train(model_is_training) # revert to previous training state
    return nmi, recall

