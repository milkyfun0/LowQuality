import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim=256, num_embeddings=1000, commitment_cost=0.25, topk=3):
        super().__init__()
        self.topk = topk
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # torch.nn.init.uniform_(self.embeddings.weight, -1/self.num_embeddings, 1/self.num_embeddings)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        encoding_indices, weights = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices, weights)
        # quantized = x + (quantized - x).detach()

        # return quantized, encoding_indices
        if not self.training:
            return quantized, encoding_indices

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # print("##########")
        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, flat_x):
        weight = self.embeddings.weight
        flat_x = F.normalize(flat_x, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, weight.t())
        )  # [N, M]
        # encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        top_values, top_indices = torch.topk(distances, k=self.topk, dim=1, largest=False)
        top_values = 1 / top_values
        weights = top_values / top_values.sum(dim=1, keepdims=True)
        weights = weights.unsqueeze(-1)
        return top_indices, weights

    # def quantize(self, weight, encoding_indices):
    def quantize(self, encoding_indices, weights):
        """Returns embedding tensor for a batch of indices."""
        topk_quantized = self.embeddings(encoding_indices)
        quantized = torch.sum(topk_quantized * weights, dim=1)
        return quantized

class AlexNet(nn.Module):
    def __init__(self, hash_bit, classes, num_embedding, topk):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias
        self.codebook = VectorQuantizer(hash_bit, num_embedding, 0.25, topk)
        self.fc_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit)
        )
        self.fc = nn.Linear(hash_bit, classes)
        self.tanh = nn.Tanh()

    def forward(self, x):

        if not self.training:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            feat = self.fc_layer(x)
            quantized, index = self.codebook(feat)
            hash_ = self.tanh(quantized)
            return hash_
        
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc_layer(x)
        quantized, e_q_loss, index = self.codebook(x)
        # print(index)
        hash_ = self.tanh(quantized)
        logit = self.fc(quantized)
        return hash_,  e_q_loss, logit, index