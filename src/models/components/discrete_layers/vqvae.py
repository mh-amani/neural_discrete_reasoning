from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
from torch import nn
# from vector_quantize_pytorch import VectorQuantize
from entmax import sparsemax

class VQVAEDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)      
        
        self.projection_method = kwargs.get("projection_method",None)
        
        self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
        self.dictionary.weight = torch.nn.Parameter(self.project_matrix(self.dictionary.weight))
        
        self.dist_ord = kwargs.get('dist_ord', 2) 
        self.embedding_loss = torch.nn.functional.mse_loss # torch.nn.CosineSimilarity(dim=-1)
        self.hard = kwargs['hard']
        self.kernel = nn.Softmax(dim=-1)
        self.beta = kwargs.get("beta",0.25) #0.25 is the beta used in the vq-vae paper
        
    ###################
    #Probably can remove these as we are using th matrix projection now
    # def fetch_embeddings_by_index(self,indices):
    #     if self.normalize_embeddings:
    #         return nn.functional.normalize(self.dictionary(indices),dim=-1)
    #     #~else
    #     return self.dictionary(indices)
        
    # def fetch_embedding_matrix(self):
    #     if self.normalize_embeddings:
    #         return nn.functional.normalize(self.dictionary.weight,dim=-1)
    #     #~else
    #     return self.dictionary.weight
    ###################
    
    def project_matrix(self,x):
        if self.projection_method == "unit-sphere":
            return torch.nn.functional.normalize(x,dim=-1)
        if self.projection_method == "scale":
            # devide the vector by the square root of the dimension
            return x / torch.sqrt(self.dictionary_dim)
        if self.projection_method == "layer norm":
            return self.out_layer_norm(x)     
        return x
    
    def discretize(self, x, **kwargs) -> dict: 
        probs = self.kernel( - self.codebook_distances(x) / self.temperature)
        x = self.project_matrix(x)
        indices = torch.argmax(probs, dim=-1)

        if self.hard:
            # Apply STE for hard quantization
            quantized = self.dictionary(indices)#self.fetch_embeddings_by_index(indices)
            quantized = quantized + x - (x).detach()
        else:
            quantized = torch.matmul(probs,  self.dictionary.weight)

        if kwargs.get("supervision",False):
            true_quantized = self.dictionary(kwargs.get("true_ids",None))
            commitment_loss = self.embedding_loss(true_quantized.detach(),x)
            embedding_loss = self.embedding_loss(true_quantized,x.detach())
            
        else:
            commitment_loss = self.embedding_loss(quantized.detach(),x) 
            embedding_loss = self.embedding_loss(quantized,x.detach())
            
        vq_loss = self.beta * commitment_loss + embedding_loss
        
        return indices, probs, quantized, vq_loss

    def codebook_distances(self, x):
        
        #dictionary_expanded = self.fetch_embedding_matrix().unsqueeze(0).unsqueeze(1) # Shape: (batch, 1, vocab, dim)
        dictionary_expanded = self.dictionary.weight.unsqueeze(0).unsqueeze(1)
        x_expanded = x.unsqueeze(2)
        # if self.normalize_embeddings:
        #     x_expanded = nn.functional.normalize(x,dim=-1).unsqueeze(2)  # Shape: (batch, length, 1, dim)
        # else:   
        #     x_expanded = x.unsqueeze(2)  # Shape: (batch, length, 1, dim)

        # Compute the squared differences
        dist = torch.linalg.vector_norm(x_expanded - dictionary_expanded, ord=self.dist_ord, dim=-1)
        return dist

    