from lightly.loss import NegativeCosineSimilarity,BarlowTwinsLoss,VICRegLoss, NTXentLoss

def get_loss(name=""):
    
    if name=="VICRegLoss":
        return VICRegLoss()
    
    elif name=="BarlowTwinsLoss":
        return BarlowTwinsLoss()
    
    elif name=="NegativeCosineSimilarity":
        return NegativeCosineSimilarity()
    elif name=="SimCLR":
        return NTXentLoss()
      
    else:
       raise ValueError(f"Unknown loss name: {name}")

    