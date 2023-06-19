from lightly.loss import NegativeCosineSimilarity,BarlowTwinsLoss,VICRegLoss

def get_loss(name=""):
    
    if name=="VICRegLoss":
        return VICRegLoss()
    
    elif name=="BarlowTwinsLoss":
        return BarlowTwinsLoss()
    
    elif name=="NegativeCosineSimilarity":
        return NegativeCosineSimilarity()
    
    else:
       raise ValueError(f"Unknown loss name: {name}")

    