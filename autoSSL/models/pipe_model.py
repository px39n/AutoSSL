from autoSSL.models import BarlowTwins, BYOL, MoCo, SimCLR, SimSiam, VICReg,Toymodel,FastSiam
from autoSSL.data import PipeDataset
# Function to get the model
def pipe_model(name="InputYourModelName", config=None, **kwargs):
    if config is not None:
        backbone = config["backbone"]
        stop_gradient = config["stop_gradient"]
        prjhead_dim = config["prjhead_dim"]
        view = config["view"]
        predhead_dim = config["predhead_dim"]
        loss_func = config["loss_func"]
        view_model = config["view_model"]
        optimizer=config["optimizer"]  
        schedule= config["schedule"]
        batch= config["batch_size"]
        max_epochs=config["max_epochs"]
        name=config["model"]
        
        samples=len(PipeDataset(config=config))
        
        if name == "MoCo":
            return MoCo(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name == "BYOL":
            return BYOL(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name == "SimCLR":
            return SimCLR(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name == "SimSiam":
            return SimSiam(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name == "BarlowTwins":
            return BarlowTwins(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name == "VICReg":
            return VICReg(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name=="FastSiam":
            return FastSiam(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim[0])
        elif name=="Toymodel":
            return Toymodel(backbone=backbone, stop_gradient=stop_gradient, prjhead_dim=prjhead_dim, views=view, predhead_dim=predhead_dim,loss_func=loss_func, view_model=view_model,optimizer=optimizer,schedule=schedule , batch=batch,max_epochs=max_epochs, **kwargs)
        else:
            raise ValueError(f"Unknown model name: {name}")

    else:
        # Use the original implementation if config is not provided
        if name == "MoCo":
            return MoCo(**kwargs)
        elif name == "BYOL":
            return BYOL(**kwargs)
        elif name == "SimCLR":
            return SimCLR(**kwargs)
        elif name == "SimSiam":
            return SimSiam(**kwargs)
        elif name == "BarlowTwins":
            return BarlowTwins(**kwargs)
        elif name == "VICReg":
            return VICReg(**kwargs)
        elif name == "FastSiam":
            return FastSiam(**kwargs)
        elif name == "Toymodel":
            return Toymodel(**kwargs)    
        
        else:
            raise ValueError(f"Unknown model name: {name}")
