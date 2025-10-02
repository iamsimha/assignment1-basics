import torch

def save_checkpoint(model, optimizer, iteration, out):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    save_dict = {"model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "iteration": iteration}
    torch.save(save_dict, out)

def load_checkpoint(src, model, optimizer):
    save_dict = torch.load(src)
    model.load_state_dict(save_dict["model_state_dict"])
    optimizer.load_state_dict(save_dict["optimizer_state_dict"])
    return save_dict["iteration"]