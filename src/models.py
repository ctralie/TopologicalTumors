import torch
from torch import nn
import numpy as np

def get_sigmoid_accuracy(model, dataset):
    """
    Compute the accuracy of the output of a network 
    to binary ground truth labels

    Parameters
    ----------
    model: torch.nn.Module
        The model to apply
    dataset: torch.utils.data.Dataset
        Dataset to which to apply the model
    
    Returns
    -------
    A number between 0 and 1 indicating the proportion of correct
    binary classifications
    """
    num_correct = 0
    for data in dataset:
        inputs, labels = data
        outputs = model(inputs)
        outputs = np.sign(outputs.detach().numpy()).flatten()
        outputs = 0.5*(outputs + 1)
        num_correct += np.sum(labels == outputs)
    return num_correct / len(dataset)

def get_auroc(model, dataset):
    """
    Compute the AUROC of the output of a network 
    to binary ground truth labels

    Parameters
    ----------
    model: torch.nn.Module
        The model to apply
    dataset: torch.utils.data.Dataset
        Dataset to which to apply the model
    
    Returns
    -------
    A number between 0 and 1 indicating the area 
    under the ROC curve
    """
    from torchmetrics import AUROC
    pred = []
    target = []
    for data in dataset:
        inputs, labels = data
        outputs = model(inputs)
        outputs = outputs.detach().numpy().flatten()
        pred.append(outputs[0])
        target.append(labels)
    pred = torch.from_numpy(np.array(pred))
    target = torch.from_numpy(np.array(target, dtype=int))
    auroc = AUROC(task="binary")
    return auroc(pred, target).item()

def get_roc_image_html(fp, tp, title="", figsize=(5, 5)):
    """
    Create HTML code with base64 binary to display an ROC curve
    
    Parameters
    ----------
    fp: torch array(N)
        False positive cutoffs
    tp: torch array(N)
        True positive cutoffs
    title: string
        Title of plot
    figsize: tuple(float, float)
        Matplotlib figure size
    
    Returns
    -------
    string: HTML code with base64 binary blob of matplotlib image
    """
    import matplotlib.pyplot as plt
    import base64
    import io 
    auroc = torch.sum((fp[1::]-fp[0:-1])*tp[1::]).item()
    plt.figure(figsize=figsize)
    plt.plot(fp, tp)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} (AUROC {:.3f})".format(title, auroc))
    # https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64
    sstream = io.BytesIO()
    plt.savefig(sstream, format='png')
    sstream.seek(0)
    blob = base64.b64encode(sstream.read())
    s = "<img src=\"data:image/png;base64,{}\">".format(blob.decode())
    return s

class PImgCNNBinary(nn.Module):
    def __init__(self, dataset, depth, first_channels, pen_dim=0, device="cuda"):
        """
        A convolutional neural network that takes as input persistence image stacks
        and which outputs a soft binary value

        Parameters
        ----------
        dataset: torch dataset
            A dataset that will be used with this network, which is used to get the 
            dimensions of the linear layers right
        depth: int
            Depth of the CNN
        first_channels: int
            The number of channels in the output of the first convolutional layer
        pen_dim: int
            Dimension of the penultimate layer.  If 0, this layer is skipped
        """
        super(PImgCNNBinary, self).__init__()
        self.depth = depth
        self.pen_dim = pen_dim
        self.device = device
        
        ## Step 1: Create Convolutional Down Network
        images, _ = dataset[0] # Dummy to figure out shape
        print("images.shape", images.shape)
        layers = nn.ModuleList()
        lastchannels = images.shape[1]
        channels = first_channels
        for i in range(depth):
            ## TODO: Use batch norm?
            layers.append(nn.Conv2d(lastchannels, channels, 3, stride=2, padding=1))
            layers.append(nn.ReLU())
            lastchannels = channels
            if i < depth-1:
                channels *= 2
        
        ## Step 2: Create linear layers
        ## Step 2a: Setup flatten layers and determine correct dimensions
        layers.append(nn.Flatten())
        # Send a dummy tensor through
        y = images
        for layer in layers:
            y = layer(y)
        flatten_dim = y.shape[-1]
        ## Step 2b: Create penultimate layer, if desired
        if pen_dim > 0:
            penultimate = nn.Linear(flatten_dim, pen_dim)
            penultimate_relu = nn.ReLU()
            self.convdown += [penultimate, penultimate_relu]
            flatten_dim = pen_dim
        ## Step 2c: Create final layers to output
        layers.append(nn.Linear(flatten_dim, 1))
        self.layers = layers
    
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y



class PImgShallowCNNBinary(nn.Module):
    def __init__(self, dataset, out_channels, device="cuda"):
        """
        A convolutional neural network that takes as input persistence image stacks
        and which outputs a soft binary value.
        This network simply has one convolutional layer that's one pixel wide, before
        the final linear layer

        Parameters
        ----------
        dataset: torch dataset
            A dataset that will be used with this network, which is used to get the 
            dimensions of the linear layers right
        out_channels: int
            The number of output channels of the convolutional layer
        """
        super(PImgShallowCNNBinary, self).__init__()
        self.device = device
        
        ## Step 1: Create Convolutional Down Network
        images, _ = dataset[0] # Dummy to figure out shape
        
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(images.shape[1], out_channels, 1, stride=1, padding=0))
        layers.append(nn.ReLU())
        ## Step 2: Create linear layer
        ## Step 2a: Setup flatten layers and determine correct dimensions
        layers.append(nn.Flatten())
        # Send a dummy tensor through
        y = images
        for layer in layers:
            y = layer(y)
        flatten_dim = y.shape[-1]
        ## Step 2c: Create final layers to output
        layers.append(nn.Linear(flatten_dim, 1))
        self.layers = layers
    
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
