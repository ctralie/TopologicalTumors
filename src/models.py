from torch import nn

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