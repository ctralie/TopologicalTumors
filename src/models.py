from torch import nn

class PImgCNNBinary(nn.Module):
    def __init__(self, dataset_train, dataset_test, depth, pen_dim=0, device="cuda"):
        """
        A convolutional neural network that takes as input persistence image stacks
        and which outputs a soft binary value

        Parameters
        ----------
        dataset_train: torch dataset
            Dataset to be used for training
        dataset_test: torch dataset
            Dataset to be used for testing
        depth: int
            Depth of the CNN
        pen_dim: int
            Dimension of the penultimate layer.  If 0, this layer is skipped
        """
        super(PImgCNNBinary, self).__init__()
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.depth = depth
        self.pen_dim = pen_dim
        self.device = device
        
        ## Step 1: Create Convolutional Down Network
        images, _ = dataset_train[0] # Dummy to figure out shape
        
        layers = nn.ModuleList()
        lastchannels = images.shape[1]
        channels = lastchannels*2
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
