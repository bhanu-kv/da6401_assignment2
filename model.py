import torch
from torch import nn

class CNN_Model(nn.Module):
    def __init__(
        self,
        conv_layers_config,
        fc_layers_config,
        input_shape=(3, 224, 224),
        batch_norm=False,
        conv_dropout=0.1,
        fc_dropout=0.1,
        output_classes=10,
        use_softmax=False,  # Only use if not using nn.CrossEntropyLoss
    ):
        """
        conv_layers_config: list of dicts, each dict contains:
            - in_channels
            - out_channels
            - kernel_size
            - stride
            - padding
            - activation (str or None)
            - pooling: dict with keys 'type', 'kernel_size', 'stride' or None
        fc_layers_config: list of dicts, each dict contains:
            - out_features
            - activation (str or None)
        input_shape: tuple, e.g., (3, 224, 224)
        batch_norm: bool, whether to use BatchNorm2d after conv
        conv_dropout: float, dropout rate after conv layers
        fc_dropout: float, dropout rate after FC layers
        output_classes: int, number of output classes
        use_softmax: bool, whether to apply softmax at the end
        """
        super().__init__()
        self.use_softmax = use_softmax

        # Activation function mapping
        self.activation_dict = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'ELU': nn.ELU,
            'SELU': nn.SELU,
            'CELU': nn.CELU,
            'GELU': nn.GELU,
            'Mish': nn.Mish,
            'Sigmoid': nn.Sigmoid,
            'Tanh': nn.Tanh,
            None: nn.Identity
        }

        # Pooling function mapping
        self.pooling_dict = {
            'Max': nn.MaxPool2d,
            'Avg': nn.AvgPool2d,
            None: None
        }

        # Build convolutional layers
        conv_layers = []

        for cfg in conv_layers_config:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=cfg['in_channels'],
                    out_channels=cfg['out_channels'],
                    kernel_size=cfg['kernel_size'],
                    stride=cfg.get('stride', 1),
                    padding=cfg.get('padding', 0)
                )
            )

            if batch_norm: 
                conv_layers.append(nn.BatchNorm2d(cfg['out_channels']))

            activation = cfg.get('activation', None)
            if activation not in self.activation_dict:
                raise ValueError(f"Unsupported activation: {activation}")
            conv_layers.append(self.activation_dict[activation]())

            conv_layers.append(nn.Dropout2d(p=conv_dropout))

            pooling = cfg.get('pooling', None)
            if pooling and pooling['type'] in self.pooling_dict and pooling['type'] is not None:
                pool_cls = self.pooling_dict[pooling['type']]
                conv_layers.append(pool_cls(kernel_size=pooling['kernel_size'], stride=pooling['stride']))

        self.conv_model = nn.Sequential(*conv_layers)

        # Calculate flattened feature size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv_model(dummy)
            self.flat_features = conv_out.view(1, -1).shape[1]

        # Build fully connected layers
        fc_layers = []
        in_features = self.flat_features

        for cfg in fc_layers_config:
            fc_layers.append(nn.Linear(in_features, cfg['out_features']))
            activation = cfg.get('activation', None)

            if activation not in self.activation_dict:
                raise ValueError(f"Unsupported activation: {activation}")
            
            fc_layers.append(self.activation_dict[activation]())
            fc_layers.append(nn.Dropout(p=fc_dropout))
            in_features = cfg['out_features']

        fc_layers.append(nn.Linear(in_features, output_classes))

        if use_softmax == True:
            fc_layers.append(nn.Softmax(dim=1))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_model(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
