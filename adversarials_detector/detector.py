import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, hidden=100, bidir=False, architecture=None, model_arc='small'):
        super(Detector, self).__init__()
        self.mlp = True if architecture == 'mlp' else False

        if architecture == 'mlp':
            print('Initializing MLP')
            in_features = 50 if model_arc == 'small' else 160
            self.fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=hidden),
                                    nn.ReLU(),
                                    nn.Dropout(0.5)
                                )
        else:
            print('Initializing LSTM')
            self.lstm = nn.LSTM(input_size=10, hidden_size=hidden, bidirectional=bidir, batch_first=True)

        out_size = hidden * 2 if (bidir and architecture=='lstm') else hidden
        self.classifier = nn.Linear(out_size, 1)

    def forward(self, x):
        if self.mlp:
            x = x.reshape(x.shape[0], -1)
            output = self.fc(x)
        else:
            output, _ = self.lstm(x)
            output = output[:, -1]
        return self.classifier(output)
