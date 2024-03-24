import torch.nn as nn


class CNN_LSTM_Model(nn.Module):
    """
    CNN-LSTM model for image sequence prediction.

    Args:
        image_height (int): Height of input images.
        image_width (int): Width of input images.
        num_channels (int): Number of input image channels.
        num_future_steps (int): Number of future steps to predict.

    Attributes:
        conv_block (nn.Sequential): Convolutional and pooling layers.
        reduced_size (int): Size of features after convolution and pooling.
        lstm (nn.LSTM): Long Short-Term Memory (LSTM) layer.
        fc (nn.Linear): Fully connected layer for predicting future steps.

    Example usage:
        model = CNN_LSTM_Model()
        future_predictions = model(input_data)
    """
    def __init__(self, image_height=128, image_width=128, num_channels=1, num_future_steps=3):
        super(CNN_LSTM_Model, self).__init__()

        self.num_future_steps = num_future_steps

        # Sequential block for convolution and pooling layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate the size of the features after the conv and pooling layers
        reduced_height = image_height // (2**3)
        reduced_width = image_width // (2**3)
        self.reduced_size = 64 * reduced_height * reduced_width

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.reduced_size, hidden_size=1024, num_layers=1, batch_first=True)

        # Fully connected layer for predicting future steps
        self.fc = nn.Linear(1024, num_future_steps * num_channels * image_height * image_width)

    def forward(self, x):
        """
        Forward pass through the CNN-LSTM model.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, sequence_length, num_channels, height, width).

        Returns:
            torch.Tensor: Predicted future images of shape (num_future_steps, num_channels, height, width).
        """
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)

        x = self.conv_block(x)  # Apply convolutional block
        x = x.view(batch_size, seq_len, -1)  # Flatten the features for the LSTM

        # Getting the LSTM outputs for each time step
        lstm_out, _ = self.lstm(x)

        # Predicting future steps from the last LSTM output
        future_images = self.fc(lstm_out[:, -1, :])
        future_images = future_images.view(self.num_future_steps, C, H, W)

        return future_images
