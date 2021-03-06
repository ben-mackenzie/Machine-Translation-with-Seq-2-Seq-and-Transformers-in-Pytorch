import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #############################################################################
        self.embedding_layer = nn.Embedding(self.input_size, self.emb_size)
        if model_type == "RNN":
            self.recurrent_layer = nn.RNN(input_size=self.emb_size, hidden_size=self.encoder_hidden_size, batch_first=True)
        else:
            self.recurrent_layer = nn.LSTM(input_size=self.emb_size, hidden_size=self.encoder_hidden_size, batch_first=True)
        self.linear_layer1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p = dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        
        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################
        embedding_out = self.embedding_layer(input)
        dropped_out = self.dropout(embedding_out)
        output, hidden = self.recurrent_layer(dropped_out)

        linear_1_out = self.linear_layer1(hidden)
        relu_out = self.relu(linear_1_out)
        linear_2_out = self.linear_layer2(relu_out)
        hidden = self.tanh(linear_2_out)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden