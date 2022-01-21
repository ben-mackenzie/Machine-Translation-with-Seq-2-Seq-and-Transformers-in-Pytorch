import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################
        
        #i_t: input gate
        self.W_ii = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(self.hidden_size))
        self.sigmoid_it = nn.Sigmoid()

        # f_t: the forget gate
        self.W_if = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.sigmoid_ft = nn.Sigmoid()

        # g_t: the cell gate
        self.W_ig = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tanh_gt = nn.Tanh()
        
        # o_t: the output gate
        self.W_io = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(self.hidden_size))
        self.sigmoid_ot = nn.Sigmoid()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        # This helped me understand everything a lot better! https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
        batch_size, sequence_size, _ = x.shape[0], x.shape[1], x.shape[2]

        if init_states == None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(sequence_size):
            x_t = x[:,t,:]
            i_t = self.sigmoid_it(x_t @ self.W_ii + self.b_ii + h_t @ self.W_hi + self.b_hi)
            f_t = self.sigmoid_ft(x_t @ self.W_if + self.b_if + h_t @ self.W_hf + self.b_hf)
            g_t = self.tanh_gt(x_t @ self.W_ig + self.b_ig + h_t @ self.W_hg + self.b_hg)
            o_t = self.sigmoid_ot(x_t @ self.W_io + self.b_io + h_t @ self.W_ho + self.b_ho)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

