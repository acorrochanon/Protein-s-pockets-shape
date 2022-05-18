import torch
import e3nn
from e3nn import o3
from torch_scatter import scatter
from torch_cluster import radius_graph
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace


class EarlyStopping():
    def __init__(self, model, tolerance = 5):
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = 1000000000
        self.best_model = model
        
    def __call__(self, train_loss, validation_loss, model):
        if self.min_val_loss > validation_loss:
            self.min_val_loss = validation_loss
            self.best_model = model
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_neighbors) -> None:
        super().__init__()
        
        self.num_neighbors = num_neighbors
        
        # Required to know how many weights are required in the Multi-Layer Perceptron (MLP)
        tp = FullyConnectedTensorProduct(
            irreps_in1 = irreps_in,
            irreps_in2 = irreps_sh,
            irreps_out = irreps_out,
            internal_weights = False,
            shared_weights = False,
        )
        
        # MLP: [Input, internal, and output dimensions], activation function
        self.fc = FullyConnectedNet([3, 256, tp.weight_numel], torch.relu)
        # Tensor product
        self.tp = tp
        # Visualize TP
        self.irreps_out = self.tp.irreps_out

    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        # To map the relative distances to the weights of the tensor product we will embed the distances
        # using a basis function and then feed this embedding (edge_scalars) to a neural network. 
        weight = self.fc(edge_scalars)
        # To compute this quantity per edges, so we will need to “lift” the input feature to the edges.
        # For that we use edge_src that contains, for each edge, the index of the source node.
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        # Sum over the neighbors. Get final output
        node_features = scatter(edge_features, edge_dst, dim=0).div(self.num_neighbors**0.5)
        
        return node_features


class Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Number of neighbors hyperparameter
        self.num_neighbors = 3.8
        
        # Set the spherical harmonics 
        self.irreps_sh = o3.Irreps.spherical_harmonics(3)

        # 160 -> 128
        gate = Gate(
            "16x0e + 16x0o", [torch.relu, torch.abs],  # scalar
            "8x0e + 8x0o + 8x0e + 8x0o", [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
            "16x1o + 16x1e"  # gated tensors, num_irreps has to match with gates
        )
        # 128 -> 104
        gate2 = Gate(
            "16x0e + 16x0o", [torch.relu, torch.abs],  # scalar
            "6x0e + 6x0o + 6x0e + 6x0o", [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
            "12x1o + 12x1e"  # gated tensors, num_irreps has to match with gates
        )
        # 104 -> 88
        gate3 = Gate(
            "20x0e + 20x0o", [torch.relu, torch.abs],  # scalar
            "4x0e + 4x0o + 4x0e + 4x0o", [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
            "8x1o + 8x1e"  # gated tensors, num_irreps has to match with gates
        )
        
        # Gate. irreps_out = irreps_scalars + (ElementWiseTensorProduct(irreps_gates, irreps_gated))
        self.gate = gate
        self.gate2 = gate2
        self.gate3 = gate3
        
        # Convolutional layer. Irreps_sh, irreps_sh, gate.irreps_in, num_neighbors
        self.conv = Convolution(self.irreps_sh, self.irreps_sh, gate.irreps_in, self.num_neighbors)
        
        # Second convolutional layer 
        self.conv2 = Convolution(self.gate.irreps_out, self.irreps_sh, self.gate2.irreps_in, self.num_neighbors)
        
        # Third convolutional layer 
        self.conv3 = Convolution(self.gate2.irreps_out, self.irreps_sh, self.gate3.irreps_in, self.num_neighbors)
        
        # Final layer. gate ouput, irreps_sh, output specified, num_neighbors. 
        self.final = Convolution(self.gate3.irreps_out, self.irreps_sh, "1x0e", self.num_neighbors)
        
        # Final output
        self.irreps_out = self.final.irreps_out

        # Sigmoid
        self.sigmoid = torch.nn.Sigmoid()
        
        # Linear 
        # self.linear = torch.nn.Linear(gate_shape(self.gate3, 'out') + 12 , 1)
        self.linear = torch.nn.Linear(13 , 1)
        
        
    def forward(self, data, prnt = False) -> torch.Tensor:
        # Set the number of nodes and max radius.
        num_nodes = 4
        max_radius = 6.1
        
        # Generate graph using the node positions and creating the edges when the relative distance 
        # between a pair of nodes is smaller than max_radius (r).
        edge_src, edge_dst = radius_graph(x = data.pos, r = max_radius, batch=data.batch)
        edge_vec = data.pos[edge_src] - data.pos[edge_dst]

        # Computing the sh
        # Normalize=True ensure that x is divided by |x| prior computation
        edge_attr = o3.spherical_harmonics(l=self.irreps_sh, x = edge_vec, normalize=True, normalization='component')
        
        # Embed the distances then feed this embedding to the MLP (Convolutional class)
        edge_length_embedded = soft_one_hot_linspace(x=edge_vec.norm(dim=1), start=0.5, end=2.5, number=3, 
            basis='smooth_finite', cutoff=True) * 3**0.5
        
        # Data must be loaded into the GPU if available. 
        edge_src = edge_src.to(self.device)
        edge_dst = edge_dst.to(self.device)
        edge_attr = edge_attr.to(self.device)
        edge_length_embedded = edge_length_embedded.to(self.device)
        data.batch = data.batch.to(self.device)
        #data.x = data.x.to(self.device)
        
        #---------------------- LAYERS + GATES --------------------------
        x = scatter(edge_attr, edge_dst, dim=0).div(self.num_neighbors**0.5)
        if prnt is True:
            print('1- Scatter:', x.shape)

        x = self.conv(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        if prnt is True:
            print('2- Conv:', x.shape)
        
        x = self.gate(x)
        if prnt is True:
            print('3- Gate:', x.shape)    
        
        x = self.conv2(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        if prnt is True:
            print('4- Conv:', x.shape)

        x = self.gate2(x)
        if prnt is True:
            print('5- Gate:', x.shape)
        
        x = self.conv3(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        if prnt is True:
            print('6- Conv:', x.shape)
            
        x = self.gate3(x)
        if prnt is True:
            print('7- Gate:', x.shape)

        x = self.final(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        if prnt is True:
            print('8- Conv:', x.shape)        

        # Add fpocket features
        #x = torch.cat([x, data.x], dim = 1)
        
        if prnt is True:
            print('9- Add features:', x.shape)

        x = scatter(x, data.batch, dim=0).div(num_nodes**0.5)
        if prnt is True:
            print('10- Scatter:', x.shape,'\n')
        
        #x = self.linear(x)
        x = self.sigmoid(x)


        return x

