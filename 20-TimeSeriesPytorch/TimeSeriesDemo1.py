#https://www.zhihu.com/question/451816360/answer/2998793893

import torch
import torch.nn as nn

class TransformerModel( nn.Module ):
    def __init__(self, n_tokes, n_input, n_output, n_hidden=256, n_layers=3 , dropout=0.5 ):
        super( TransformerModel, self ).__init__()
        
        # Create embedding for input tokens
        self.embeddings = nn.Embedding( n_tokes, n_hidden )
        
        # Create multi-head attention layer
        self.self_attention = nn.ModuleList( [
            nn.MultiheadAttention( n_hidden, num_heads ) for num_heads in [8,8] 
            ])
        
        #Create feed-forward layers
        self.feed_forward = nn.Sequential(
            nn.Linear( n_hidden, n_hidden ),
            nn.ReLU(),
            nn.Linear( n_hidden, n_hidden ),
            nn.ReLU(),
            nn.Linear( n_hidden, n_hidden ),
            nn.ReLU(),
            nn.Linear( n_hidden, 1 ),
        )

        # Create input/output sequece mask
        self.input_mask =  self.generate_square_subsquent_maks( n_input )
        self.output_mask = self.generate_square_subsquent_maks( n_output )

        #create position endcodings
        self.positional_encodings = nn.Parameter( torch.randn( n_input + n_output, n_hidden))

        #Create final dropout and output layer
        self.dropout = nn.Dropout( dropout )
        self.output_layer = nn.Linear( n_hidden, 1 )


    def generate_square_subsquent_maks( self, x ):
        mask = ( torch.triu( torch.ones( x, x )) == 1 ).transpose( 0, 1 )
        mask = mask.float().masked_fill( mask==0, float( '-inf' )).masked_fill( mask==1, float( 0.0 ) )
        return mask
    
    def forward( self, x ):
        # Generate embeddings
        embeddings = self.dropout( self.embeddings( x ) )

        # Add postional encoding
        embeddings += self.positional_encodings[ 0:embeddings.shape[1], : ]

        #Generate input mask
        input_mask = self.input_mask[ :, :embeddings.shape[1], :embeddings.shape[1] ].to( x.device )

        # Apply multi-head attention
        for attention in self.self_attention:
            embeddings, _ = attention( embeddings, embeddings, embeddings, attn_mask = input_mask )
            embeddings = self.dropout( embeddings )

        # Generate output mask
        output_mask = self.output_mask[ :, -1:, :embeddings.shape[1] ].to( x.device )
        # Apply feed-forward layers
        output = torch.flatten( embeddings,start_dim=1 )
        output = self.feed_forward( output )
        output = output.transpose( 0, 1 )
        output = output.masked_fill( output_mask==float( '-inf'), float( 0.0 ) )
        output = output[ :, -1 ]

        #Apply final output layer and return result
        output = self.output_layer( output )
        return output
    

def generate_data( batch_size, sequence_length ):
    while True:
        # Generate random sequence data
        x = torch.rand( ( batch_size, sequence_length ))
        y = x.sum( dim=1 )
        yield x, y 


# Set up model and optimizer
model = TransformerModel( n_tokes=100, n_input=10, n_output=5 )
optimizer = torch.optim.Adam( model.parameters(), lr=0.001 )

# Train model on random sequence data
for i, (x, y ) in enumerate( generate_data( batch_size=32, sequence_length=10 )):
    optimizer.zero_grad()
    pred = model( x.long() )
    loss = nn.MSELoss()( pred, y )
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print( f"Step {i} loss: {loss.item()}" )

    

    