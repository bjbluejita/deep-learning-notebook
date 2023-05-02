import torch

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
#print( device )
tensor = torch.rand(3,4).to( device )

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

tensor1 = torch.ones(4, 4)
tensor2 = torch.ones(4, 4) * 2
t1 = torch.cat( [ tensor1, tensor2 ], dim=1 )
print( t1 )

y3 = torch.rand_like(tensor1)
print( t1.sum( dim=1 ) )