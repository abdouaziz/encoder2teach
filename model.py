import torch 
import torch.nn as nn




# exmaple of Linear  dim 3 
input = torch.randn(8, 666 , 768)
input = input.permute(0,2,1)

m = nn.Linear(666, 32)
output = m(input)



print(f"input: {input.shape}")
print(f"output: {output.shape}")
