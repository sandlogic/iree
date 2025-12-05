import torch
from iree.turbine import ops
import iree.turbine.aot as aot

class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 30)

    def forward(self, x):
        l1 = self.linear1(x)
        r = torch.relu(l1)
        r_n = ops.iree.transfer_to_logical_device("cuda", r)
        x = self.linear2(r_n)
        return x   

def main():
    input_tensor = torch.randn(5, 10)
    
    cm = aot.export(TestNet(), args=(input_tensor,))
    asm = str(cm.mlir_module)

    with open("multi_device.mlir", "w") as f:
        f.write(asm)

if __name__ == "__main__":
    main()