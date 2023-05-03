import torch.nn as nn

class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            )
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32, 
                kernel_size=5, 
                stride=1, 
                padding=2
            )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        # self.conv1 = nn.Sequential(         
        #     nn.Conv2d(
        #         in_channels=1,              
        #         out_channels=16,            
        #         kernel_size=5,              
        #         stride=1,                   
        #         padding=2,                  
        #     ),                              
        #     nn.ReLU(),                      
        #     nn.MaxPool2d(kernel_size=2),    
        # )
        # self.conv2 = nn.Sequential(         
        #     nn.Conv2d(
        #         in_channels=16,
        #         out_channels=32, 
        #         kernel_size=5, 
        #         stride=1, 
        #         padding=2),     
        #     nn.ReLU(),                      
        #     nn.MaxPool2d(kernel_size=2),                
        # )
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

    def _init_weights(self, module):
          if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()