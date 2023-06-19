from torch import nn
import torch 
import timm 
import warnings
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights

class PretrainedModel(nn.Module):
    def __init__(self,):
        super().__init__()


class timmModel(PretrainedModel):
    """
    Load a pretrained model from timm by specifying a model name, whether to use the
    pretrained or randomly initialized version, what sort of pooling to use and the number
    of in_channels

    Adapted form from Henkel et. al. 
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
    def __init__(
        self, 
        model_name: str, 
        pretrained: bool=True, 
        global_pool: str='',
        in_chans=None, 
        freeze_extractor: bool=True, 
        **kwargs, 
    ):
        """
        
        Parameters:
            model_name:
                the model name of the desired timm model
            pretrained:
                Whether to load the pretrained model or randomly initialize parameters
            global_pool:
                parameter in timm.create_model
            in_chans:
                None or int. If specified it is passed to the initialized model 'model_name'
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0, # this way, the model has no output head
            global_pool=global_pool,
            in_chans=in_chans,
        )

        if "efficientnet" in model_name:
            self.backbone_out = self.backbone.num_features
        else:
            self.backbone_out = self.backbone.feature_info[-1]["num_chs"]
        
        if freeze_extractor:
            for param in self.backbone.parameters():
                param.requires_grad = False 
    
    def get_out_dim(self):
        return self.backbone_out

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)  # (bs, channels, feats, time)
        while x.dim() > 2:
            x = x.mean(axis=-1)
        return x 

class OutputHead(nn.Module):
    """
    It is useful to split this part from the model itself in case we want to do some 
    post-processing on the model outputs before making predictions
    """
    def __init__(
        self, 
        n_in: int, 
        n_out: int, 
        activation: callable=nn.Sigmoid()
    ):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation 
    
    def forward(self, x, return_logits=True):
        x = self.layer(x)
        if return_logits: return x 
        else: return self.activation(x)

    

warnings.filterwarnings("ignore", category=UserWarning) 


available_models = {'resnet50':(resnet50, ResNet50_Weights.DEFAULT), 'resnet152':(resnet152, ResNet152_Weights.DEFAULT)}  # Different available models


class TorchPretrainedModel(PretrainedModel):

    def __init__(self, freeze_extractor=True, model_name='resnet50', pretrained: bool=True, **kwargs):

        super().__init__()
        # Select model and corresponding weights from models list and load it
        # for name in available_models.keys():
        #     if model_name == name:
        weights = available_models[model_name][1]
        model = available_models[model_name][0](weights=weights)
       
        # This is used since this model will have to be loaded a bit differently that a regular nn.Module model.
        # Specifically, model = ResNet50(), model = model.load_model() instead of just model = ResNet50()
        self.pre_trained = pretrained     # This is used since this model will have to be loaded 

        # Freeze all parameters if feature extractor should be fixed
        if freeze_extractor:
            for param in model.parameters():
                param.requires_grad = False

        self.model = model

    def get_model(self):
        return self.model
    
    def get_out_dim(self):
        return self.model.fc.out_features  
    
    def forward(self, x):
        x = torch.concat([x]*3, axis=1)
        return self.model(x)


if __name__ == '__main__':
    # Test that everything works
    input = torch.randn((10,3,224,224))

    # resnet 50
    model = TorchPretrainedModel(chosen_model='resnet50')
    resnet50_model = model.get_model()
    output = resnet50_model(input)
    assert output.size() == torch.Size([10,4])

    # resnet 152
    model = TorchPretrainedModel(chosen_model='resnet152')
    resnet152_model = model.get_model()
    output = resnet152_model(input)
    assert output.size() == torch.Size([10,4])
    print('works')
