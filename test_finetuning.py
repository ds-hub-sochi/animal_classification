from __future__ import annotations

from subprocess import run

import torch

from irbis_classifier.src.models.factory import Factory
from irbis_classifier.src.utils import is_valid_url


"""
path = 'https://sochi-hub.obs.rffu-moscow-1.hc.sbercloud.ru:443/checkpoints/biodiversity/animal_classification/weights.pth?AccessKeyId=FJWBG0QM6COHSE2QYT6U&Expires=1743174467&x-obs-security-token=gQtydS1tb3Njb3ctMYXf5y-6ah2uw8wPmCbQlK6dZ1ddIShfTucTLDpVZZdtvpt9sMoINeLnJXJEXSxZRGf7DLVii-3P6-h8ZeCIeOkyAniv72RnwHSmB57eCLHFi0yn-hOm_sYVWC1cFJm65MgvEE1NzEb74sJy4vPRz5Uw03Z4qs1fYbevYCS9SZVCNG_I_eOzP3Vmv8DysckUG10ViVdeYQGFgwk2-vgpb09lsMN6NnOugHGc97CNLixu44XS1co8PaFo3xz6y_oRlv5r5WsvDfnpFHZUvmxAl44Kd80Ul3J7-8_ta4FltqeJ_WFgcxj31TcZjzBA9CrxuqmPSLLOi6HM56YJPo3G01qjRinrYjduVH4P1KHaMQYFcS4AZ7Wp5nO4jGzQZNCOdrQk5vbQ1iZU53t_emYFjFFW83WyNDTb_VemaHrrT4S-5Gil_D80EQ9IKBc-jhWsZ6Ky4Ni9V78mlhr8UpVDFEuyRk2qnNVnNhwM3LT8CdnIYb2hzH6MHJ2P5yvqdTerwz-QezF4opbrkcgTTDI5uM7HdPR4VLORDZdBgffPv_0gzmRBe6S2SunRTgMnZwo7yKjFTkVUk3wpCk3kzfr0-dX8ZMJc_MkMok_pVhYPJwyihVQPXo1rpYSLqFD7in02PxjVTzY6jjZMLnrI9ExT5RqXou1AnK08T6OgrJveLFbCSU_CnE8g-3cYSUnf90X5cELk-4rkqBqUfqJCGJrXWQc5hVG5WMh9osLlANQkN2zmGVHZm5CPLYXNXRYMn0KMdwHTs_A_m6zQCBFEt34BW9qcyyCO7qDbm-E4mboTcHCrumLUbVa_18f-1_WPxQHHNdb3HUknbjMDVLXTpHwYXir1Z046D8_UNMuNZCnwUT9cSA%3D%3D&Signature=7mdhWsNDMLpaElXWtL19fmySuqk%3D'

run(['wget', path, '-O', 'classifier_v35_weights.pth'])

    
print(is_valid_url(path))

model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'

model = Factory().build_model(model_name, n_classes=26)

try:
    model.load_state_dict(
        torch.load(
            './classifier_v35_weights.pth',
            map_location='cpu',
        ),
    )
except EOFError:
    print('error')

print(model)
"""

import torch
from torch import nn

from collections import OrderedDict

from irbis_classifier.src.models.utils import replace_last_linear, get_last_linear


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model: nn.Module = nn.Sequential(
    nn.Sequential(
        OrderedDict(
            [
                ('linear1', nn.Linear(10, 20)),
                ('relu', nn.ReLU()),
                ('linear2', nn.Linear(20, 5)),
            ]
        )
    ),
    nn.Sequential(
        OrderedDict(
            [
                ('linear1', nn.Linear(5, 20)),
                ('relu', nn.ReLU()),
                ('linear2', nn.Linear(20, 5)),
            ]
        )
    ),
).to(device)

classification_head: nn.Linear = get_last_linear(model)

classification_head_params: set[torch.Tensor] = set(classification_head.parameters())
backbone_params: list[torch.Tensor] = [p for p in model.parameters() if p not in classification_head_params]

learning_rate: float = 1e-3

print(model[1].linear2.weight.data)

optimizer = torch.optim.AdamW(
    [
        {
            'params': backbone_params,
            'lr': 0.0,
        },
        {
            'params': list(classification_head_params),
            'lr': learning_rate
        },
    ],
)

criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

for i in range(100):
    batch: torch.Tensor = torch.ones((32, 10)).to(device)
    targets: torch.Tensor = torch.ones((32,)).type(torch.LongTensor).to(device)

    predictions: torch.Tensor = model(batch)

    optimizer.zero_grad()

    loss = criterion(predictions, targets)
    loss.backward()

    optimizer.step()

classification_head: nn.Linear = get_last_linear(model)

# print(classification_head.weight.data)
classification_head_params_after = set(classification_head.parameters())

print(list(classification_head_params)[0])
print(list(classification_head_params_after)[0])
backbone_params_after: list[torch.Tensor] = [p for p in model.parameters() if p not in classification_head_params_after]


for i, j in zip(backbone_params, backbone_params_after):
    assert torch.allclose(i, j)

for i, j in zip(classification_head_params, classification_head_params_after):
    assert not torch.allclose(i, j)
