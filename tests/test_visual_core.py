import torch
import torch.nn as nn

from egomimic.models.stems import SpatialSoftmax, VisualCore


def test_spatial_softmax_returns_bounded_keypoints():
    pool = SpatialSoftmax(in_channels=3, in_h=4, in_w=5, num_kp=2)
    output = pool(torch.randn(6, 3, 4, 5))

    assert output.shape == (6, 2, 2)
    assert torch.isfinite(output).all()
    assert torch.all(output.abs() <= 1.0)


def test_visual_core_preserves_leading_shape_and_group_norm_contract():
    encoder = VisualCore(
        image_size=40,
        feature_dimension=12,
        num_kp=4,
        pretrained=False,
        crop_aug=True,
        crop_height=32,
        crop_width=32,
        crop_eval_mode="center",
        norm_layer="group",
    ).eval()

    with torch.inference_mode():
        output = encoder(torch.randn(2, 3, 3, 40, 40))

    assert output.shape == (2, 3, 12)
    assert torch.isfinite(output).all()
    assert not any(isinstance(module, nn.BatchNorm2d) for module in encoder.modules())


def test_libero_native_half_resnet_architecture_and_parameter_count():
    encoder=VisualCore(image_size=96,feature_dimension=32,num_kp=32,
        pretrained=False,crop_aug=True,crop_height=84,crop_width=84,
        crop_eval_mode="center",crop_sample_mode="v02",crop_scope="frame",
        norm_layer="group",pool_type="spatial_softmax",resnet_model="resnet18_half").eval()
    assert encoder.backbone[0].out_channels==32
    assert [encoder.backbone[i][-1].conv2.out_channels for i in (4,5,6,7)]==[32,64,128,256]
    assert sum(p.numel() for p in encoder.parameters())==2809184
    assert not any(isinstance(m,nn.BatchNorm2d) for m in encoder.modules())
    with torch.inference_mode():
        result=encoder(torch.zeros(1,3,96,96))
    assert result.shape==(1,32) and torch.isfinite(result).all()
