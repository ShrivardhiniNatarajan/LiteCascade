import torch
from litecascade.models.teacher import TeacherCNNBiLSTM
def test_teacher_forward():
    model = TeacherCNNBiLSTM(input_dim=115, num_classes=11)
    x = torch.randn(2, 10, 115)
    logits = model(x)
    assert logits.shape == (2, 11)
    
def test_teacher_params():
    model = TeacherCNNBiLSTM(input_dim=115, num_classes=11)
    params = sum(p.numel() for p in model.parameters())
    assert params > 500000
