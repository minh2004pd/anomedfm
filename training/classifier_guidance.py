import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.utils import ModelWrapper

class GuidedVelocityModel(ModelWrapper):
    """
    Wraps a flow model (velocity predictor) and a classifier to provide
    classifier-guided flow matching.
    
    The guided velocity is:
        v_guided(x, t) = v_flow(x, t) + scale * \nabla_x \log p(y=0 | x, t)
    """
    def __init__(self, flow_model: nn.Module, classifier: nn.Module, guidance_scale: float = 1.0):
        super().__init__(flow_model)
        self.classifier = classifier
        self.guidance_scale = guidance_scale
        self.nfe_counter = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        """
        ODE Solver calls this.
        x: (B, C, H, W)
        t: scalar or (B,)
        """
        self.nfe_counter += 1
        
        # 1. Đảm bảo t có shape (B,)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
            
        # Tách x ra khỏi đồ thị hiện tại để làm node lá (leaf node) cho autograd
        x_in = x.detach().requires_grad_(True)
        
        # 2. Tính Gradient của Classifier (Chỉ bật grad cho phần này)
        with torch.enable_grad():
            logits = self.classifier(x_in, t)
            
            # Logit là cho y=1 (unhealthy).
            # \log p(y=0|x) = \log(\sigma(-logits)) = -softplus(logits)
            # Dùng tổng (.sum()) để có thể tính đạo hàm cho cả batch cùng lúc
            log_p_healthy = -F.softplus(logits).sum()
            
            # Tính \nabla_x
            grad_x = torch.autograd.grad(log_p_healthy, x_in)[0]
            
        # 3. Tính Velocity của Flow Model (Tắt grad hoàn toàn)
        with torch.no_grad():
            # Sử dụng x gốc (không có requires_grad) để tránh tạo graph dư thừa
            v_flow = self.model(x, t, extra=kwargs.get("extra", {}))
        
        # 4. Kết hợp Velocity và Classifier Gradient
        # Tuỳ thuộc vào công thức Flow Matching cụ thể, đôi khi guidance_scale 
        # có thể cần nhân thêm với (1 - t) hoặc một hàm phụ thuộc thời gian khác.
        v_guided = v_flow + self.guidance_scale * grad_x
        
        return v_guided

    def reset_nfe_counter(self):
        self.nfe_counter = 0
        
    def get_nfe(self):
        return self.nfe_counter