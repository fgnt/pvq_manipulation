import torch
import paderbox as pb

from padertorch.base import Model
from pvq_manipulation.helper.ode_solver import ode_solver


def broadcast_to(tensor, shape, device):
    if isinstance(tensor, (float, int)):
        tensor = torch.tensor(tensor, dtype=torch.float32, device=device)
    return torch.broadcast_to(tensor, shape)


class FlowMatching(Model):
    def __init__(self, ode_function, transformation_fkt):
        super().__init__()
        self.ode_function = ode_function
        self.transformation_fkt = transformation_fkt
        self.latent_dist = torch.distributions.MultivariateNormal(
            torch.zeros(ode_function.input_dim),
            torch.eye(ode_function.input_dim)
        )

    def get_transformed_noise(self, x_0, x_1, t):
        return self.transformation_fkt.sigma(x_1, t) * x_0 + self.transformation_fkt.mean(x_1, t)

    def get_target_vector_field(self, x_t, x_1, t):
        fraction = self.transformation_fkt.sigma_derivative(x_1, t) / (self.transformation_fkt.sigma(x_1, t) + 1e-6)
        return fraction * (x_t - self.transformation_fkt.mean(x_1, t)) + self.transformation_fkt.mean_derivative(x_1, t)

    def forward(self, data):
        data, condition = data
        if self.training:
            noise = self.latent_dist.sample((data.shape[0],))
        else:
            torch.manual_seed(0)            
            noise = self.latent_dist.sample((data.shape[0],))
        t = torch.rand(data.shape[0], 1, device=data.device)

        x_t = self.get_transformed_noise(noise, data, t)
        target_vector_field = self.get_target_vector_field(x_t, data, t)
        estimated_vector_field = self.ode_function(t, x_t, condition)
        return estimated_vector_field, target_vector_field

    def review(self, input, output):
        estimated_vector_field, target_vector_field = output
        loss = torch.mean((estimated_vector_field - target_vector_field) ** 2)
        return dict( 
            losses={
                'mse_loss': loss, 
            },
            scalars=dict(loss=loss)
        )
    
    def sample(
        self, 
        noise, 
        condition, 
        start=0, 
        stop=1, 
        steps=10, 
    ):
        with torch.no_grad():
            def helper_fn(y, t):
                t = broadcast_to(t, [y.shape[0], 1], device=y.device)
                Δy = self.ode_function(t, y, condition)  
                return Δy
            _, x = ode_solver(helper_fn, noise, start=start, stop=stop, steps=steps)  
            x = torch.stack(x, dim=0)
            return x

    @staticmethod
    def load_model(model_path, checkpoint):
        model_dict = pb.io.load_yaml(model_path / "config.yaml")
        model = Model.from_config(model_dict['model'])
        cp = torch.load(
            model_path / checkpoint,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=False
        )
        model_weights = cp.copy()
        if 'model' in model_weights.keys():
            model.load_state_dict(model_weights['model'])
        else:
            model.load_state_dict(model_weights)
        model.eval()
        return model
    
    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        return summary
    
    def example_to_device(self, examples, device):
        data = [example['observation'] for example in examples]
        condition = [example['speaker_conditioning'] for example in examples if 'speaker_conditioning' in example]

        data = torch.tensor(
            data, 
            device=device, 
            dtype=torch.float
        ).squeeze(1)
        condition_tensor = torch.tensor(
            condition,
            device=device,
            dtype=torch.float
        ) if condition else None
        if condition_tensor is not None:
            if condition_tensor.ndim == 1:
                condition_tensor = condition_tensor[:, None]
        return data, condition_tensor