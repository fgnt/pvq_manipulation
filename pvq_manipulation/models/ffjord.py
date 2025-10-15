import torch
import paderbox as pb

from padertorch.base import Model
from torchdiffeq import odeint_adjoint as odeint
from pvq_manipulation.helper.moving_batch_norm import MovingBatchNorm1d


if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'


class ODEBlock(torch.nn.Module):
    def __init__(
        self,
        ode_function,
        train_flag=True,
        reverse=False,
    ):
        super().__init__()
        self.time_deriv_func = ode_function
        self.noise = None
        self.reverse = reverse
        self.train_flag = train_flag

    def forward(
        self,
        time: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to use a neural network for dy(t)/dt = f_theta(t, y(t))

        Hutchinson’s trace estimator, as proposed in the FFJORD Paper, was adapted from:
        https://github.com/RameenAbdal/StyleFlow/blob/master/module/odefunc.py

        Args:
            time (torch.Tensor): Scalar tensor representing time
            states (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                - z (torch.Tensor): (batch_size, feature_dim) representing the input data.
                - d_log_dz_dt (torch.Tensor): (batch_size, 1) representing the log derivative.
                - labels (torch.Tensor): (batch_size, num_labeled_classes)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dz_dt (torch.Tensor): (batch_size, feature_dim) The derivative of z w.r.t. time
                - d_log_dz_dt (torch.Tensor): (batch_size, 1) The negative log derivative
                - labels (torch.Tensor): (batch_size, num_labeled_classes)
        """

        z, d_log_dz_dt, labels = states

        if self.noise is None:
            self.noise = self.sample_rademacher_like(z)

        with torch.enable_grad():
            z.requires_grad_(True)

            dz_dt = self.time_deriv_func.forward(time, z, labels)
            if self.train_flag:
                d_log_dz_dt = self.divergence_approx(dz_dt, z, self.noise)
            else:
                d_log_dz_dt = torch.zeros_like(z[:, 0]).requires_grad_(True)

        labels = torch.zeros_like(labels).requires_grad_(True)
        return dz_dt, -d_log_dz_dt.view(z.shape[0], 1), labels

    def divergence_approx(self, f, y, e=None):
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx.mul(e)

        cnt = 0
        while not e_dzdx_e.requires_grad and cnt < 10:
            e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
            e_dzdx_e = e_dzdx * e
            cnt += 1

        approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
        assert approx_tr_dzdx.requires_grad, \
            "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
            % (
                f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad,
                e_dzdx_e.requires_grad, cnt)
        return approx_tr_dzdx

    def before_odeint(self, e=None):
        self.noise = e

    def sample_rademacher_like(self, z):
        if not self.training:
            torch.manual_seed(0)
        return torch.randint(low=0, high=2, size=z.shape).to(z) * 2 - 1


class FFJORD(Model):
    """
    This class is an implementation of the FFJORD model as proposed in
    https://arxiv.org/pdf/1810.01367
    """
    def __init__(self, ode_function, normalize=True):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = ode_function.input_dim
        self.time_deriv_func = ODEBlock(ode_function=ode_function)
        self.latent_dist = torch.distributions.MultivariateNormal(
            torch.zeros(self.input_dim, device=device),
            torch.eye(self.input_dim, device=device),
        )
        self.normalize = normalize
        if self.normalize:
            self.input_norm = MovingBatchNorm1d(self.input_dim, bn_lag=0)
            self.output_norm = MovingBatchNorm1d(self.input_dim, bn_lag=0)

    @staticmethod
    def load_model(model_path, checkpoint, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = pb.io.load(model_path / "config_norm_flow.json")
        model = Model.from_config(model_dict)
        cp = torch.load(
            model_path / checkpoint,
            map_location=device,
            weights_only=True
        )
        model_weights = cp.copy()
        model.load_state_dict(model_weights)
        model.eval()
        model.to(device)
        return model

    def forward(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        integration_times: torch.Tensor = torch.tensor([0.0, 1.0]
        )
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integration from t_1 (data distribution) to t_0 (base distribution).
        (training step)

        Args:
            state (Tuple[torch.Tensor, torch.Tensor]):
                - z (torch.Tensor): (batch_size, feature_dim) representing the input data.
                - labels (torch.Tensor): (batch_size, num_labeled_classes)

            integration_times (torch.Tensor, optional): A tensor of shape (2,)
            specifying the start and end times for integration. Defaults to torch.tensor([0.0, 1.0]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dz_dt (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the derivative of z w.r.t. time.
                - -d_log_dz_dt (torch.Tensor): (batch_size, 1) representing the negative log derivative.
                - labels (torch.Tensor): (batch_size, num_labeled_classes)
        """
        z_1, labels = state

        if z_1.dim() == 3:
            z_1 = z_1.squeeze(1)

        delta_logpz = torch.zeros(z_1.shape[0], 1).to(z_1.device)

        if self.normalize:
            z_1, delta_logpz = self.input_norm(z_1, context=labels, logpx=delta_logpz)

        self.time_deriv_func.before_odeint()
        state = odeint(
            self.time_deriv_func,  # Calculates time derivatives.
            (z_1, delta_logpz, labels),  # Values to update. init states
            integration_times.to(z_1.device),  # When to evaluate.
            method='dopri5',  # Runge-Kutta
            atol=1e-5,  # Error tolerance
            rtol=1e-5,  # Error tolerance
        )
        if self.normalize:
            dz_dt, d_delta_log_dz_t = self.output_norm(state[0], context=state[2], logpx=state[1])
        else:
            dz_dt, d_delta_log_dz_t = state[0], state[1]

        state = (dz_dt, d_delta_log_dz_t, labels)

        if len(integration_times) == 2:
            state = tuple(s[1] if s.shape[0] > 1 else s[0] for s in state)
        return state

    def sample(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        integration_times: torch.Tensor = torch.tensor([1.0, 0.0])
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the sampling step. Integration from t_0 (base distribution) to t_1 (data distribution).

        Args:
            state (Tuple[torch.Tensor, torch.Tensor]):
                - z_0 (torch.Tensor): (batch_size, feature_dim) representing the initial state from the base distribution
                - labels (torch.Tensor): (batch_size, num_labeled_classes)

            integration_times (torch.Tensor, optional): A tensor of shape (2,) specifying the start (t_0) and end (t_1) times for integration.
                Defaults to torch.tensor([1.0, 0.0])

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z_t1 (torch.Tensor): (batch_size, feature_dim) representing the sampled data at time t_1 (data distribution).
                - labels (torch.Tensor): (batch_size, num_labeled_classes)
        """
        z_0, labels = state
        delta_logpz = torch.zeros(z_0.shape[0], 1).to(z_0.device)
        if self.normalize:
            z_0, delta_logpz = self.output_norm(
                z_0,
                context=labels,
                logpx=delta_logpz,
                reverse=True
            )

        state = odeint(
            self.time_deriv_func,  # Calculates time derivatives.
            (z_0, delta_logpz, labels),  # Values to update. init states
            integration_times.to(z_0.device),  # When to evaluate.
            method='dopri5',  # Runge-Kutta
            atol=1e-5,  # Error tolerance
            rtol=1e-5,  # Error tolerance
        )
        if self.normalize:
            dz_dt, d_delta_log_dz_t = self.input_norm(
                state[0],
                context=state[2],
                logpx=state[1],
                reverse=True
            )
        else:
            dz_dt, d_delta_log_dz_t = state[0], state[1]
        state = (dz_dt, d_delta_log_dz_t, labels)

        if len(integration_times) == 2:
            state = tuple(s[1] if s.shape[0] > 1 else s[0] for s in state)
        return state

    def example_to_device(self, examples, device):
        observations = [example['observation'] for example in examples]
        labels = [example['speaker_conditioning'].tolist() for example in examples if 'speaker_conditioning' in example]
        observations_tensor = torch.tensor(observations, device=device, dtype=torch.float)
        labels_tensor = torch.tensor(labels, device=device, dtype=torch.float) if labels else None
        return observations_tensor, labels_tensor

    def review(self, example, outputs):
        z_t0, delta_logpz, _ = outputs
        logpz_t1 = self.latent_dist.log_prob(z_t0) - delta_logpz
        loss = -torch.mean(logpz_t1)
        return dict(loss=loss, scalars=dict(loss=loss))

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        return summary

