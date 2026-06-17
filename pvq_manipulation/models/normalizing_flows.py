import torch
import paderbox as pb

from collections import defaultdict
from torchdiffeq import odeint_adjoint as odeint
from padertorch.base import Model
from pvq_manipulation.helper.moving_batch_norm import MovingBatchNorm1d
from pvq_manipulation.models.layers import grad_reverse


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
                - conditions (torch.Tensor): (batch_size, num_conditioned_classes)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dz_dt (torch.Tensor): (batch_size, feature_dim) The derivative of z w.r.t. time
                - d_log_dz_dt (torch.Tensor): (batch_size, 1) The negative log derivative
                - conditions (torch.Tensor): (batch_size, num_conditioned_classes)
        """

        z, d_log_dz_dt, conditions = states

        if self.noise is None:
            self.noise = self.sample_rademacher_like(z)

        with torch.enable_grad():
            z.requires_grad_(True)

            dz_dt = self.time_deriv_func.forward(time, z, conditions)
            if self.train_flag:
                d_log_dz_dt = self.divergence_approx(dz_dt, z, self.noise)
            else:
                d_log_dz_dt = torch.zeros_like(z[:, 0]).requires_grad_(True)

        conditions = torch.zeros_like(conditions).requires_grad_(True)
        return dz_dt, -d_log_dz_dt.view(z.shape[0], 1), conditions

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


class CCNF(Model):
    """
    This class is an implementation of the FFJORD model as proposed in
    https://arxiv.org/pdf/1810.01367
    """
    def __init__(self, ode_function, normalize=True):
        super().__init__()
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

    def forward(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        integration_times: torch.Tensor = torch.tensor([0.0, 1.0]
        )
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integration from t_1 (data distribution) to t_0 (base distribution).

        Args:
            state (Tuple[torch.Tensor, torch.Tensor]):
                - z (torch.Tensor): (batch_size, feature_dim) representing the input data.
                - conditions (torch.Tensor): (batch_size, num_conditioned_classes)
                - integration_times (torch.Tensor, optional): A tensor of shape (2,)
                    specifying the start and end times for integration. Defaults to torch.tensor([0.0, 1.0]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dz_dt (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the derivative of z w.r.t. time.
                - -d_log_dz_dt (torch.Tensor): (batch_size, 1) representing the negative log derivative.
                - conditions (torch.Tensor): (batch_size, num_conditioned_classes)
        """
        z_1, conditions = state
        delta_logpz = torch.zeros(z_1.shape[0], 1).to(z_1.device)

        if self.normalize:
            z_1, delta_logpz = self.input_norm(z_1, context=conditions, logpx=delta_logpz)

        self.time_deriv_func.before_odeint()
        state = odeint(
            self.time_deriv_func,  # Calculates time derivatives.
            (z_1, delta_logpz, conditions),  # Values to update. init states
            integration_times.to(z_1.device),  # When to evaluate.
            method='dopri5',  # Runge-Kutta
            atol=1e-5,  # Error tolerance
            rtol=1e-5,  # Error tolerance
        )
        if self.normalize:
            dz_dt, d_delta_log_dz_t = self.output_norm(state[0], context=state[2], logpx=state[1])
        else:
            dz_dt, d_delta_log_dz_t = state[0], state[1]

        state = (dz_dt, d_delta_log_dz_t, conditions)

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
                - conditions (torch.Tensor): (batch_size, num_conditioned_classes)
                - integration_times (torch.Tensor, optional): A tensor of shape (2,) specifying the start (t_0) and end (t_1) times for integration.
                    Defaults to torch.tensor([1.0, 0.0])

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z_t1 (torch.Tensor): (batch_size, feature_dim) representing the sampled data at time t_1 (data distribution).
                - conditions (torch.Tensor): (batch_size, num_conditioned_classes)
        """
        z_0, conditions = state
        delta_logpz = torch.zeros(z_0.shape[0], 1).to(z_0.device)
        if self.normalize:
            z_0, delta_logpz = self.output_norm(
                z_0,
                context=conditions,
                logpx=delta_logpz,
                reverse=True
            )

        state = odeint(
            self.time_deriv_func,  # Calculates time derivatives.
            (z_0, delta_logpz, conditions),  # Values to update. init states
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
        state = (dz_dt, d_delta_log_dz_t, conditions)

        if len(integration_times) == 2:
            state = tuple(s[1] if s.shape[0] > 1 else s[0] for s in state)
        return state
    
    def apply_resampling(
            self, 
            d_vector, 
            estimated_condtioning,
            target_conditioning,
        ):
        output_forward = self.forward(
            (d_vector, estimated_condtioning)
        )[0]
        sampled_class_manipulated = self.sample(
            (output_forward, target_conditioning)
        )[0]
        return sampled_class_manipulated
    
    @staticmethod
    def load_model(model_path, checkpoint):
        model_dict = pb.io.load(model_path / "config_norm_flow.yaml")
        model = Model.from_config(model_dict['model'])
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

    def example_to_device(self, examples):
        observations = []
        conditions = []
        for example in examples:
            observation = torch.tensor(example["observation"])
            if observation.dim() == 1:
                observation = observation[None, :]
    
            observations.append(observation)
    
            if "speaker_conditioning" in example:
                condition = torch.tensor(example["speaker_conditioning"])
                if condition.dim() == 1:
                    condition = condition[None]   
                elif condition.dim() == 0:
                    condition = condition[None, None]
                conditions.append(condition)
    
        observations = torch.concatenate(observations, dim=0).float().to(device)
        if conditions:
            conditions = torch.concatenate(conditions, dim=0).float().to(device)  
        else:
            conditions = None
    
        return observations, conditions

    def review(self, example, outputs):
        z_t0, delta_logpz, _ = outputs
        logpz_t1 = self.latent_dist.log_prob(z_t0) - delta_logpz
        loss = -torch.mean(logpz_t1)
        return dict(loss=loss, scalars=dict(loss=loss))

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        return summary


class StackedFlow(Model):
    def __init__(
        self,
        high_level_flow,
        low_level_flow,
        condition_list,
        classifier,
        regressor,
        lambda_adv=1
    ):
        super().__init__()
        self.high_level_flow = ODEBlock(ode_function=high_level_flow)
        self.low_level_flow = ODEBlock(ode_function=low_level_flow)
        self.latent_dist = torch.distributions.MultivariateNormal(
            torch.zeros(high_level_flow.input_dim, device=device),
            torch.eye(high_level_flow.input_dim, device=device),
        )
        self.input_norm = MovingBatchNorm1d(
            high_level_flow.input_dim, bn_lag=0
        )
        self.bottleneck_norm = MovingBatchNorm1d(
            high_level_flow.input_dim, bn_lag=0
        )
        self.output_norm = MovingBatchNorm1d(
            high_level_flow.input_dim, bn_lag=0
        )

        self.condition_list = condition_list
        self.classifier = classifier
        self.regressor = regressor
        self.lambda_adv = lambda_adv

    def forward(
            self,
            state: list[torch.Tensor, torch.Tensor],
            integration_times: torch.Tensor = [
                torch.tensor([0.0, 0.5]),
                torch.tensor([0.5, 1.0])
            ]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integration from t_1 (data distribution) to t_0 (base distribution).

        Args:
            state (Tuple[torch.Tensor, torch.Tensor]):
                - z (torch.Tensor): (batch_size, feature_dim) representing the input data.
                - conditions list of (torch.Tensor): 
                    [(batch_size, high_level_condition_dim), (batch_size, low_level_condition_dim)]
                - integration_times (torch.Tensor, optional): A tensor of shape (2,)
                    specifying the start and end times for integration. Defaults to torch.tensor([0.0, 1.0]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dz_dt (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the derivative of z w.r.t. time.
                - y (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the bottleneck state before the second flow.
                - d_delta_log_dz_t (torch.Tensor): (batch_size, 1) representing the negative log derivative at the end of integration
                - predictions (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the predictions from the classifier and regressor, if they are not None. Each tensor has shape (batch_size, num_classes) for the classifier and (batch_size, 1) for the regressor.
        """
        z_1, conditions = state

        delta_logpz = torch.zeros(z_1.shape[0], 1).to(z_1.device)
        z_1, delta_logpz = self.input_norm(
            z_1, 
            context=None, 
            logpx=delta_logpz
        )

        self.high_level_flow.before_odeint()
        z_1, delta_logpz, _ = odeint(
            self.high_level_flow, 
            (z_1, delta_logpz, conditions[0]),  
            integration_times[0].to(z_1.device),  
            method='dopri5', atol=1e-5, rtol=1e-5, 
        )

        dz_dt, d_delta_log_dz_t = self.bottleneck_norm(
            z_1[-1], context=None, logpx=delta_logpz[-1]
        )

        y = z_1[-1].detach().clone()

        dz_dt_reverse = grad_reverse(dz_dt, self.lambda_adv)
        prediction_clf = self.classifier(dz_dt_reverse) if self.classifier else None
        prediction_reg = self.regressor(dz_dt_reverse) if self.regressor else None

        self.low_level_flow.before_odeint()
        z_1, d_delta_log_dz_t, _ = odeint(
            self.low_level_flow,  
            (dz_dt, d_delta_log_dz_t, conditions[1][:, None]), 
            integration_times[1].to(z_1.device), 
            method='dopri5', atol=1e-5, rtol=1e-5,  
        )

        dz_dt, d_delta_log_dz_t = self.output_norm(
            z_1[-1], context=None, logpx=d_delta_log_dz_t[-1]
        )
        return dz_dt, y, d_delta_log_dz_t, (prediction_clf, prediction_reg)

    def sample(
            self,
            state: list[torch.Tensor, torch.Tensor],
            integration_times: torch.Tensor = [
                torch.tensor([1.0, 0.5]),
                torch.tensor([0.5, 0.0])
            ]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integration from t_1 (data distribution) to t_0 (base distribution).

        Args:
            state (Tuple[torch.Tensor, torch.Tensor]):
                - z (torch.Tensor): (batch_size, feature_dim) representing the input data.
                - conditions list of (torch.Tensor): 
                    [(batch_size, high_level_condition_dim), (batch_size, low_level_condition_dim)]
                - integration_times (torch.Tensor, optional): A tensor of shape (2,)
                    specifying the start and end times for integration. Defaults to torch.tensor([1.0, 0.0]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - dz_dt (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the derivative of z w.r.t. time.
                - -d_log_dz_dt (torch.Tensor): (batch_size, 1) representing the negative log derivative.
        """
        z_1, conditions = state

        if z_1.dim() == 3:
            z_1 = z_1.squeeze(1)

        delta_logpz = torch.zeros(z_1.shape[0], 1).to(z_1.device)
        z_1, delta_logpz = self.output_norm(
            z_1, 
            context=None, 
            logpx=delta_logpz, 
            reverse=True
        )
        
        self.low_level_flow.before_odeint()
        z_1, delta_logpz, _ = odeint(
            self.low_level_flow,
            (z_1, delta_logpz, conditions[1]),
            integration_times[0].to(z_1.device),
            method='dopri5', atol=1e-5, rtol=1e-5
        )

        y = z_1[-1].detach().clone()

        z_1, delta_logpz = self.bottleneck_norm(
            z_1[-1], 
            context=None, 
            logpx=delta_logpz[-1], 
            reverse=True
        )

        self.high_level_flow.before_odeint()
        z_1, delta_logpz, _ = odeint(
            self.high_level_flow,
            (z_1, delta_logpz, conditions[0]),
            integration_times[1].to(z_1.device),        
            method='dopri5', atol=1e-5, rtol=1e-5
        )
        z_1, delta_logpz = self.input_norm(
            z_1[-1], 
            context=None, 
            logpx=delta_logpz[-1], 
            reverse=True
        )
        return z_1, y, delta_logpz
    
    def apply_resampling(
            self, 
            d_vector, 
            estimated_condtioning,
            target_conditioning,
        ):
        output_forward = self.forward(
            (d_vector, estimated_condtioning)
        )[0]
        sampled_class_manipulated = self.sample(
            (output_forward, target_conditioning)
        )[0]
        return sampled_class_manipulated


    @staticmethod
    def load_model(model_path, checkpoint):
        model_dict = pb.io.load_yaml(model_path / "config_norm_flow_stacked.yaml")
        model = Model.from_config(model_dict['model'])
        cp = torch.load(
            model_path / checkpoint,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=False
        )
        model_weights = cp.copy()
        if 'model' in model_weights:
            model_weights = model_weights['model']
        model.load_state_dict(model_weights)
        model.eval()
        return model

    def example_to_device(self, examples, device):
        observations = [example['observation'] for example in examples]
        flow_conditions_dict = defaultdict(list)
        for example in examples:
            for idx, conditions in enumerate(self.condition_list):
                flow_conditions_dict[idx].append(
                    [example[condition] for condition in conditions]
                )
        observations_tensor = torch.tensor(observations, device=device, dtype=torch.float)
        for idx, labels in flow_conditions_dict.items():
            labels = torch.tensor(labels, device=device, dtype=torch.float)
            flow_conditions_dict[idx] = labels
        return observations_tensor, flow_conditions_dict

    def review(self, example, outputs):
        z_t0, _, delta_logpz, predictions = outputs
        _, labels = example

        logpz_t1 = self.latent_dist.log_prob(z_t0) - delta_logpz
        losses = {'likelihood': -torch.mean(logpz_t1)}
        scalars = {'delta_logpz': torch.mean(delta_logpz)}

        if self.classifier is not None:
            categorical_labels = labels[0][:, 0].long()
            categorical_probs = predictions[0]
            
            loss_bce = torch.nn.CrossEntropyLoss()(categorical_probs.squeeze(), categorical_labels.float())
            losses['bce_loss'] = loss_bce
            
            acc = (torch.argmax(torch.nn.functional.softmax(categorical_probs, dim=-1), dim=-1) == torch.argmax(categorical_labels, dim=-1)).float().mean()
            scalars.update({'bce_loss': loss_bce, 'acc': acc})

        if self.regressor is not None:
            pitch_labels = labels[0][:, 1]
            loss_mse = torch.nn.MSELoss()(predictions[1].squeeze(), pitch_labels)
            losses['mse_loss'] = loss_mse
            scalars['mae_loss'] = torch.sqrt(loss_mse)
            
        return dict(losses=losses, scalars=scalars)

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        return summary
