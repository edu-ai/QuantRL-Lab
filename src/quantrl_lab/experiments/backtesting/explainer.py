from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


class AgentExplainer:
    """
    Analyzes a trained RL agent's behavior to provide interpretability.

    Phase 1: Behavioral analysis using Pearson correlation.
    Phase 2: Deep Learning Feature Attribution (Input x Gradient / Saliency).
    """

    def __init__(self, model: Any, env: Any):
        self.model = model
        self.env = env

    def extract_base_env(self):
        """Extract the base gym environment from potentially
        wrapped/vectorized envs."""
        if hasattr(self.env, "envs"):
            return self.env.envs[0]
        elif hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        return self.env

    def analyze_feature_importance(self, top_k: int = 5, method: str = "saliency") -> Dict[str, float]:
        """
        Run an evaluation episode and attribute feature values to action
        output.

        Args:
            top_k: Number of top features to return.
            method: "saliency" (Input x Gradient) or "correlation" (Pearson).

        Returns:
            Dict mapping feature name to importance score, sorted by magnitude.
        """
        base_env = self.extract_base_env()

        if not hasattr(base_env, "observation_strategy") or not hasattr(
            base_env.observation_strategy, "get_feature_names"
        ):
            raise NotImplementedError(
                "Environment's observation_strategy must implement get_feature_names() "
                "to run feature importance analysis."
            )

        feature_names = base_env.observation_strategy.get_feature_names(base_env)

        # Ensure vectorized environment for model.predict
        if not isinstance(self.env, VecEnv):
            vec_env = DummyVecEnv([lambda e=self.env: e])
        else:
            vec_env = self.env

        obs = vec_env.reset()
        dones = [False]

        observations = []
        actions = []
        attributions = []

        is_recurrent = self.model.__class__.__name__ == "RecurrentPPO"
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

        device = self.model.device

        # RecurrentPPO LSTM states are hard to backprop through — use correlation.
        if is_recurrent and method == "saliency":
            method = "correlation"

        if method == "saliency":
            self.model.policy.set_training_mode(False)

        saliency_failed = False

        try:
            while not dones[0]:
                observations.append(obs[0].copy())

                # --- Deep Learning Attribution (Input x Gradient) ---
                if method == "saliency" and not saliency_failed:
                    try:
                        obs_th = torch.tensor(obs, dtype=torch.float32, device=device, requires_grad=True)
                        dist = self.model.policy.get_distribution(obs_th)
                        action_mean = dist.mode()
                        target_val = torch.atleast_1d(action_mean.view(-1))[0]
                        target_val.backward()
                        grad = obs_th.grad.cpu().numpy()[0]
                        attributions.append(grad * obs[0])
                    except Exception as e:
                        print(f"⚠️ Saliency attribution failed: {e}")
                        saliency_failed = True

                # --- Step environment ---
                if is_recurrent:
                    action, next_lstm_states = self.model.predict(
                        obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                act_val = float(np.atleast_1d(action[0])[0])
                actions.append(act_val)

                obs, _, dones, _ = vec_env.step(action)

                if is_recurrent:
                    lstm_states = next_lstm_states
                    episode_starts = np.zeros((vec_env.num_envs,), dtype=bool)
        finally:
            # Always restore training mode regardless of how the loop exits.
            if method == "saliency":
                self.model.policy.set_training_mode(True)

        obs_matrix = np.array(observations)
        act_array = np.array(actions)
        importance_scores = {}

        if method == "saliency" and not saliency_failed and len(attributions) == len(observations):
            self.last_method_used = "Saliency"
            mean_attr = np.mean(np.abs(attributions), axis=0)
            for i in range(min(len(feature_names), obs_matrix.shape[1])):
                importance_scores[feature_names[i]] = float(mean_attr[i])
        else:
            self.last_method_used = "Correlation"
            for i in range(min(len(feature_names), obs_matrix.shape[1])):
                feat_col = obs_matrix[:, i]
                if np.std(feat_col) > 1e-9:
                    corr = np.corrcoef(feat_col, act_array)[0, 1]
                    if not np.isnan(corr):
                        importance_scores[feature_names[i]] = float(corr)

        sorted_scores = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_scores[:top_k])
