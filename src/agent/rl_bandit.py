import random
from collections import defaultdict
from typing import Any

from src.core.logger import LoggerFactory
from src.core.model import ResponseStyle, UserContext


class ContextualBandit:
    """Epsilon-greedy contextual bandit for response style optimization"""

    def __init__(self, epsilon: float = 0.1, log_level: str = "INFO"):
        """
        Initialize contextual bandit

        Args:
            epsilon: Exploration rate (0-1). Higher = more exploration
            log_level: Logging level
        """
        self.epsilon = max(0.0, min(1.0, epsilon))  # Clamp to [0, 1]
        self.logger = LoggerFactory.create_logger("RLBandit", level=log_level)

        # Arms (actions) are response styles
        self.arms = [
            ResponseStyle.CONCISE,
            ResponseStyle.DETAILED,
            ResponseStyle.VERBOSE
        ]

        # Track rewards for each (context, arm) combination
        self.arm_rewards: dict[tuple, list[float]] = defaultdict(list)
        self.arm_counts: dict[tuple, int] = defaultdict(int)

        # Track total interactions
        self.total_interactions = 0

        self.logger.info(f"Initialized Contextual Bandit with epsilon={self.epsilon}")

    def _extract_context_features(self, context: UserContext) -> tuple:
        """
        Extract discrete features from user context

        Args:
            context: User context with continuous features

        Returns:
            Tuple of discrete features for bucketing
        """
        # Discretize time of day into periods
        if context.time_of_day < 12:
            time_period = "morning"
        elif context.time_of_day < 18:
            time_period = "afternoon"
        else:
            time_period = "evening"

        # Discretize interaction count
        if context.interaction_count < 5:
            interaction_level = "low"
        elif context.interaction_count < 20:
            interaction_level = "medium"
        else:
            interaction_level = "high"

        # Discretize error rate
        if context.recent_errors == 0:
            error_level = "none"
        elif context.recent_errors < 3:
            error_level = "some"
        else:
            error_level = "many"

        return (time_period, interaction_level, error_level)

    def select_arm(self, context: UserContext) -> ResponseStyle:
        """
        Select response style using epsilon-greedy policy

        Args:
            context: Current user context

        Returns:
            Selected response style
        """
        features = self._extract_context_features(context)

        # Check for user preference override
        if context.preferred_style:
            self.logger.debug(f"Using preferred style: {context.preferred_style}")
            return context.preferred_style

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: random selection
            arm = random.choice(self.arms)
            self.logger.debug(
                f"Exploring: selected {arm.value} for context {features}"
            )
            return arm
        else:
            # Exploit: choose best arm based on average reward
            best_arm = self._get_best_arm(features)
            self.logger.debug(
                f"Exploiting: selected {best_arm.value} for context {features}"
            )
            return best_arm

    def _get_best_arm(self, features: tuple) -> ResponseStyle:
        """
        Get the arm with highest average reward for given features

        Args:
            features: Context feature tuple

        Returns:
            Best performing arm, or random if no data
        """
        best_arm = None
        best_avg_reward = -float('inf')

        for arm in self.arms:
            key = (features, arm.value)
            if key in self.arm_rewards and len(self.arm_rewards[key]) > 0:
                avg_reward = sum(self.arm_rewards[key]) / len(self.arm_rewards[key])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_arm = arm

        # If no data, return random arm (pure exploration)
        if best_arm is None:
            best_arm = random.choice(self.arms)
            self.logger.debug(
                f"No data for context {features}, selecting random arm: {best_arm.value}"
            )

        return best_arm

    def update_reward(
        self,
        context: UserContext,
        arm: ResponseStyle,
        reward: float
    ) -> None:
        """
        Update arm statistics with observed reward

        Args:
            context: User context when arm was selected
            arm: The arm (response style) that was used
            reward: Observed reward (0-1, higher is better)
        """
        features = self._extract_context_features(context)
        key = (features, arm.value)

        # Clamp reward to valid range
        reward = max(0.0, min(1.0, reward))

        # Update statistics
        self.arm_rewards[key].append(reward)
        self.arm_counts[key] += 1
        self.total_interactions += 1

        avg_reward = sum(self.arm_rewards[key]) / len(self.arm_rewards[key])

        self.logger.debug(
            f"Updated {arm.value} with reward {reward:.3f} "
            f"for context {features} (avg: {avg_reward:.3f}, count: {self.arm_counts[key]})"
        )

    def calculate_implicit_reward(
        self,
        response_time: float,
        error_occurred: bool,
        user_repeated: bool
    ) -> float:
        """
        Calculate implicit reward from interaction signals

        Args:
            response_time: Time taken to respond (seconds)
            error_occurred: Whether an error occurred during interaction
            user_repeated: Whether user had to repeat their input

        Returns:
            Reward value between 0 and 1
        """
        reward = 1.0  # Start with max reward

        # Penalty for slow response (> 3 seconds)
        if response_time > 3.0:
            penalty = min(0.3, (response_time - 3.0) * 0.1)
            reward -= penalty
            self.logger.debug(f"Response time penalty: -{penalty:.3f} ({response_time:.2f}s)")

        # Penalty for errors
        if error_occurred:
            reward -= 0.5
            self.logger.debug("Error penalty: -0.5")

        # Penalty if user had to repeat
        if user_repeated:
            reward -= 0.4
            self.logger.debug("Repeat penalty: -0.4")

        # Clamp to valid range
        reward = max(0.0, min(1.0, reward))

        return reward

    def get_statistics(self) -> dict[str, Any]:
        """
        Get bandit statistics for monitoring

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_interactions": self.total_interactions,
            "epsilon": self.epsilon,
            "num_contexts": len(set(k[0] for k in self.arm_counts.keys())),
            "arm_counts": {},
            "arm_averages": {}
        }

        # Aggregate counts and averages per arm
        for arm in self.arms:
            total_count = sum(
                count for (features, arm_val), count in self.arm_counts.items()
                if arm_val == arm.value
            )
            stats["arm_counts"][arm.value] = total_count

            # Calculate overall average reward for this arm
            all_rewards = []
            for (features, arm_val), rewards in self.arm_rewards.items():
                if arm_val == arm.value:
                    all_rewards.extend(rewards)

            if all_rewards:
                stats["arm_averages"][arm.value] = sum(all_rewards) / len(all_rewards)
            else:
                stats["arm_averages"][arm.value] = 0.0

        return stats

    def get_context_statistics(self, context: UserContext) -> dict[str, Any]:
        """
        Get statistics for a specific context

        Args:
            context: User context to query

        Returns:
            Dictionary with context-specific statistics
        """
        features = self._extract_context_features(context)

        stats = {
            "features": features,
            "arm_performance": {}
        }

        for arm in self.arms:
            key = (features, arm.value)
            if key in self.arm_rewards and len(self.arm_rewards[key]) > 0:
                avg_reward = sum(self.arm_rewards[key]) / len(self.arm_rewards[key])
                stats["arm_performance"][arm.value] = {
                    "avg_reward": avg_reward,
                    "count": self.arm_counts[key]
                }
            else:
                stats["arm_performance"][arm.value] = {
                    "avg_reward": None,
                    "count": 0
                }

        return stats

    def set_epsilon(self, epsilon: float) -> None:
        """
        Update exploration rate

        Args:
            epsilon: New exploration rate (0-1)
        """
        self.epsilon = max(0.0, min(1.0, epsilon))
        self.logger.info(f"Updated epsilon to {self.epsilon}")

    def reset(self) -> None:
        """Reset all learned statistics"""
        self.arm_rewards.clear()
        self.arm_counts.clear()
        self.total_interactions = 0
        self.logger.info("Reset all bandit statistics")

    def __repr__(self) -> str:
        return f"ContextualBandit(epsilon={self.epsilon}, interactions={self.total_interactions})"
