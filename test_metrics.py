#!/usr/bin/env python3
"""
Simple test script to verify Prometheus metrics calculation for branch management
"""

import time
from prometheus_client import Gauge, Counter

# Create mock metrics
try:
    # Try to use existing metrics if available
    from agents.utils import (
        app_active_branches, app_theoretical_token_load,
        app_branch_age_seconds, app_pruned_branches_total,
        app_token_generation_rate
    )
    print("Using existing metrics from agents.utils")
except ImportError:
    # Create new metrics for testing
    print("Creating new metrics for testing")
    app_active_branches = Gauge('test_active_branches', 'Number of active branches')
    app_theoretical_token_load = Gauge('test_theoretical_token_load', 'Theoretical token load')
    app_branch_age_seconds = Gauge('test_branch_age_seconds', 'Branch age in seconds', ['branch_id'])
    app_pruned_branches_total = Counter('test_pruned_branches_total', 'Total pruned branches')
    app_token_generation_rate = Gauge('test_token_generation_rate', 'Token generation rate', ['branch_id'])


class MockBranchRegistry:
    _branches = {}

    @classmethod
    def register_branch(cls, branch):
        cls._branches[branch.branch_id] = branch

    @classmethod
    def unregister_branch(cls, branch):
        if branch.branch_id in cls._branches:
            del cls._branches[branch.branch_id]
            app_branch_age_seconds.remove(branch.branch_id)
            app_token_generation_rate.remove(branch.branch_id)

    @classmethod
    def get_active_branches(cls):
        return [branch for branch in cls._branches.values() if branch.status == 'RUNNING']

    @classmethod
    def _update_metrics(cls):
        active_branches = cls.get_active_branches()
        app_active_branches.set(len(active_branches))
        theoretical_load = sum(branch.total_tokens for branch in active_branches)
        app_theoretical_token_load.set(theoretical_load)
        for branch in active_branches:
            age = time.time() - branch.created_at
            app_branch_age_seconds.labels(branch_id=branch.branch_id).set(age)


class MockBranchTracker:
    def __init__(self, branch_id, parent_id=None):
        self.branch_id = branch_id
        self.parent_id = parent_id
        self.status = 'RUNNING'
        self.created_at = time.time()
        self.last_token_generated_at = self.created_at
        self.prompt_len = 0
        self.gen_len = 0
        self.total_tokens = self.prompt_len
        MockBranchRegistry.register_branch(self)

    def update_status(self, new_status):
        old_status = self.status
        self.status = new_status
        if new_status == 'PRUNED' and old_status != 'PRUNED':
            app_pruned_branches_total.inc()
            MockBranchRegistry.unregister_branch(self)
        elif new_status == 'COMPLETED':
            MockBranchRegistry.unregister_branch(self)

    def increment_gen_len(self, num_tokens=1):
        self.gen_len += num_tokens
        self.total_tokens = self.prompt_len + self.gen_len
        current_time = time.time()
        if current_time > self.last_token_generated_at:
            time_diff = current_time - self.last_token_generated_at
            rate = num_tokens / time_diff
            app_token_generation_rate.labels(branch_id=self.branch_id).set(rate)
        self.last_token_generated_at = current_time


def test_metrics():
    print('Testing metrics computation...')
    print('-' * 50)

    # Create branches
    branch1 = MockBranchTracker('test-branch-1', 'main')
    branch2 = MockBranchTracker('test-branch-2', 'main')
    MockBranchRegistry._update_metrics()
    print(f'After creating 2 branches:')
    print(f'Active branches: {app_active_branches._value.get()}')
    print(f'Theoretical token load: {app_theoretical_token_load._value.get()}')
    print()

    # Update token counts
    branch1.increment_gen_len(100)
    branch2.increment_gen_len(200)
    MockBranchRegistry._update_metrics()
    print(f'After updating token counts:')
    print(f'Theoretical token load: {app_theoretical_token_load._value.get()}')
    print(f'Branch 1 tokens: {branch1.total_tokens}')
    print(f'Branch 2 tokens: {branch2.total_tokens}')
    print()

    # Complete a branch
    time.sleep(0.5)
    branch1.update_status('COMPLETED')
    MockBranchRegistry._update_metrics()
    print(f'After completing branch 1:')
    print(f'Active branches: {app_active_branches._value.get()}')
    print(f'Theoretical token load: {app_theoretical_token_load._value.get()}')
    print()

    # Prune a branch
    time.sleep(0.5)
    branch2.update_status('PRUNED')
    MockBranchRegistry._update_metrics()
    print(f'After pruning branch 2:')
    print(f'Active branches: {app_active_branches._value.get()}')
    print(f'Theoretical token load: {app_theoretical_token_load._value.get()}')
    print(f'Total pruned branches: {app_pruned_branches_total._value.get()}')
    print()

    print("\nAll tests completed!")


if __name__ == "__main__":
    test_metrics()