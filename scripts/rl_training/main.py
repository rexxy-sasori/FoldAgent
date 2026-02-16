import importlib
import os
from pathlib import Path


def main():
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


def _import_all_agent_loops():
    """Import all agent loops from agent_loops/ directory to ensure registration.
    
    This is necessary for Ray distributed training because Ray workers run in 
    separate processes and need to import custom agent loops to register them.
    """
    agent_loops_dir = Path(__file__).parent / 'agent_loops'
    
    for file_path in agent_loops_dir.glob('*.py'):
        if file_path.name.startswith('_'):
            continue
        
        module_name = f'scripts.rl_training.agent_loops.{file_path.stem}'
        try:
            importlib.import_module(module_name)
        except Exception as e:
            print(f'Warning: Failed to import {module_name}: {e}')


if __name__ == "__main__":
    _import_all_agent_loops()
    main()
