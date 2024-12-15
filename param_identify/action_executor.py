import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import subprocess
from threading import Thread


class ActionExecutor:
    def __init__(self, working_dir="./executor_workspace", 
                 check_interval=3, 
                 read_interval=1,
                 execute_timeout=100,
                 max_wait_attempts=3):
        self.working_dir = Path(working_dir)
        self.status_file = self.working_dir / "status.json"  # Executor status file
        self.policy_file = self.working_dir / "current_policy.json"  # Policy file
        self.read_flag_file = self.working_dir / "policy_read.json"  # Policy read flag
        self.execution_flag_file = self.working_dir / "execution_completed.json"  # Execution completed flag
        
        self.check_interval = check_interval  # Interval for checking executor existence
        self.read_interval = read_interval    # Interval for checking policy read status
        self.execute_timeout = execute_timeout  # Execution timeout period
        self.max_wait_attempts = max_wait_attempts  # Maximum number of wait attempts
        
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._write_status(False)
        
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None

    def _monitor_pipe(self, pipe, prefix=''):
        try:
            while True:
                line = pipe.readline()
                if line:
                    print(f"{prefix}{line.strip()}")
                elif self.process.poll() is not None:
                    break
        except Exception as e:
            print(f"Error in monitoring: {e}")

    def start_executor_process(self):
        """Start executor subprocess"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        executor_path = os.path.join(current_dir, "executor_implement.py")
        
        # Modify environment variables to force Python not to use buffering
        my_env = os.environ.copy()
        my_env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            ["python", "-u", executor_path],  # Add -u parameter to disable buffering
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=my_env,
            universal_newlines=True
        )

        # Start two monitoring threads for stdout and stderr
        self.stdout_thread = Thread(target=self._monitor_pipe, 
                                  args=(self.process.stdout, "OUT: "))
        self.stderr_thread = Thread(target=self._monitor_pipe, 
                                  args=(self.process.stderr, "ERR: "))
        
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _write_status(self, active: bool):
        """Write executor status"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump({"executor_active": active}, f)
        except Exception as e:
            print(f"Failed to write status: {str(e)}")

    def check_executor(self) -> bool:
        """Check if executor is online"""
        try:
            if not self.status_file.exists():
                return False
                
            with open(self.status_file, 'r') as f:
                status = json.load(f)
                return status.get("executor_active", False)
        except Exception as e:
            print(f"Failed to check executor status: {str(e)}")
            return False
    
    def write_policy(self, policy: np.ndarray) -> bool:
        """Write policy"""
        try:
            policy_data = {
                "policy": policy.tolist(),
                "timestamp": time.time()
            }
            with open(self.policy_file, 'w') as f:
                json.dump(policy_data, f)
            return True
        except Exception as e:
            print(f"Failed to write policy: {str(e)}")
            return False

    def check_policy_read(self) -> bool:
        """Check if policy has been read"""
        try:
            if not self.read_flag_file.exists():
                return False
                
            with open(self.read_flag_file, 'r') as f:
                flag = json.load(f)
                return flag.get("read", False)
        except Exception as e:
            print(f"Failed to check read flag: {str(e)}")
            return False

    def check_execution_completed(self) -> bool:
        """Check if execution is completed"""
        try:
            if not self.execution_flag_file.exists():
                return False
                
            with open(self.execution_flag_file, 'r') as f:
                flag = json.load(f)
                return flag.get("completed", False)
        except Exception as e:
            print(f"Failed to check execution flag: {str(e)}")
            return False

    def clear_flags(self):
        """Clear all flag files"""
        for file in [self.read_flag_file, self.execution_flag_file]:
            if file.exists():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Failed to clear flag file {file}: {str(e)}")

    def wait_for_executor(self) -> bool:
        """Wait for executor to come online"""
        attempts = 0
        while attempts < self.max_wait_attempts:
            if self.check_executor():
                return True
            print(f"Waiting for executor to come online... Attempt {attempts + 1}/{self.max_wait_attempts}")
            time.sleep(self.check_interval)
            attempts += 1
        return False

    def wait_for_policy_read(self) -> bool:
        """Wait for policy to be read"""
        attempts = 0
        while attempts < self.max_wait_attempts:
            if self.check_policy_read():
                return True
            print(f"Waiting for policy to be read... Attempt {attempts + 1}/{self.max_wait_attempts}")
            time.sleep(self.read_interval)
            attempts += 1
        return False

    def execute_action(self, policy: np.ndarray) -> bool:
        """Main function for executing actions"""
        # Start executor process
        self.start_executor_process()
        
        # Wait for executor to come online
        if not self.wait_for_executor():
            print("Executor failed to come online after maximum attempts")
            return False

        # Clear previous flags
        self.clear_flags()
        
        # Write policy
        if not self.write_policy(policy):
            return False

        # Wait for policy to be read
        if not self.wait_for_policy_read():
            print("Policy was not read after maximum attempts")
            return False

        # Wait for execution to complete
        start_time = time.time()
        while time.time() - start_time < self.execute_timeout:
            if self.check_execution_completed():
                # Clear execution completion flag, keep executor status
                self.execution_flag_file.unlink()
                return True
            time.sleep(0.5)

        print(f"Execution timeout after {self.execute_timeout} seconds")
        return False


if __name__ == "__main__":
    executor = ActionExecutor(
        check_interval=3,
        read_interval=1,
        execute_timeout=100,
        max_wait_attempts=3
    )

    policy = np.array([1.0, 2.0, 1.0])
    success = executor.execute_action(policy)
    print(f"Action execution {'succeeded' if success else 'failed'}")

    # Give some time for output to complete
    time.sleep(1)