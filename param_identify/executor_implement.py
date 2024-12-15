import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

class ExecutorServer:
    def __init__(self, working_dir="./executor_workspace", 
                 check_interval=2,
                 process_time=3):
        self.working_dir = Path(working_dir)
        self.status_file = self.working_dir / "status.json"
        self.policy_file = self.working_dir / "current_policy.json"
        self.read_flag_file = self.working_dir / "policy_read.json"
        self.execution_flag_file = self.working_dir / "execution_completed.json"
        
        self.check_interval = check_interval  # Interval for checking new policies
        self.process_time = process_time      # Time to simulate policy processing
        
        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last processed policy timestamp to identify new policies
        self.last_policy_timestamp = 0
        
        # Mark executor as online on startup
        self._mark_active()

    def _mark_active(self):
        """Mark executor as online"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump({"executor_active": True}, f)
        except Exception as e:
            print(f"Failed to mark executor active: {str(e)}")

    def _mark_policy_read(self):
        """Mark policy as read"""
        try:
            with open(self.read_flag_file, 'w') as f:
                json.dump({"read": True}, f)
        except Exception as e:
            print(f"Failed to mark policy as read: {str(e)}")

    def _mark_execution_completed(self):
        """Mark execution as completed"""
        try:
            with open(self.execution_flag_file, 'w') as f:
                json.dump({"completed": True}, f)
        except Exception as e:
            print(f"Failed to mark execution as completed: {str(e)}")

    def read_policy(self) -> Optional[np.ndarray]:
        """Read policy file and return policy if it's new"""
        try:
            if not self.policy_file.exists():
                return None

            with open(self.policy_file, 'r') as f:
                policy_data = json.load(f)
                
            # Check if this is a new policy
            timestamp = policy_data.get("timestamp", 0)
            if timestamp <= self.last_policy_timestamp:
                return None
                
            self.last_policy_timestamp = timestamp
            return np.array(policy_data["policy"])
                
        except Exception as e:
            print(f"Failed to read policy: {str(e)}")
            return None

    def process_policy(self, policy: np.ndarray):
        """Process the policy (simulated with sleep)"""
        print(f"Processing policy: {policy}")
        time.sleep(self.process_time)
        print("Policy execution completed")

    def run(self):
        """Main executor loop"""
        print("Executor server starting...")
        try:
            while True:
                # Check for new policy
                policy = self.read_policy()
                
                if policy is not None:
                    print("New policy detected")
                    
                    # Mark as read
                    self._mark_policy_read()
                    print("Policy marked as read")
                    
                    # Process the policy
                    self.process_policy(policy)
                    
                    # Mark execution as completed
                    self._mark_execution_completed()
                    print("Execution marked as completed")
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\nExecutor server shutting down...")
        finally:
            # Cleanup flag files
            for file in [self.status_file, self.read_flag_file, self.execution_flag_file]:
                if file.exists():
                    file.unlink()
            print("Cleanup completed")

def main():
    # Create and run executor
    executor = ExecutorServer(
        check_interval=2,  # Check for new policies every 2 seconds
        process_time=3     # Simulate processing taking 3 seconds
    )
    executor.run()

if __name__ == "__main__":
    main()