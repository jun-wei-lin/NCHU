import datetime

class BehaviorTracker:
    def __init__(self):
        self.behavior_data = []

    def log_behavior(self, user_id, action, metadata=None):
        timestamp = datetime.datetime.now()
        entry = {
            "user_id": user_id,
            "action": action,
            "metadata": metadata,
            "timestamp": timestamp,
        }
        self.behavior_data.append(entry)

    def get_data(self):
        return self.behavior_data
