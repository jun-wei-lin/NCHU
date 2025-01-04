import pandas as pd
import streamlit as st
from modules.behavior_tracking import BehaviorTracker

def analyze_behavior(tracker):
    data = tracker.get_data()
    if not data:
        st.write("No user behavior data available yet.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    st.write("User Behavior Data", df)

    # Display activity count
    action_counts = df['action'].value_counts()
    st.bar_chart(action_counts)

def main():
    tracker = BehaviorTracker()
    
    # Simulate logging
    tracker.log_behavior("user_1", "search", {"query": "AI"})
    tracker.log_behavior("user_2", "click", {"item_id": "12345"})
    
    # Analyze behavior
    st.title("User Behavior Analysis")
    analyze_behavior(tracker)

if __name__ == "__main__":
    main()
