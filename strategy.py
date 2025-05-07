import numpy as np
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

from typing import List

# Arms
# Will define arms as a discretized range of proportions of the maximum allowable bid, in increments of 0.05
ARMS: List[float] = list(np.arange(0.05, 1.05, 0.05))

def get_name() -> str:
    return "vishnu_style_bandit"

def calculate_budget_constrained_max_bid(value: float, totalValue: float, totalPayment: float, ROI: float) -> float:
    """
    Calculate the maximum bid based on the budget constraint.
    """
    if totalPayment == 0:
        return value
    else:
        return min(value, (totalValue + value) / (ROI) - totalPayment)
    
import pandas as pd

def history_to_df(myHistory: list[list]) -> pd.DataFrame:
    if not myHistory:
        return pd.DataFrame(columns=["value", "bid", "win", "payment"])

    df = pd.DataFrame(myHistory, columns=["value", "bid", "win", "payment"])
    return df

def preprocess_auction_history(df: pd.DataFrame, calculate_budget_constrained_max_bid, roi: float) -> pd.DataFrame:
    """
    Preprocess the auction history DataFrame to compute maximum bids, actions, and rewards.
    """
    # Initialize cumulative state prior to any rounds.
    cum_value = 0.0
    cum_payment = 0.0

    max_bid_list = []
    action_list = []
    reward_list = []
    
    # Process each round in sequence.
    for idx, row in df.iterrows():
        current_value = row['value']
        
        # Compute the maximum bid for this round using the state prior to this round.
        max_bid_r = calculate_budget_constrained_max_bid(
            value=current_value,
            totalValue=cum_value,
            totalPayment=cum_payment,
            ROI=roi
        )
        max_bid_list.append(max_bid_r)
        
        # Express the agent's decision as a fraction of the maximum bid.
        # If the max bid is zero, then set action to zero.
        if max_bid_r > 0:
            action = row['bid'] / max_bid_r
        else:
            action = 0.0
        
        action = min(ARMS, key=lambda x: abs(x - action))
        
        action_list.append(action)
        
        # Define reward: if win == 1, reward equals the round's value; otherwise 0.
        reward = current_value if row['win'] == 1 else 0.0
        reward_list.append(reward)
        
        # Update the cumulative state if the auction was won.
        if row['win'] == 1:
            cum_value += current_value
            cum_payment += row['payment']
    
    # Add the new columns to the DataFrame.
    df['max_bid'] = max_bid_list
    df['action'] = action_list
    df['reward'] = reward_list
    
    return df

def strategy(value: float, totalValue: float, totalPayment: float, ROI: float, myHistory: List[List[float]]) -> float:
    current_max_bid = calculate_budget_constrained_max_bid(value, totalValue, totalPayment, ROI)

    # preprocess data
    history_df = history_to_df(myHistory)
    train_df = preprocess_auction_history(history_df, calculate_budget_constrained_max_bid, ROI)
    
    # fit the model 
    model = MAB(arms=ARMS,learning_policy=LearningPolicy.LinUCB(alpha=1.25))

    model.fit(
        decisions=train_df['action'],
        rewards=train_df['reward'],
        contexts=train_df[['max_bid', 'value']],
    )

    # create test data
    test_df = pd.DataFrame({
        'max_bid': [current_max_bid],
        'value': [value],
    })

    return model.predict(test_df) * current_max_bid
