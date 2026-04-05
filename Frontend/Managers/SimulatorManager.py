import numpy as np
import joblib
import os

class SimulatorManager:

    def __init__(self, df):
        self.df = df.copy()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.churn_model = joblib.load(
            os.path.join(BASE_DIR, '..', '..', 'Models', 'Outputs', 'churn_model.pkl')
        )
        self.cltv_model = joblib.load(
            os.path.join(BASE_DIR, '..', '..', 'Models', 'Outputs', 'cltv_model.pkl')
        )

    #simulates adjusted features and returns predictions
    def simulate(self, cashback=0, complaints_red=0,
                 price_hike=0, subscription_push=0):

        if cashback == 0 and complaints_red == 0 and price_hike == 0 and subscription_push == 0:
            return None, None, None

        sim_df = self.df.copy()
        original_dtypes = sim_df.dtypes.to_dict()

        #applies cashback adjustment and updates satisfaction and loyalty
        if cashback != 0:
            new_cashback = sim_df['Average_Cash_Back'].astype(float) * (1 + cashback / 100)
            sim_df['Average_Cash_Back'] = new_cashback.clip(lower=0).round().astype(np.int64)

            if cashback > 0:
                boost = min(cashback * 0.02, 0.5)
                new_sat = sim_df['Satisfaction_Score'].astype(float) + boost
                sim_df['Satisfaction_Score'] = new_sat.clip(upper=5).round().astype(np.int64)

                new_loyalty = sim_df['Loyalty_Engagement_Score'].astype(float) + cashback * 0.01
                sim_df['Loyalty_Engagement_Score'] = new_loyalty.round().astype(np.int64)
            else:
                drop = min(abs(cashback) * 0.015, 0.5)
                new_sat = sim_df['Satisfaction_Score'].astype(float) - drop
                sim_df['Satisfaction_Score'] = new_sat.clip(lower=1).round().astype(np.int64)

                new_loyalty = sim_df['Loyalty_Engagement_Score'].astype(float) - abs(cashback) * 0.008
                sim_df['Loyalty_Engagement_Score'] = new_loyalty.clip(lower=0).round().astype(np.int64)

        #applies complaint reduction and updates satisfaction
        if complaints_red > 0:
            new_complaints = sim_df['Complain_Count_In_Last_Month'].astype(float) * (1 - complaints_red / 100)
            sim_df['Complain_Count_In_Last_Month'] = new_complaints.clip(lower=0).round().astype(np.int64)

            boost = min(complaints_red * 0.015, 0.5)
            new_sat = sim_df['Satisfaction_Score'].astype(float) + boost
            sim_df['Satisfaction_Score'] = new_sat.clip(upper=5).round().astype(np.int64)

        #applies price hike adjustment and updates spend and recency
        if price_hike != 0:
            new_spent = sim_df['Total_Amount_Spent'].astype(float) * (1 + price_hike / 100)
            sim_df['Total_Amount_Spent'] = new_spent.clip(lower=0).round().astype(np.int64)

            if price_hike > 0:
                new_hike = sim_df['Order_Amount_Hike_From_Last_Year'].astype(float) + price_hike * 0.5
                sim_df['Order_Amount_Hike_From_Last_Year'] = new_hike.round().astype(np.int64)

                factor = min(price_hike * 0.03, 0.5)
                new_days = sim_df['Days_Since_Last_Order'].astype(float) * (1 - factor)
                sim_df['Days_Since_Last_Order'] = new_days.clip(lower=0).round().astype(np.int64)
            else:
                factor = min(abs(price_hike) * 0.02, 0.3)
                new_days = sim_df['Days_Since_Last_Order'].astype(float) * (1 + factor)
                sim_df['Days_Since_Last_Order'] = new_days.round().astype(np.int64)

        #applies subscription push and updates loyalty for converted customers
        if subscription_push > 0:
            subs_str = sim_df['Total_Subscriptions'].astype(str).str.strip()
            mask_one = subs_str == '1'
            eligible = mask_one.sum()

            if eligible > 0:
                convert_count = min(int(eligible * (subscription_push / 100)), eligible)

                if convert_count > 0:
                    convert_idx = sim_df[mask_one].sample(
                        convert_count, random_state=42
                    ).index

                    original_subs = self.df['Total_Subscriptions'].astype(str).str.strip()
                    non_one = self.df[original_subs != '1']['Total_Subscriptions']

                    if len(non_one) > 0:
                        replacement_val = non_one.iloc[0]
                    else:
                        replacement_val = 'More than 1'

                    sim_df.loc[convert_idx, 'Total_Subscriptions'] = replacement_val

            subs_after = sim_df['Total_Subscriptions'].astype(str).str.strip()
            multi_mask = subs_after != '1'

            if multi_mask.any():
                loyalty_vals = sim_df['Loyalty_Engagement_Score'].astype(float).copy()
                loyalty_vals[multi_mask] += subscription_push * 0.008
                sim_df['Loyalty_Engagement_Score'] = loyalty_vals.round().astype(np.int64)

        #generates churn and cltv predictions on simulated data
        sim_churn_probs = self.churn_model.predict_proba(sim_df)[:, 1]
        sim_clv_log = self.cltv_model.predict(sim_df)
        sim_clv_dollars = np.expm1(sim_clv_log)

        return sim_df, sim_churn_probs, sim_clv_dollars