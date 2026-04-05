import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules


class MarketBasketEngine:
    def __init__(self, min_support=0.05, min_confidence=0.4, min_lift=1.3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    #builds boolean basket matrix from order-department data
    def build_basket(self, df):
        basket = (
            df.groupby(['Order_ID', 'Department_Name'])['Department_Name']
            .count()
            .unstack()
            .fillna(0)
            .map(lambda x: 1 if x > 0 else 0)
            .astype(bool)
        )
        return basket

    #generates frequent itemsets and association rules
    def generate_rules(self, df):
        basket = self.build_basket(df)

        frequent_itemsets = fpgrowth(
            basket,
            min_support=self.min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            return pd.DataFrame(), frequent_itemsets, basket

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence
        )

        rules = rules[rules['lift'] >= self.min_lift]

        if rules.empty:
            return pd.DataFrame(), frequent_itemsets, basket

        #formats rules for display
        rules['antecedents'] = rules['antecedents'].apply(
            lambda x: ', '.join(sorted(list(x)))
        )
        rules['consequents'] = rules['consequents'].apply(
            lambda x: ', '.join(sorted(list(x)))
        )

        rules = rules.sort_values(by='lift', ascending=False)

        return rules, frequent_itemsets, basket