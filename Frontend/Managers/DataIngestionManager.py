import pandas as pd
import numpy as np
import datetime

class DataIngestionManager:
    def __init__(self, df):
        self.original_data = df.copy()
        self.data = df.copy()
        self.report_log = []
        self.processing_steps = []
        self.missing_summary = {}
        self.format_summary = {}
        self.outlier_summary = {}

    #runs quality audit on loaded data
    def smart_quality_audit(self):

        #calculates missing value percentages
        missing_percent = (self.data.isnull().sum() / len(self.data)) * 100
        self.missing_summary = missing_percent.round(2).to_dict()

        #checks format integrity for each column
        for col in self.data.columns:

            if self.data[col].dtype == "object":
                try:
                    pd.to_numeric(self.data[col])
                    self.format_summary[col] = "Stored as text but numeric detected"
                except:
                    pass

            if "email" in col.lower():
                invalid = self.data[~self.data[col].astype(str).str.contains(
                    r"^[\w\.-]+@[\w\.-]+\.\w+$", na=False
                )]
                if len(invalid) > 0:
                    self.format_summary[col] = f"{len(invalid)} invalid email formats"

            if "date" in col.lower():
                try:
                    pd.to_datetime(self.data[col])
                    self.format_summary[col] = "Valid date format"
                except:
                    self.format_summary[col] = "Invalid date format"

        #detects outliers using IQR on numeric columns
        numeric_cols = self.data.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = self.data[
                (self.data[col] < (Q1 - 1.5 * IQR)) |
                (self.data[col] > (Q3 + 1.5 * IQR))
            ]

            if len(outliers) > 0:
                self.outlier_summary[col] = len(outliers)
                self.report_log.append(
                    f"Outlier Detection: {len(outliers)} outliers detected in '{col}'"
                )

    #imputes missing values and removes duplicates
    def auto_preprocess(self):

        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:

                if self.data[col].dtype in ["int64", "float64"]:
                    fill_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(fill_val)
                    self.processing_steps.append(
                        f"Filled missing in '{col}' with median ({round(fill_val,2)})"
                    )

                else:
                    fill_val = self.data[col].mode()[0]
                    self.data[col] = self.data[col].fillna(fill_val)
                    self.processing_steps.append(
                        f"Filled missing in '{col}' with mode ({fill_val})"
                    )

        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)

        if before != after:
            self.processing_steps.append(
                f"Removed {before - after} duplicate rows"
            )

        self.data["data_lineage"] = "Auto_Validated_v2"
        self.data["audit_timestamp"] = datetime.datetime.now()

        return self.data

    #generates audit report dictionary
    def generate_audit_report(self):

        report = {
            "missing_summary": self.missing_summary,
            "format_issues": self.format_summary,
            "outliers": self.outlier_summary,
            "actions_taken": self.processing_steps,
            "audit_notes": self.report_log
        }

        return report