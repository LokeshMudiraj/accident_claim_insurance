
import numpy as np
import pandas as pd
from pathlib import Path


def generate_scaled_claims_dataset(
    input_csv_path: str | None = None,
    output_csv_path: str = "data/accident_claims.csv",
    n_rows: int = 10000,
    age_min: int = 18,
    age_max: int = 70,
    amount_min: int = 2000,
    amount_max: int = 200000,
    random_state: int = 42,
) -> pd.DataFrame:

    """
    Generate an insurance claims dataset for Random Forest training.

    Features:
        Age
        Claim_Amount
        Incident_Severity

    Target:
        Claim_Status

    Output:
        Exactly n_rows records.
    """

    # --------------------------------------------------
    # 1. VALIDATION
    # --------------------------------------------------

    if n_rows <= 0:
        raise ValueError("n_rows must be greater than zero.")

    if age_min >= age_max:
        raise ValueError("age_min must be smaller than age_max.")

    if amount_min <= 0 or amount_min >= amount_max:
        raise ValueError("Invalid claim amount range.")

    rng = np.random.default_rng(random_state)

    # --------------------------------------------------
    # 2. ORIGINAL SUBGROUP DISTRIBUTIONS
    # --------------------------------------------------

    # (Severity, Status, Joint Probability,
    #  Age Mean, Age Std, Amount Mean, Amount Std)

    subgroup_specs = [
        (1, 0, 0.055, 45.05, 15.37, 2415.73, 1268.06),
        (1, 1, 0.213, 45.42, 14.61, 2090.19, 1019.49),

        (2, 0, 0.086, 40.53, 14.88, 4596.52, 1121.41),
        (2, 1, 0.144, 41.51, 14.47, 4366.30, 1126.11),

        (3, 0, 0.137, 43.84, 15.26, 7286.14, 1208.17),
        (3, 1, 0.124, 44.55, 15.64, 6880.65, 1248.11),

        (4, 0, 0.099, 43.86, 15.18, 10988.54, 1211.96),
        (4, 1, 0.047, 45.11, 15.93, 11084.66, 1190.17),

        (5, 0, 0.085, 43.56, 14.30, 16074.64, 1026.35),
        (5, 1, 0.010, 50.90, 13.12, 15115.10, 1318.58),
    ]

    # --------------------------------------------------
    # 3. TRUNCATED NORMAL GENERATOR
    # --------------------------------------------------

    # Unlike np.clip(), rejection sampling avoids creating
    # large artificial groups at the minimum and maximum.

    def generate_bounded_normal(mean, std, size, low, high):

        values = rng.normal(mean, std, size)

        invalid = (values < low) | (values > high)

        attempts = 0

        while invalid.any() and attempts < 100:

            values[invalid] = rng.normal(
                mean,
                std,
                invalid.sum()
            )

            invalid = (values < low) | (values > high)

            attempts += 1

        # Fallback for extreme distributions.
        values = np.clip(values, low, high)

        return values

    # --------------------------------------------------
    # 4. AMOUNT SCALING
    # --------------------------------------------------

    original_amount_min = 800.0
    original_amount_max = 18052.0

    def rescale_amount(values):

        scaled = amount_min + (
            (values - original_amount_min)
            / (original_amount_max - original_amount_min)
        ) * (amount_max - amount_min)

        return np.clip(
            np.round(scaled),
            amount_min,
            amount_max
        ).astype(int)

    # --------------------------------------------------
    # 5. GENERATE DATA
    # --------------------------------------------------

    if input_csv_path is not None:

        # Bootstrap from an existing real dataset.
        df_orig = pd.read_csv(input_csv_path)

        required_columns = [
            "Age",
            "Claim_Amount",
            "Incident_Severity",
            "Claim_Status"
        ]

        missing = set(required_columns) - set(df_orig.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        df_orig = df_orig[required_columns].copy()

        if df_orig.empty or df_orig.isnull().any().any():
            raise ValueError(
                "Input dataset is empty or contains missing values."
            )

        # Validate target and severity.
        if not df_orig["Claim_Status"].isin([0, 1]).all():
            raise ValueError("Claim_Status must contain only 0 and 1.")

        if not df_orig["Incident_Severity"].isin(
            [1, 2, 3, 4, 5]
        ).all():
            raise ValueError("Incident_Severity must be between 1 and 5.")

        # Convert original amount range into the requested range.
        df_orig["Claim_Amount"] = rescale_amount(
            df_orig["Claim_Amount"].to_numpy(dtype=float)
        )

        df_orig["Age"] = np.clip(
            df_orig["Age"].to_numpy(dtype=float),
            age_min,
            age_max
        ).astype(int)

        # Generate exactly n_rows records.
        indices = rng.choice(
            len(df_orig),
            size=n_rows,
            replace=(n_rows > len(df_orig))
        )

        final_df = df_orig.iloc[indices].copy().reset_index(drop=True)

        # Add small controlled noise to numeric features.
        age_noise = rng.normal(0, 2.0, n_rows)

        amount_noise = rng.normal(
            0,
            max(500, (amount_max - amount_min) * 0.015),
            n_rows
        )

        final_df["Age"] = np.clip(
            np.round(final_df["Age"] + age_noise),
            age_min,
            age_max
        ).astype(int)

        final_df["Claim_Amount"] = np.clip(
            np.round(final_df["Claim_Amount"] + amount_noise),
            amount_min,
            amount_max
        ).astype(int)

    else:

        # Normalize subgroup probabilities.
        weights = np.array(
            [spec[2] for spec in subgroup_specs],
            dtype=float
        )

        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid subgroup probabilities.")

        weights /= weights.sum()

        # Allocate records to each subgroup.
        counts = rng.multinomial(n_rows, weights)

        chunks = []

        for spec, count in zip(subgroup_specs, counts):

            if count == 0:
                continue

            (
                severity,
                status,
                _,
                age_mean,
                age_std,
                amount_mean,
                amount_std
            ) = spec

            # Generate age without excessive boundary clipping.
            ages = generate_bounded_normal(
                mean=age_mean,
                std=age_std,
                size=count,
                low=age_min,
                high=age_max
            )

            # Generate claim amounts within original range.
            raw_amounts = generate_bounded_normal(
                mean=amount_mean,
                std=amount_std,
                size=count,
                low=original_amount_min,
                high=original_amount_max
            )

            amounts = rescale_amount(raw_amounts)

            chunks.append(
                pd.DataFrame({
                    "Age": np.round(ages).astype(int),
                    "Claim_Amount": amounts,
                    "Incident_Severity": severity,
                    "Claim_Status": status
                })
            )

        final_df = pd.concat(
            chunks,
            ignore_index=True
        )

    # --------------------------------------------------
    # 6. FINAL DATASET PROCESSING
    # --------------------------------------------------

    final_df = final_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    # Ensure exactly the requested columns and row count.
    final_df = final_df[
        [
            "Age",
            "Claim_Amount",
            "Incident_Severity",
            "Claim_Status"
        ]
    ]

    assert len(final_df) == n_rows

    # --------------------------------------------------
    # 7. SAVE DATASET
    # --------------------------------------------------

    output_path = Path(output_csv_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df.to_csv(output_path, index=False)

    # --------------------------------------------------
    # 8. DATASET DIAGNOSTICS
    # --------------------------------------------------

    print("\n========== DATASET GENERATED ==========")

    print(f"Total Records: {len(final_df)}")

    print("\nClass Distribution:")

    print(
        final_df["Claim_Status"]
        .value_counts()
        .sort_index()
    )

    print("\nClass Percentages:")

    print(
        (final_df["Claim_Status"]
         .value_counts(normalize=True)
         .sort_index() * 100).round(2)
    )

    print("\nSeverity vs Claim Status:")

    print(
        pd.crosstab(
            final_df["Incident_Severity"],
            final_df["Claim_Status"],
            normalize="index"
        ).round(3)
    )

    print("\nFeature Statistics:")

    print(final_df.describe().round(2))

    print(f"\nSaved Dataset: {output_path.resolve()}")

    return final_df


# ======================================================
# MAIN EXECUTION
# ======================================================

if __name__ == "__main__":

    df_scaled = generate_scaled_claims_dataset(
        input_csv_path=None,
        output_csv_path="./data/accident_claims.csv",
        n_rows=10000,
        age_min=18,
        age_max=70,
        amount_min=2000,
        amount_max=200000,
        random_state=42
    )

    print("\nFirst 10 Records:")

    print(df_scaled.head(10))
