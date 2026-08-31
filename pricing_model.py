"""
Motor insurance pricing model - French MTPL (freMTPL2).

Predicts how often each policy claims and how much each claim costs, combines
them into an expected annual claims cost, then loads that into a premium.

Run:  python3 pricing_model.py

Everything is in this one file, top to bottom, in the order it happens.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 50)


# =====================================================================
# 1. LOAD AND CLEAN
# =====================================================================

def load_data(path="data/fremtpl2.csv"):
    """Read the cached dataset and apply the standard cleaning steps."""
    df = pd.read_csv(path)

    # Exposure is the fraction of a year the policy was on risk. A few rows
    # exceed 1.0, which is impossible for an annual policy.
    df["Exposure"] = df["Exposure"].clip(upper=1.0)

    # A thin tail of policies show 10+ claims. Capping stops a handful of rows
    # dominating the fit.
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)

    # Contradictory rows: a claim with no cost, or a cost with no claim.
    # Neither can contribute to a severity estimate, so zero both consistently.
    bad = (((df["ClaimNb"] > 0) & (df["ClaimAmount"] <= 0)) |
           ((df["ClaimNb"] == 0) & (df["ClaimAmount"] > 0)))
    df.loc[bad, ["ClaimNb", "ClaimAmount"]] = 0

    df = df[df["Exposure"] > 0].copy()

    # Derived columns used throughout.
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = np.where(df["ClaimNb"] > 0,
                                    df["ClaimAmount"] / df["ClaimNb"], 0.0)
    return df


# =====================================================================
# 2. EXPLORE
# =====================================================================

def frequency_by(df, column):
    """Observed claim frequency for each level of `column`.

    Total claims divided by total exposure, within each group. Same
    calculation as the portfolio-wide frequency, just done per group.
    """
    g = df.groupby(column, observed=True)
    return g["ClaimNb"].sum() / g["Exposure"].sum()


def explore(df):
    claims = df[df["ClaimNb"] > 0]

    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"Policies          : {len(df):,}")
    print(f"Total exposure    : {df['Exposure'].sum():,.0f} policy-years")
    print(f"Claims            : {int(df['ClaimNb'].sum()):,}")
    print(f"Frequency         : {df['ClaimNb'].sum() / df['Exposure'].sum():.4f} per year")
    print(f"Policies claiming : {(df['ClaimNb'] > 0).mean():.2%}")
    print(f"Mean severity     : EUR {claims['ClaimAmount'].sum() / claims['ClaimNb'].sum():,.0f}")
    print(f"Median claim      : EUR {claims['ClaimAmount'].median():,.0f}")
    print(f"Largest claim     : EUR {claims['ClaimAmount'].max():,.0f}")

    # Mean far above median is the skew that forces a separate severity model.

    d = df.copy()
    d["age_band"] = pd.cut(d["DrivAge"], AGE_BANDS)
    d["bm_band"] = pd.cut(d["BonusMalus"], BM_BANDS, include_lowest=True)

    print("\nFrequency by driver age:")
    print(frequency_by(d, "age_band").round(4).to_string())
    print("\nFrequency by bonus-malus:")
    print(frequency_by(d, "bm_band").round(4).to_string())
    print("\nPolicies per bonus-malus band:")
    print(d["bm_band"].value_counts().sort_index().to_string())
    print("\nFrequency by area (A rural -> F urban):")
    print(frequency_by(df, "Area").round(4).to_string())

    _plot_exploration(df, d, claims)


def _plot_exploration(df, d, claims):
    """Four charts, saved to outputs/."""
    # Claim cost distribution. Bins must be log-spaced too, not just the axis:
    # equal-width bins in euros put almost every claim in the first bin.
    bins = np.logspace(np.log10(claims["ClaimAmount"].min()),
                       np.log10(claims["ClaimAmount"].max()), 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(claims["ClaimAmount"], bins=bins)
    ax.set_xscale("log")
    ax.set_xlabel("Claim amount (EUR, log scale)")
    ax.set_ylabel("Number of claims")
    ax.set_title("Distribution of claim costs")
    fig.tight_layout()
    fig.savefig("outputs/1_severity_distribution.png", dpi=150)
    plt.close(fig)

    for name, data, fname, rot in [
        ("driver age", frequency_by(d, "age_band"), "2_frequency_by_age", 45),
        ("bonus-malus", frequency_by(d, "bm_band"), "3_frequency_by_bonusmalus", 45),
        ("area", frequency_by(df, "Area"), "4_frequency_by_area", 0),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Claims per year")
        ax.set_xlabel(name)
        ax.set_title(f"Claim frequency by {name}")
        ax.tick_params(axis="x", rotation=rot)
        fig.tight_layout()
        fig.savefig(f"outputs/{fname}.png", dpi=150)
        plt.close(fig)

    print("\n[charts written to outputs/]")


# =====================================================================
# 3. BUILD FEATURES
# =====================================================================

# Band boundaries. Chosen from the exploration above, not by an automatic
# rule: the young-driver effect is concentrated in 18-25, so those bands are
# narrow, while the relationship is flat after 50 so those bands are wide.
# An equal-population or equal-width rule puts no boundary at 21 and averages
# the strongest signal in the data away.
AGE_BANDS = [17, 21, 25, 30, 40, 50, 60, 70, 100]
BM_BANDS = [49, 50, 60, 70, 85, 100, 350]
VEHAGE_BANDS = [-1, 1, 3, 5, 8, 12, 20, 100]
POWER_BANDS = [0, 5, 6, 7, 8, 10, 20]


def build_features(df):
    """Turn the raw columns into a numeric matrix the GLM can use.

    A GLM multiplies each column by a coefficient and adds the results, so
    every column has to be a number. Two conversions are needed:

      - Categories ("Area F") become one 0/1 column per level.
      - Continuous columns are banded first, then one-hot encoded, because
        their relationship with claims is not a straight line.

    pd.get_dummies does the one-hot encoding. drop_first=True drops one level
    of each factor to act as the baseline that every relativity is measured
    against.
    """
    X = pd.DataFrame(index=df.index)

    X["age"] = pd.cut(df["DrivAge"], AGE_BANDS)
    X["bonus_malus"] = pd.cut(df["BonusMalus"], BM_BANDS, include_lowest=True)
    X["veh_age"] = pd.cut(df["VehAge"], VEHAGE_BANDS)
    X["veh_power"] = pd.cut(df["VehPower"], POWER_BANDS)

    # Density spans 1 to 30,000, so log it before banding.
    X["density"] = pd.qcut(np.log1p(df["Density"]), 6, duplicates="drop")

    X["area"] = df["Area"]
    X["fuel"] = df["VehGas"]
    X["region"] = df["Region"]
    X["brand"] = df["VehBrand"]

    return pd.get_dummies(X, drop_first=True, dtype=float)


def align(X_train, X_other):
    """Make sure both matrices have the same columns in the same order.

    A rare region present in training but absent from test would otherwise
    produce a different number of columns and the model would refuse to
    predict.
    """
    return X_other.reindex(columns=X_train.columns, fill_value=0.0)


# =====================================================================
# 4. FIT THE MODELS
# =====================================================================

def fit_frequency(X, df, alpha=1e-3):
    """Poisson GLM for claims per year.

    THE KEY IDEA IN THIS PROJECT. Exposure is how long a policy was on risk.
    A policy held three months has had a quarter of the chance to produce a
    claim as one held all year. Ignore that and short policies look safe when
    they are simply young.

    The standard treatment is an offset of log(Exposure). scikit-learn has no
    offset argument, so the equivalent form is used: model the RATE
    (ClaimNb / Exposure) and weight each row by its Exposure.
    """
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(X, df["Frequency"], sample_weight=df["Exposure"])
    return model


def fit_severity(X, df, alpha=1e-2):
    """Gamma GLM for cost per claim.

    Two things worth defending:

      - Fitted only on policies that actually claimed (~5% of the book).
        "How much does a claim cost" is undefined for a policy with no claim,
        and including 95% zeros answers a different question.
      - Weighted by claim count. A policy with three claims is three times as
        much evidence about severity as one with a single claim.

    Gamma rather than ordinary least squares because claim amounts are strictly
    positive, right-skewed, and their variance grows with their mean. OLS
    assumes constant variance and can predict a negative claim cost.
    """
    mask = (df["ClaimNb"] > 0) & (df["AvgClaimAmount"] > 0)
    model = GammaRegressor(alpha=alpha, max_iter=1000)
    model.fit(X[mask.values], df.loc[mask, "AvgClaimAmount"],
              sample_weight=df.loc[mask, "ClaimNb"])
    print(f"[severity] fitted on {mask.sum():,} claiming policies "
          f"({mask.mean():.1%} of the book)")
    return model


def relativities(model, columns):
    """Convert coefficients into rating multipliers.

    Both models use a log link, which makes their effects multiply rather than
    add. That is why exp(coefficient) is the multiplier:

        coefficient  0.30  ->  exp(0.30) = 1.35  ->  35% more claims
        coefficient  0.00  ->  exp(0.00) = 1.00  ->  no effect
        coefficient -0.16  ->  exp(-0.16) = 0.85 ->  15% fewer claims

    Each is measured against the dropped baseline level of its own factor.
    """
    return (pd.DataFrame({"feature": columns,
                          "coefficient": model.coef_,
                          "relativity": np.exp(model.coef_)})
              .sort_values("relativity", ascending=False)
              .reset_index(drop=True))


# =====================================================================
# 5. VALIDATE
# =====================================================================

def check_balance(model, X, df, label):
    """Predicted total against actual total.

    A GLM with a log link should reproduce the observed total almost exactly on
    its own training data. A ratio away from 1.00 on train means something is
    wrong, usually that sample weights were not passed.
    """
    pred = model.predict(X)
    actual = df["ClaimNb"].sum() / df["Exposure"].sum()
    expected = np.average(pred, weights=df["Exposure"])
    dev = mean_poisson_deviance(df["Frequency"], pred,
                                sample_weight=df["Exposure"])
    print(f"  {label:6s}  observed {actual:.4f}   predicted {expected:.4f}   "
          f"balance {expected / actual:.4f}   deviance {dev:.4f}")


def actual_vs_expected(freq_model, X, df, column):
    """Observed and predicted frequency side by side across one factor.

    The most useful diagnostic in pricing. If the two track each other the
    model is fit for purpose; where they diverge tells you exactly where it is
    weak.
    """
    d = df.copy()
    d["pred"] = freq_model.predict(X)
    d["_lvl"] = (pd.cut(d[column], AGE_BANDS) if column == "DrivAge"
                 else d[column])
    g = d.groupby("_lvl", observed=True)
    out = pd.DataFrame({
        "exposure": g["Exposure"].sum(),
        "claims": g["ClaimNb"].sum(),
    })
    out["observed"] = out["claims"] / out["exposure"]
    out["predicted"] = g.apply(
        lambda t: np.average(t["pred"], weights=t["Exposure"])
    )
    return out


def lift_table(df, n=10):
    """Rank the book by predicted cost, then check actual cost follows.

    Split into ten equal groups by predicted pure premium. If the model
    genuinely separates risk, actual cost rises across the bands. This answers
    the underwriter's real question - can it tell a cheap risk from an
    expensive one - in a way no single accuracy score does.
    """
    d = df.copy()
    d["band"] = pd.qcut(d["pure_premium"], n, labels=False,
                        duplicates="drop") + 1
    g = d.groupby("band", observed=True)
    out = pd.DataFrame({
        "policies": g.size(),
        "exposure": g["Exposure"].sum(),
        "claims": g["ClaimNb"].sum(),
    })
    out["predicted"] = g.apply(
        lambda t: np.average(t["pure_premium"], weights=t["Exposure"])
    )
    out["actual"] = g["ClaimAmount"].sum() / out["exposure"]
    return out


# =====================================================================
# 6. PRICING
# =====================================================================

# Illustrative assumptions, not derived from the data. Stating which of your
# inputs are estimated and which are judgemental is the point of the exercise.
FIXED_EXPENSE = 45.0        # EUR per policy: admin, issuance
VARIABLE_EXPENSE = 0.15     # commission and tax, as a share of premium
PROFIT = 0.05               # target underwriting margin
RISK_MARGIN = 0.08          # loading for uncertainty in the estimate


def office_premium(pure_premium, fixed=FIXED_EXPENSE, var=VARIABLE_EXPENSE,
                   profit=PROFIT, risk=RISK_MARGIN):
    """Load the expected claims cost into a chargeable premium.

    Note the DIVISION. Commission, tax and profit are percentages of the
    premium itself - the very thing being solved for - so they cannot simply
    be added to the claims cost. The formula resolves that circularity:

        premium = (pure premium x (1 + risk) + fixed) / (1 - var - profit)
    """
    return (np.asarray(pure_premium) * (1 + risk) + fixed) / (1 - var - profit)


def premium_breakdown(pp):
    """Itemised build-up for a single risk."""
    prem = float(office_premium(pp))
    rows = [
        ("Expected claims cost (pure premium)", pp),
        ("Risk margin", pp * RISK_MARGIN),
        ("Fixed expenses", FIXED_EXPENSE),
        ("Variable expenses / commission", prem * VARIABLE_EXPENSE),
        ("Profit margin", prem * PROFIT),
        ("OFFICE PREMIUM", prem),
    ]
    return pd.DataFrame(rows, columns=["Component", "EUR"])


# =====================================================================
# 7. SENSITIVITY
# =====================================================================

def sensitivity(pure_premiums, exposure):
    """Re-price the book under stressed assumptions.

    Severity is stressed harder than frequency because it is the more volatile
    of the two: it absorbs claims inflation, as repair costs and injury
    settlements climb year on year, while accident rates move slowly. The
    relative sizes are a judgement, not a measurement.
    """
    base = np.average(office_premium(pure_premiums), weights=exposure)
    rows = []
    for name, f_shock, s_shock, var, prof in [
        ("Base case", 0.00, 0.00, VARIABLE_EXPENSE, PROFIT),
        ("Claims inflation +6%", 0.00, 0.06, VARIABLE_EXPENSE, PROFIT),
        ("Claims inflation +12%", 0.00, 0.12, VARIABLE_EXPENSE, PROFIT),
        ("Frequency +5%", 0.05, 0.00, VARIABLE_EXPENSE, PROFIT),
        ("Frequency -5%", -0.05, 0.00, VARIABLE_EXPENSE, PROFIT),
        ("Inflation +12% and frequency +5%", 0.05, 0.12, VARIABLE_EXPENSE, PROFIT),
        ("Expense ratio 15% -> 20%", 0.00, 0.00, 0.20, PROFIT),
        ("Profit target 5% -> 10%", 0.00, 0.00, VARIABLE_EXPENSE, 0.10),
    ]:
        shocked = pure_premiums * (1 + f_shock) * (1 + s_shock)
        prem = np.average(office_premium(shocked, var=var, profit=prof),
                          weights=exposure)
        rows.append({"Scenario": name, "Avg premium": prem,
                     "Change %": (prem / base - 1) * 100})
    return pd.DataFrame(rows).set_index("Scenario")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("Loading data...")
    df = load_data()

    explore(df)

    train, test = train_test_split(df, test_size=0.25, random_state=42)
    print(f"\nTrain {len(train):,}   Test {len(test):,}")

    X_train = build_features(train)
    X_test = align(X_train, build_features(test))
    print(f"Features: {X_train.shape[1]} columns after banding and encoding")

    print("\n" + "=" * 60)
    print("FREQUENCY MODEL (Poisson GLM, exposure-weighted)")
    print("=" * 60)
    freq_model = fit_frequency(X_train, train)
    check_balance(freq_model, X_train, train, "train")
    check_balance(freq_model, X_test, test, "test")

    rf = relativities(freq_model, X_train.columns)
    print("\nHighest relativities:")
    print(rf.head(12).to_string(index=False))
    print("\nLowest relativities:")
    print(rf.tail(8).to_string(index=False))

    print("\nActual vs expected by driver age (test set):")
    print(actual_vs_expected(freq_model, X_test, test, "DrivAge").round(4).to_string())

    print("\n" + "=" * 60)
    print("SEVERITY MODEL (Gamma GLM, claim-count-weighted)")
    print("=" * 60)
    sev_model = fit_severity(X_train, train)

    mask = (test["ClaimNb"] > 0) & (test["AvgClaimAmount"] > 0)
    sev_pred = sev_model.predict(X_test[mask.values])
    print(f"  observed  EUR {np.average(test.loc[mask, 'AvgClaimAmount'], weights=test.loc[mask, 'ClaimNb']):,.0f}")
    print(f"  predicted EUR {np.average(sev_pred, weights=test.loc[mask, 'ClaimNb']):,.0f}")

    rs = relativities(sev_model, X_train.columns)
    f_spread = rf["relativity"].max() / rf["relativity"].min()
    s_spread = rs["relativity"].max() / rs["relativity"].min()
    print(f"\n  frequency relativities span {rf['relativity'].min():.2f} to "
          f"{rf['relativity'].max():.2f}  ({f_spread:.1f}x)")
    print(f"  severity  relativities span {rs['relativity'].min():.2f} to "
          f"{rs['relativity'].max():.2f}  ({s_spread:.1f}x)")
    if s_spread < f_spread:
        print("  Severity is the flatter of the two: rating factors predict how")
        print("  OFTEN a driver claims better than how EXPENSIVE the claim is,")
        print("  since cost depends more on circumstance than on the driver.")
    else:
        print("  Severity is not flatter here. On real motor data it usually is,")
        print("  so worth checking whether a few large claims are driving this.")

    print("\nLargest severity relativities:")
    print(rs.head(8).to_string(index=False))

    print("\n" + "=" * 60)
    print("PRICING")
    print("=" * 60)
    priced = test.copy()
    priced["pred_frequency"] = freq_model.predict(X_test)
    priced["pred_severity"] = sev_model.predict(X_test)
    priced["pure_premium"] = priced["pred_frequency"] * priced["pred_severity"]
    priced["premium"] = office_premium(priced["pure_premium"])

    w = priced["Exposure"]
    print(f"Average pure premium : EUR {np.average(priced['pure_premium'], weights=w):,.2f}")
    print(f"Average premium      : EUR {np.average(priced['premium'], weights=w):,.2f}")
    print(f"Cheapest / dearest   : EUR {priced['premium'].min():,.0f} / "
          f"EUR {priced['premium'].max():,.0f}")

    print("\nPremium build-up for a risk with a EUR 200 pure premium:")
    print(premium_breakdown(200.0).round(2).to_string(index=False))

    print("\n" + "=" * 60)
    print("VALIDATION - does the model separate good risks from bad?")
    print("=" * 60)
    lift = lift_table(priced)
    print(lift.round(2).to_string())
    print("\nMean exposure by band (is band 1 full of very short policies?):")
    print(priced.assign(
        band=pd.qcut(priced["pure_premium"], 10, labels=False,
                     duplicates="drop") + 1
    ).groupby("band")["Exposure"].mean().round(3).to_string())
    print("\nBand 1 claim costs (are a few large claims driving it?):")
    b = priced.assign(
        band=pd.qcut(priced["pure_premium"], 10, labels=False,
                     duplicates="drop") + 1
    )
    b1 = b[(b["band"] == 1) & (b["ClaimAmount"] > 0)]
    print(f"  claims: {len(b1)}   total: EUR {b1['ClaimAmount'].sum():,.0f}")
    print(f"  largest 5: {b1['ClaimAmount'].nlargest(5).round(0).tolist()}")
    print(f"  top 5 as share of band total: "
          f"{b1['ClaimAmount'].nlargest(5).sum() / b1['ClaimAmount'].sum():.1%}")
    ratio = lift["actual"].iloc[-1] / lift["actual"].iloc[0]
    print(f"\nActual cost, dearest decile vs cheapest: {ratio:.1f}x")
    print("Rising actual cost down the table means the model genuinely ranks risk.")

    print("\n" + "=" * 60)
    print("SENSITIVITY TO ASSUMPTIONS")
    print("=" * 60)
    print(sensitivity(priced["pure_premium"].to_numpy(), w.to_numpy()).round(2).to_string())

    rf.to_csv("outputs/relativities_frequency.csv", index=False)
    rs.to_csv("outputs/relativities_severity.csv", index=False)
    lift.to_csv("outputs/lift_table.csv")
    print("\n[tables written to outputs/]")


if __name__ == "__main__":
    main()
