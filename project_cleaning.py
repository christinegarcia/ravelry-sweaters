import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

#########

output_after_cleaning_filename = "data/pattern_author_cleaned_data.csv"

# Turn lists saved as strings (from API) into Python lists when possible
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []

# Data cleaning
def preprocess_data(raw_data):
    data = raw_data.copy()

    print("\nStarting data preprocessing...")

    # Drop 6 rows with extreme yardage outliers; distorts all engineered yardage features
    # data.drop([43133, 85076], inplace=True)
    # data.drop()
    ids_todrop = [370725, 962475, 1114028, 820543, 892165, 1154060]
    data = data[~data["id"].isin(ids_todrop)]

    # Popularity rank (continuous) -> popularity class
    # bins = [-1, 499, 999, 1999, 2999, np.inf]
    # labels = [0, 1, 2, 3, 4]
    bins = [-1, 9999, np.inf]
    labels = [1, 0]
    data["popularity_class"] = pd.cut(data["popularity_rank"], bins=bins, labels=labels, include_lowest=True)
    data["popularity_class"] = data["popularity_class"].astype(int)
    print(" - Converted popularity rank to popularity class")

    # Impute two null Free values manually
    data.loc[data["permalink"] == "union-square-market-pullover", "free"] = False
    data.loc[data["permalink"] == "stripy-sweater-4", "free"] = False

    # # Handle missing prices
    # # Missing price - paid patterns
    # data["price_was_missing"] = 0
    # data.loc[(data["price"].isnull()) & (data["free"] == False), "price_was_missing"] = 1
    # num_missing_price = data["price_was_missing"].sum()
    # median_price = data.loc[data["free"] == False, "price"].median()
    # data['price'] = data["price"].fillna(median_price)
    # data.drop(columns=["price_was_missing"], inplace=True)
    # print(f"- Imputed {num_missing_price} missing prices with the median paid price: ${median_price:.2f}")
    #
    # # Missing price - free patterns
    # data.loc[(data["price"].isnull()) & (data["free"] == True), "price"] = 0.0
    # print("- Imputed 'price' for free patterns.")
    #
    # # Missing currency - free patterns
    # data.loc[data["currency"].isnull(), "currency"] = 'FREE'
    # print("- Imputed 'currency' for patterns with missing values.")

    #  Handle missing publish dates - first impute generally_available, then updated_at, then median for dataset
    data["published_dt"] = pd.to_datetime(data["published"], errors="coerce", utc=True)
    generally_available_dt = pd.to_datetime(data["generally_available"], errors="coerce", utc=True)
    updated_at_dt = pd.to_datetime(data["updated_at"], errors="coerce", utc=True)

    data["published_dt"] = data["published_dt"].fillna(generally_available_dt)
    data["published_dt"] = data["published_dt"].fillna(updated_at_dt)
    data["published_dt"] = data["published_dt"].fillna(data["published_dt"].median())
    print("- Filled missing 'published' values")

    # Convert publish date -> publish year
    data["published_year"] = data["published_dt"].dt.year.astype(int)

    # New features: author data
    first_published_s = data.groupby("pattern_author_name")["published_dt"].transform("min")
    data["author_tenure_days"] = (pd.to_datetime("now", utc=True) - first_published_s).dt.days
    print("- Added column for author tenure")

    # Impute NaNs created during author feature engineering
    data['author_patterns_count'].fillna(5, inplace=True)
    # data['author_tenure_days'].fillna(1, inplace=True)
    print("- Imputed missing values in 'author_patterns_count' feature")

    # Manually impute real values for 5 author_tenure_days=0 from Ravelry
    # All patterns are from an author who removed their patterns, so I imputed from the earliest date listed on Ravelry for these patterns: https://www.ravelry.com/designers/none
    author_tenure_zeros = ["herren-raglanpullover", "variations-on-rvo", "miss-elinor", "miss-nina", "pulloverchen-mit-herz"]
    affected_tenure = (pd.to_datetime("now", utc=True) - pd.to_datetime('2016-10-23', utc=True)).days
    data.loc[data["permalink"].isin(author_tenure_zeros), "author_tenure_days"] = affected_tenure

    # Convert string/list features into interpretable versions
    # List of languages -> number of languages supported
    data["languages_num"] = data["languages"].apply(safe_literal_eval).apply(len)
    print("\n- Created 'languages_num' column")

    # Yarn weight as name -> as number
    yarn_weight_map = {
        'Thread': 0, 'Cobweb': 0, 'Lace': 0, 'Light Fingering': 0.5,
        'Fingering': 1, 'Sport': 2, 'DK': 3, 'Worsted': 4, 'Aran': 4.5,
        'Bulky': 5, 'Super Bulky': 6, 'Jumbo': 7, 'Any gauge': 8
    }
    data['yarn_weight_num'] = data['yarn_weight_name'].map(yarn_weight_map)
    print("- Created 'yarn_weight_num' column from 'yarn_weight_name'")

    # List of needle sizes -> number of needle sizes
    data['pattern_needle_num'] = data['pattern_needle_sizes'].apply(safe_literal_eval).apply(len)
    print("- Created 'pattern_needle_num' column")

    # Date of last update -> days since last update
    updated_at_dt = pd.to_datetime(data['updated_at'], errors='coerce', utc=True)
    now = pd.Timestamp.now(tz="utc")
    data["days_since_update"] = (now - updated_at_dt).dt.days
    print("- Converted updated_at date into days_since_update")

    # Thorough imputation for yardage columns
    data.rename(columns={'yardage': 'yardage_min'}, inplace=True)
    print("- Renamed 'yardage' to 'yardage_min'")

    mask_max_missing = data['yardage_min'].notnull() & data['yardage_max'].isnull()
    data.loc[mask_max_missing, 'yardage_max'] = data.loc[mask_max_missing, 'yardage_min']
    print(f"- Imputed 'yardage_max' with 'yardage_min' for {mask_max_missing.sum()} rows")

    yardage_min_avg = data["yardage_min"].mean()
    yardage_max_avg = data["yardage_max"].mean()
    data["yardage_min"] = data["yardage_min"].fillna(yardage_min_avg)
    data["yardage_max"] = data["yardage_max"].fillna(yardage_max_avg)
    print("- Imputed remaining missing 'yardage_min', 'yardage_max' based on averages")

    # Impute values for anything still missing in yarn_weight_num
    if data['yarn_weight_num'].isnull().any():
        data['yarn_weight_num'] = data['yarn_weight_num'].fillna(8.0)
        print(f"- Imputed remaining missing 'yarn_weight_num' with 8.0 (Any Gauge)")

    # Create yardage_diff column to approximate size inclusivity
    data['yardage_diff'] = data['yardage_max'] - data['yardage_min']
    data.loc[data['yardage_diff'] < 0, 'yardage_diff'] = 0
    print("- Calculated 'yardage_diff' column from imputed min/max values")

    # Calculate yardage_ratio (comparing specified yardage to avg sweater yardage)
    # If yardage_min=0, ratio=NaN because of division by zero -> impute with 1
    data["yardage_ratio"] = data["yardage_min"] / yardage_min_avg
    data["yardage_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    data["yardage_ratio"].fillna(1, inplace=True)
    print("- Created 'yardage_ratio' feature")

    # Drop rows still missing yardage info
    initial_rows = len(data)
    data.dropna(subset=['yardage_min', 'yardage_diff'], inplace=True)
    rows_dropped = initial_rows - len(data)
    if rows_dropped > 0:
        print(f"- Dropped {rows_dropped} rows with remaining missing yardage information")

    # One-hot encoding for pattern_attributes
    mlb_attr = MultiLabelBinarizer()
    attributes_series = data['pattern_attributes'].apply(safe_literal_eval)
    one_hot_attributes = pd.DataFrame(mlb_attr.fit_transform(attributes_series),
                                      columns="attr_" + mlb_attr.classes_,
                                      index=data.index)
    data = pd.concat([data, one_hot_attributes], axis=1)
    print(f"- One-hot encoded 'pattern_attributes' into {len(one_hot_attributes.columns)} new columns")

    # Feature engineering to reduce sparsity in one-hot encoded attr_ features
    # Using new features rather than dim. reduction (e.g. PCA) to retain interpretability

    # Create features summing difficult techniques and also interaction of hard techniques + small yarn weight
    hard_techniques = ["colorwork", "fairisle", "icelandic", "finnish", "norwegian", "other-colorwork", "stranded", "intarsia", "lace", "lace-edging", "estonian", "cables", "aran", "bavarian", "guernsey", "irish", "beads", "entrelac", "textured", "embroidery", "short-rows", "steeks", "twisted-stitches", "brioche-tuck", "bobble-or-popcorn"]

    technique_cols_to_sum = [f"attr_{t}" for t in hard_techniques if f"attr_{t}" in data.columns]
    data["technique_count"] = data[technique_cols_to_sum].sum(axis=1)
    data["technique_yarn_interaction"] = data["technique_count"] * data["yarn_weight_num"]

    # Create the sum of difficult techniques feature
    print(f"- Created 'technique_count' feature")
    print(f"- Created 'technique_yarn_interaction' feature")

    # Create helpful attributes feature
    helpful_attributes = ["chart", "phototutorial", "schematic", "video-tutorial", "written-pattern", "captioned-video", "color-blind-accessible", "pattern-recipe"]

    helpful_cols = [f"attr_{ha}" for ha in helpful_attributes if f"attr_{ha}" in data.columns]
    data["helpful_attributes_count"] = data[helpful_cols].sum(axis=1)
    print("- Created 'helpful_attributes_count' feature")

    # Create construction technique feature
    construction_logic = {
        "construction_seamless": ["seamless", "in-the-round", "one-piece"],
        "construction_seamed": ["seamed", "worked-flat", "selvedge"],
        "construction_top_down": ["top-down", "top-cuff-down"],
        "construction_bottom_up": ["bottom-up"]
    }

    for new_col, keywords in construction_logic.items():
        construction_cols = [f"attr_{k}" for k in keywords if f"attr_{k}" in data.columns]
        data[new_col] = data[construction_cols].any(axis=1).astype(int)
    print("- Created 'construction_seamless', 'construction_seamed', 'construction_bottom_up', 'construction_top_down' features")

    # Create shoulder construction feature
    shoulder_styles = {
        "shoulder_raglan": ["raglan-sleeve"],
        "shoulder_yoke": ["circular-yoke"],
        "shoulder_drop": ["drop-sleeve", "modified-drop-sleeve"],
        "shoulder_set_in": ["set-in-sleeve"]
    }

    for new_col, keywords in shoulder_styles.items():
        shoulder_cols = [f"attr_{k}" for k in keywords if f"attr_{k}" in data.columns]
        data[new_col] = data[shoulder_cols].any(axis=1).astype(int)
    print("- Created 'shoulder_raglan', 'shoulder_yoke', 'shoulder_drop', 'shoulder_set_in' features")

    # Create colorwork indicator feature
    colorwork_attributes = ["colorwork", "fairisle", "stranded", "mosaic", "stripes", "intarsia", "duplicate-stitch", "steeks", "tapestry-crochet", "icelandic", "finnish", "norwegian", "other-colorwork"]
    color_cols = [f"attr_{a}" for a in colorwork_attributes if f"attr_{a}" in data.columns]
    data["has_colorwork"] = data[color_cols].any(axis=1).astype(int)
    print("- Created 'has_colorwork' feature")

    # Create fit features
    fit_attributes = {
        "fit_oversized": ["oversized", "positive-ease", "swing"],
        "fit_fitted": ["fitted", "negative-ease", "no-ease", "waist"],
        "fit_diverse": ["plus", "petite", "tall", "maternity", "unisex"]
    }

    for new_col, keywords in fit_attributes.items():
        fit_cols = [f"attr_{k}" for k in keywords if f"attr_{k}" in data.columns]
        data[new_col] = data[fit_cols].any(axis=1).astype(int)
    print("- Created 'fit_oversized', 'fit_fitted', 'fit_diverse' features")

    # Create regional identifier feature
    regional_keywords = ["icelandic", "finnish", "norwegian", "estonian", "danish", "sami", "latvian", "swedish", "turkish", "bavarian", "shetland", "faroese", "irish", "guernsey", "aran", "cowichan", "andean", "dutch-heel", "orenburg"]

    regional_cols = [f"attr_{k}" for k in regional_keywords if f"attr_{k}" in data.columns]
    data["is_regional"] = data[regional_cols].any(axis=1).astype(int)
    print(f"- Created 'is_regional' feature")

    # Create pattern style feature
    style_attributes = {
        "style_vintage": ["vintage", "traditional", "victorian", "classic", "textured", "cables"],
        "style_modern": ["modern", "minimalist", "seamless", "top-down", "oversized", "positive-ease", "cropped"],
        "style_unique": ["asymmetric", "modular", "freeform", "mosaic", "entrelac"]
    }

    for new_col, keywords in style_attributes.items():
        style_cols = [f"attr_{k}" for k in keywords if f"attr_{k}" in data.columns]
        data[new_col] = data[style_cols].any(axis=1).astype(int)
    print(f"- Created 'style_vintage', 'style_modern', and 'style_unique' features")

    # Create has_short_rows feature
    data.rename(columns={'attr_short': 'has_short_rows'}, inplace=True)
    print("- Renamed 'attr_short' to 'has_short_rows'")

    # Drop all unneeded columns
    data.drop(columns=[col for col in data.columns if col.startswith("attr_")], inplace=True)
    data.drop(columns=["published_dt", "generally_available", "updated_at", "published", "languages", "yarn_weight_name", "pattern_needle_sizes", "pattern_attributes"], inplace=True)

    # Save data after preprocessing
    data.to_csv(output_after_cleaning_filename, index=False)
    print("\nPreprocessing complete")
