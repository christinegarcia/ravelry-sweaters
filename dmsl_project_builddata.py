from project_querying import *
from project_cleaning import *

pd.set_option('display.width', None)

############### Querying the dataset

output_filename = "data/pattern_data.csv"
output_with_author_filename = "data/pattern_author_data.csv"
# output_with_author_trimmed_filename = "data/pattern_author_trim_data.csv"
author_stats_filename = "data/author_stats.csv"
output_after_cleaning_filename = "data/pattern_author_cleaned_data.csv"
output_for_modeling_filename = "data/final_data.csv"

### Build dataset with Ravelry API
# Ran code piecewise to deal with potential API issues (hence frequent saves/loads)

# Get 100,000 most popular sweater pattern IDs
get_popular_pattern_ids()

# Fix errors during pattern ID pull
with open("pattern_ids.pkl", "rb") as file:
    pattern_ids = pickle.load(file)
reget_popular_patterns([417], pattern_ids)

# Get details for top 100,000 patterns
# Built off of pattern_data.csv from HW 5: data for first 10,000 patterns
with open("pickle/pattern_ids_v2.pkl", "rb") as file:
    pattern_ids = pickle.load(file)
build_full_df(pattern_ids)

# Check for all pattern ranks 0-99,999 in there
raw_data = pd.read_csv(output_filename)
print(f"\nAll pattern IDs in raw data are unique: {raw_data["id"].is_unique}")
all_pop_ranks = range(0, 100000)
print(f"Missing/extra pattern pop. ranks: {list(set(all_pop_ranks) - set(raw_data["popularity_rank"]))}")

# Pattern ID 7415398 gave 404 error, so pattern with rank 16945 is missing from data
# Shift all following popularity ranks up to fill that gap
raw_data.loc[raw_data["popularity_rank"] > 16944, "popularity_rank"] -= 1
all_pop_ranks = range(0, 99999)
print(f"Recheck - Missing/extra pattern pop. ranks: {list(set(all_pop_ranks) - set(raw_data["popularity_rank"]))}")
raw_data.to_csv(output_filename, index=False)

# Add author stats
# Built off author_stats.csv from HW 5: author data from first 10,000 patterns
add_author_data_to_df(raw_data)

############### Cleaning the dataset

# Data exploration (pre-cleaning)
print("Pre-cleaning: ")
data = pd.read_csv(output_with_author_filename)
print(data.info())
print(data.describe())
print(data.isna().sum())

# Data cleaning
preprocess_data(data)

# Specific data exploration (outliers, etc)
data_clean = pd.read_csv(output_after_cleaning_filename)
print(data_clean.info())
print(data_clean.describe())
print(data_clean.isna().sum())
# print(data_clean[data_clean["author_tenure_days"]==0])
# print(data_clean[data_clean["pattern_author_name"]=="Inge's Knitting Lab"])
# print(data_clean[data_clean["yardage_min"]>50000])
# print(data_clean[data_clean["yardage_ratio"]>25])

# Drop columns that are informative but not needed for modeling now
data_clean = pd.read_csv(output_after_cleaning_filename)
data_clean.drop(columns=["id", "name", "permalink", "price", "popularity_rank", "currency", "pattern_author_name"], inplace=True)
data_clean.to_csv(output_for_modeling_filename, index=False)

# Data exploration (post-cleaning)
final_data = pd.read_csv(output_for_modeling_filename)
print("Post-cleaning:")
print(final_data.info())
print(final_data.describe())
print(final_data.isna().sum())

