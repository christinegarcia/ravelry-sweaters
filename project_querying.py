import pandas as pd
import numpy as np
import time
import os
import pickle
import requests
from requests.auth import HTTPBasicAuth
import time

#########   
RAVELRY_USERNAME = "santized - add here"
RAVELRY_PASSWORD = "santized - add here"
#########

### BUILD DATABASE FROM RAVELRY API

# API endpoints & parameters
base_url = "https://api.ravelry.com"
search_endpoint = "/patterns/search.json"
pattern_endpoint = "/patterns/{id}.json"

num_pages = 1000

output_filename = "data/pattern_data.csv"
output_with_author_filename = "data/pattern_author_data.csv"
author_stats_filename = "data/author_stats.csv"

# Fetches pattern IDs by popularity; returns list of pattern IDs (ints)
def get_popular_pattern_ids():
    pattern_ids = []
    print(f"Fetching pattern IDs for {num_pages} pages...")

    for page in range(1, num_pages + 1):
        params = {
            "sort": "popularity",
            "craft": "knitting",
            "pc": "pullover",
            "page": page
        }

        try:
            print(f"Fetching page {page}/{num_pages}...")
            response = requests.get(
                base_url + search_endpoint,
                params=params,
                auth=HTTPBasicAuth(RAVELRY_USERNAME, RAVELRY_PASSWORD)
            )
            response.raise_for_status()
            data = response.json()

            if not data['patterns']:
                print("No more patterns found. Stopping.")
                break

            for pattern in data['patterns']:
                pattern_ids.append(pattern['id'])

            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error on page {page}: {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"An error occurred on page {page}: {e}")
            continue

    print(f"Finished fetching. Found {len(pattern_ids)} unique pattern IDs.")
    with open("pickle/pattern_ids.pkl", "wb") as file:
        pickle.dump(pattern_ids, file)

# For broken pattern pulls: fetches specific page and appends pattern IDs at correct place in pattern_ids list
def reget_popular_patterns(missing_pattern_pages, pattern_list):
    # page 1: indices 0-99
    # page 2: indices 100-199
    # ...
    # page i - {i-1}*100 - i*100-1

    for page in missing_pattern_pages:
        params = {
            "sort": "popularity",
            "craft": "knitting",
            "pc": "pullover",
            "page": page
        }

        try:
            print(f"Fetching page {page}/{num_pages}...")
            response = requests.get(
                base_url + search_endpoint,
                params=params,
                auth=HTTPBasicAuth(RAVELRY_USERNAME, RAVELRY_PASSWORD)
            )
            response.raise_for_status()
            data = response.json()

            idx = (page-1)*100
            pattern_list[idx:idx] = [pattern["id"] for pattern in data["patterns"]]

            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error on page {page}: {e}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"An error occurred on page {page}: {e}")
            continue

    print(f"Finished fetching.")
    with open("pickle/pattern_ids_v2.pkl", "wb") as file:
        pickle.dump(pattern_list, file)

# Fetches detailed pattern data for a given ID; returns dictionary with pattern details (or None if error)
def get_pattern_details(pattern_id):
    url = base_url + pattern_endpoint.format(id=pattern_id)

    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(RAVELRY_USERNAME, RAVELRY_PASSWORD)
        )
        response.raise_for_status()

        pattern_data = response.json()['pattern']
        yarn_weight_data = pattern_data.get('yarn_weight', {}) or {}

        details = {
            'id': pattern_data.get('id'),
            'name': pattern_data.get('name'),
            'permalink': pattern_data.get('permalink'),
            # 'created_at': pattern_data.get('created_at'),
            'currency': pattern_data.get('currency'),
            'difficulty_average': pattern_data.get('difficulty_average'),
            # 'difficulty_count': pattern_data.get('difficulty_count'),
            # 'favorites_count': pattern_data.get('favorites_count'),
            'free': pattern_data.get('free'),
            # 'gauge': pattern_data.get('gauge'),
            # 'gauge_description': pattern_data.get('gauge_description'),
            # 'gauge_divisor': pattern_data.get('gauge_divisor'),
            # 'gauge_pattern': pattern_data.get('gauge_pattern'),
            'generally_available': pattern_data.get('generally_available'),
            # 'has_uk_terminology': pattern_data.get('has_uk_terminology'),
            # 'has_us_terminology': pattern_data.get('has_us_terminology'),
            'languages': pattern_data.get('languages'),
            'pattern_author_name': pattern_data.get('pattern_author', {}).get('name'),
            'pattern_attributes': [attr['permalink'] for attr in pattern_data.get('pattern_attributes', [])],
            # 'pattern_categories': [cat['permalink'] for cat in pattern_data.get('pattern_categories', [])],
            'pattern_needle_sizes': [size['us'] for size in pattern_data.get('pattern_needle_sizes', []) if
                                     'us' in size],
            'price': pattern_data.get('price'),
            # 'projects_count': pattern_data.get('projects_count'),
            'published': pattern_data.get('published'),
            # 'queued_projects_count': pattern_data.get('queued_projects_count'),
            # 'rating_average': pattern_data.get('rating_average'),
            # 'rating_count': pattern_data.get('rating_count'),
            # 'row_gauge': pattern_data.get('row_gauge'),
            # 'sizes_available': pattern_data.get('sizes_available'),
            'updated_at': pattern_data.get('updated_at'),
            'yardage': pattern_data.get('yardage'),
            'yardage_max': pattern_data.get('yardage_max'),
            'yarn_weight_name': yarn_weight_data.get('name'),
            # 'yarn_weight_ply': yarn_weight_data.get('ply'),
            # 'yarn_weight_wpi': yarn_weight_data.get('wpi')
        }
        return details

    except requests.exceptions.HTTPError as e:
        print(f"  - HTTP error fetching pattern {pattern_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  - An error occurred fetching pattern {pattern_id}: {e}")
        return None

# Fetch detailed pattern data for each pattern ID
# Supports regular saves and prevents repeat queries
def build_full_df(pattern_ids):
    # Map popularity rank to pattern ID
    rank_lookup = {p_id: rank for rank, p_id in enumerate(pattern_ids)}

    # Check for patterns already processed
    processed_ids = set()
    if os.path.exists(output_filename):
        try:
            existing_df = pd.read_csv(output_filename, usecols=["id"])
            processed_ids = set(existing_df["id"].unique())
            print(f"Found existing file. {len(processed_ids)} patterns already completed.")
        except Exception as e:
            print(f"Could not read existing file, starting fresh: {e}")

    # Filter for unprocessed pattern IDs
    remaining_ids = [p_id for p_id in pattern_ids if p_id not in processed_ids]
    total_remaining = len(remaining_ids)

    if total_remaining == 0:
        print("All patterns already processed")
        return
    print(f"\nStarting to fetch details for {total_remaining} patterns...")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    for i, pattern_id in enumerate(remaining_ids):
        print(f"Processing pattern {i + 1}/{total_remaining} (ID: {pattern_id})...")
        details = get_pattern_details(pattern_id)
        if details:
            details["popularity_rank"] = rank_lookup[pattern_id]
            df_row = pd.DataFrame([details])
            df_row.to_csv(output_filename, mode="a", index=False, header=not os.path.isfile(output_filename))
        time.sleep(0.1)

    print(f"\nRUN COMPLETE.\n Data has been saved to '{output_filename}'.")

# Fetch designer stats for all unique authors in pattern IDs; returns df with stats by author
def get_author_stats(data):
    print("Querying Ravelry API for author stats...")

    first_pattern_by_author = data.drop_duplicates(subset=['pattern_author_name'])

    author_data_list = []
    for _, row in first_pattern_by_author.iterrows():
        author_name = row['pattern_author_name']
        pattern_id = row['id']

        if pd.isna(author_name):
            continue

        url = base_url + pattern_endpoint.format(id=pattern_id)
        try:
            response = requests.get(
                url,
                auth=requests.auth.HTTPBasicAuth(RAVELRY_USERNAME, RAVELRY_PASSWORD)
            )
            response.raise_for_status()

            pattern_data = response.json().get('pattern', {})
            author_obj = pattern_data.get('pattern_author')

            if author_obj:
                author_data_list.append({
                    'pattern_author_name': author_name,
                    # 'author_favorites_count': author_obj.get('favorites_count'),
                    'author_patterns_count': author_obj.get('patterns_count')
                })
            else:
                author_data_list.append({
                    'pattern_author_name': author_name,
                    # 'author_favorites_count': None,
                    'author_patterns_count': None
                })

            print(f"  - Fetched data for {author_name}")
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            print(f"  - Could not fetch data for author {author_name} using pattern {pattern_id}: {e}")
            author_data_list.append({
                'pattern_author_name': author_name,
                # 'author_favorites_count': None,
                'author_patterns_count': None
            })
            continue

    author_stats_df = pd.DataFrame(author_data_list)
    author_stats_df.to_csv(author_stats_filename, index=False)
    print(f"Author stats saved to {author_stats_filename}")
    return author_stats_df

def add_author_data_to_df(data):
    author_api_stats = get_author_stats(data)
    data = pd.merge(data, author_api_stats, on='pattern_author_name', how='left')
    data.to_csv(output_with_author_filename, index=False)
    print(f"Added author stats to main data file and saved as {output_with_author_filename}")
