import requests
import sqlite3
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import time
import re
import difflib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
load_dotenv()


def get_connection():
    return sqlite3.connect("V3.db")


def init_db():
    conn = get_connection()
    curr = conn.cursor()

    curr.executescript("""
        CREATE TABLE IF NOT EXISTS USERS (
            USER_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            EMAIL VARCHAR(128) UNIQUE,
            LOCATION VARCHAR(128)
        );

        CREATE TABLE IF NOT EXISTS USER_ENTRIES (
            NO INTEGER PRIMARY KEY AUTOINCREMENT,
            USER_ID INTEGER,
            LOCATION VARCHAR(128),
            WEATHER REAL,
            MOOD VARCHAR(128),
            EVENT VARCHAR(128),
            DRESS_TOP VARCHAR(128),
            TYPE VARCHAR(128),
            FABRIC VARCHAR(128),
            COLOUR VARCHAR(128),
            JEANS_SKIRT VARCHAR(128),
            LENGTH VARCHAR(128),
            TYPE_BOTTOM VARCHAR(128),
            FABRIC_BOTTOM VARCHAR(128),
            COLOUR_BOTTOM VARCHAR(128),
            EMAIL VARCHAR(128),
            MATCH_PERCENTAGE REAL,
            CATEGORY VARCHAR(128),
            USER_RATING REAL,

            FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID)
        );
    """)

    conn.commit()
    conn.close()


def migrate_db():
    """Safely add USER_RATING column to existing databases.
    If the column already exists this is a no-op — existing rows are untouched."""
    conn = get_connection()
    curr = conn.cursor()
    try:
        curr.execute("ALTER TABLE USER_ENTRIES ADD COLUMN USER_RATING REAL")
        conn.commit()
    except Exception:
        pass  # Column already exists — safe to ignore
    conn.close()


def get_or_create_user(email, location):
    conn = get_connection()
    curr = conn.cursor()

    curr.execute("SELECT USER_ID FROM USERS WHERE EMAIL = ?", (email,))
    row = curr.fetchone()

    if row:
        user_id = row[0]
        curr.execute(
            "UPDATE USERS SET LOCATION = ? WHERE USER_ID = ?",
            (location, user_id)
        )
    else:
        curr.execute(
            "INSERT INTO USERS (EMAIL, LOCATION) VALUES (?, ?)",
            (email, location)
        )
        user_id = curr.lastrowid

    conn.commit()
    conn.close()
    return user_id


def create_user_entry(user_id):
    conn = get_connection()
    curr = conn.cursor()

    curr.execute("SELECT EMAIL, LOCATION FROM USERS WHERE USER_ID = ?", (user_id,))
    user_row = curr.fetchone()

    if not user_row:
        conn.close()
        raise ValueError("User not found.")

    email, location = user_row

    curr.execute(
        "INSERT INTO USER_ENTRIES (USER_ID, EMAIL, LOCATION) VALUES (?, ?, ?)",
        (user_id, email, location)
    )
    entry_id = curr.lastrowid

    conn.commit()
    conn.close()
    return entry_id


def update_entry_field(entry_id, field, value):
    allowed_fields = {
    "LOCATION", "WEATHER", "MOOD", "EVENT", "DRESS_TOP", "TYPE",
    "FABRIC", "COLOUR", "JEANS_SKIRT", "LENGTH", "TYPE_BOTTOM",
    "FABRIC_BOTTOM", "COLOUR_BOTTOM", "EMAIL", "MATCH_PERCENTAGE",
    "CATEGORY", "TREND_SCORE", "ML_SCORE", "USER_RATING"
    }

    if field not in allowed_fields:
        raise ValueError(f"Invalid field name: {field}")

    conn = get_connection()
    curr = conn.cursor()
    curr.execute(f"UPDATE USER_ENTRIES SET {field} = ? WHERE NO = ?", (value, entry_id))
    conn.commit()
    conn.close()


def get_user_row(entry_id):
    conn = get_connection()
    curr = conn.cursor()
    curr.execute("SELECT * FROM USER_ENTRIES WHERE NO = ?", (entry_id,))
    row = curr.fetchone()
    conn.close()
    return row


def api_temp(user_id):
    conn = get_connection()
    curr = conn.cursor()
    curr.execute("SELECT LOCATION FROM USERS WHERE USER_ID = ?", (user_id,))
    row = curr.fetchone()
    conn.close()

    if not row:
        raise ValueError("User location not found.")

    loc = row[0]

    API_key = os.getenv("OPENWEATHERAPI")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={loc}&appid={API_key}&units=metric"

    res = requests.get(url)
    res.raise_for_status()
    data = res.json()

    return data["main"]["temp"]


def geocode(user_id):
    conn = get_connection()
    curr = conn.cursor()
    curr.execute("SELECT LOCATION FROM USERS WHERE USER_ID = ?", (user_id,))
    row = curr.fetchone()
    conn.close()

    if not row:
        raise ValueError("User location not found.")

    loc = row[0]

    API_key = os.getenv("OPENCAGEAPI")
    url = f"https://api.opencagedata.com/geocode/v1/json?q={loc}&key={API_key}"

    res = requests.get(url)
    res.raise_for_status()
    data = res.json()

    return data["results"][0]["components"]["country_code"].upper()


def get_user_data(entry_id):
    conn = get_connection()
    curr = conn.cursor()
    curr.execute("""
        SELECT MOOD, EVENT, WEATHER, DRESS_TOP, TYPE, FABRIC, COLOUR,
               JEANS_SKIRT, LENGTH, TYPE_BOTTOM, FABRIC_BOTTOM, COLOUR_BOTTOM
        FROM USER_ENTRIES
        WHERE NO = ?
    """, (entry_id,))
    row = curr.fetchone()
    conn.close()
    return row


def get_user_data_ml(entry_id):
    conn = get_connection()
    curr = conn.cursor()
    curr.execute("""
        SELECT WEATHER, MOOD, EVENT, DRESS_TOP, TYPE, FABRIC, COLOUR,
               JEANS_SKIRT, LENGTH, TYPE_BOTTOM, COLOUR_BOTTOM, FABRIC_BOTTOM
        FROM USER_ENTRIES
        WHERE NO = ?
    """, (entry_id,))
    row = curr.fetchone()
    conn.close()
    return row


def generate_keywords(mood, event, temp, outfit_keywords=None):
    keywords = [
        f"{mood} {event} outfit",
        f"{event} fashion",
        f"{event} wear",
        f"{mood} style"
    ]

    if temp >= 24:
        keywords.append("summer fashion")
    elif 15 <= temp < 24:
        keywords.append("autumn fashion")
    elif 11 <= temp < 15:
        keywords.append("spring outfits")
    else:
        keywords.append("winter outfits")

    if outfit_keywords:
        for kw in outfit_keywords[:2]:
            keywords.append(f"{kw} fashion")

    return keywords[:5]


def fallback(X_new, entry_id):
    conn = get_connection()
    curr = conn.cursor()

    curr.execute("""
        SELECT WEATHER, MOOD, EVENT, DRESS_TOP, TYPE, FABRIC, COLOUR,
               JEANS_SKIRT, LENGTH, TYPE_BOTTOM, COLOUR_BOTTOM, FABRIC_BOTTOM,
               MATCH_PERCENTAGE, USER_RATING
        FROM USER_ENTRIES
        WHERE MATCH_PERCENTAGE IS NOT NULL
    """)
    data = curr.fetchall()

    if len(data) < 5:
        conn.close()
        return 50.0

    # ------------------ PREP DATA ------------------
    X_train_raw = np.array([row[:-2] for row in data], dtype=object)

    # --- IMPROVEMENT 1: Better ML target ---
    # Prefer USER_RATING (real feedback) when available.
    # USER_RATING is 1–5 stars → scaled to 0–100.
    # Fall back to normalized MATCH_PERCENTAGE for unrated entries.
    match_pcts = np.array([row[-2] for row in data], dtype=float)
    user_ratings = [row[-1] for row in data]

    y_min, y_max = match_pcts.min(), match_pcts.max()
    if y_max - y_min > 1e-6:
        norm_match = (match_pcts - y_min) / (y_max - y_min) * 100
    else:
        norm_match = match_pcts.copy()

    Y_train = np.array([
        ((rating - 1) / 4) * 100 if rating is not None else norm_match[i]
        for i, rating in enumerate(user_ratings)
    ], dtype=float)

    # --- IMPROVEMENT 2: Engineered features ---
    def engineer_features(X_raw):
        """
        X_raw columns: WEATHER(0), MOOD(1), EVENT(2), DRESS_TOP(3),
        TYPE(4), FABRIC(5), COLOUR(6), JEANS_SKIRT(7), LENGTH(8),
        TYPE_BOTTOM(9), COLOUR_BOTTOM(10), FABRIC_BOTTOM(11)
        """
        hot_fabrics  = {"cotton", "linen", "chiffon", "rayon", "jersey", "silk"}
        cold_fabrics = {"wool", "velvet", "leather", "corduroy", "twill", "satin"}
        light_colours = {"white", "beige", "light blue", "yellow"}
        dark_colours  = {"black", "navy", "charcoal", "brown", "grey"}

        rows = []
        for row in X_raw:
            try:
                temp = float(row[0])
            except (ValueError, TypeError):
                temp = 20.0

            is_hot   = 1 if temp > 24 else 0
            is_cold  = 1 if temp < 15 else 0
            is_dress = 1 if str(row[3]).lower() == "dress" else 0

            fabric_top = str(row[5]).lower()
            fabric_bot = str(row[11]).lower()
            colour_top = str(row[6]).lower()

            # fabric-weather match heuristic
            if is_hot and fabric_top in hot_fabrics:
                fab_match = 1
            elif is_cold and fabric_top in cold_fabrics:
                fab_match = 1
            elif (is_hot and fabric_top in cold_fabrics) or \
                 (is_cold and fabric_top in hot_fabrics):
                fab_match = -1
            else:
                fab_match = 0

            # colour category
            if colour_top in dark_colours:
                colour_cat = 0
            elif colour_top in light_colours:
                colour_cat = 2
            else:
                colour_cat = 1   # neutral/vivid

            rows.append([is_hot, is_cold, is_dress, fab_match, colour_cat])

        return np.array(rows, dtype=float)

    eng_train = engineer_features(X_train_raw)
    eng_new   = engineer_features([X_new])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_ohe = encoder.fit_transform(X_train_raw)
    X_new_ohe   = encoder.transform([X_new])

    # Combine OHE + engineered features
    X_train = np.hstack([X_train_ohe, eng_train])
    X_new_encoded = np.hstack([X_new_ohe, eng_new])

    # ------------------ TRAIN TEST SPLIT ------------------
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    # ------------------ MODEL ------------------
    # IMPROVEMENT 4: Reduced max_depth + min_samples_leaf to prevent overfitting
    # on a ~500-entry dataset; random_state for reproducibility.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X_train_split, y_train_split)

    # ------------------ FEATURE IMPORTANCE ------------------
    feature_names = encoder.get_feature_names_out()
    importances = model.feature_importances_

    important_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("Top Features:", important_features[:5])

    # ------------------ EVALUATION ------------------
    y_pred = model.predict(X_test_split)
    mae = mean_absolute_error(y_test_split, y_pred)
    print("Model MAE:", round(mae, 2))

    # ------------------ PREDICTION ------------------
    # The model was trained on a 0-100 normalised target, so output is already
    # in that range. Clamp to [0, 100] for safety.
    predicted_percent = float(model.predict(X_new_encoded)[0])
    predicted_percent = round(max(0.0, min(100.0, predicted_percent)), 1)
    print("Min score:", np.min(Y_train))
    print("Max score:", np.max(Y_train))
    print("Avg score:", np.mean(Y_train))
    print("Predicted Score:", predicted_percent)
    print("Actual Data Points:", len(data))
    conn.close()

    return predicted_percent

def loose_overlap_score(outfit_kw, phrase):
    outfit_kw = outfit_kw.lower()
    phrase = phrase.lower()

    outfit_parts = re.findall(r"\w+", outfit_kw)
    phrase_parts = re.findall(r"\w+", phrase)

    overlap = set(outfit_parts) & set(phrase_parts)
    if overlap:
        return 1.0

    return difflib.SequenceMatcher(None, outfit_kw, phrase).ratio()

def calculate_match_percentage(entry_id, code):
    user_data = get_user_data(entry_id)
    user_data_ml = get_user_data_ml(entry_id)

    if not user_data:
        return 50.0, 50.0, 50.0, [], []

    mood, event, temperature, dress_top, type_, fabric, colour, \
    jeans_skirt, length, type_bottom, fabric_bottom, colour_bottom = user_data

    # ------------------ BUILD KEYWORDS ------------------
    if dress_top == "Dress":
        outfit_keywords = [dress_top, type_, fabric, colour]
    else:
        outfit_keywords = [
            dress_top, type_, fabric, colour,
            jeans_skirt, type_bottom, fabric_bottom, colour_bottom, length
        ]

    outfit_keywords = [kw.lower() for kw in outfit_keywords if kw and kw != "N/A"]

    trend_keywords = generate_keywords(mood, event, temperature, outfit_keywords)
    trend_keywords = [kw for kw in trend_keywords if kw and isinstance(kw, str)]

    # ------------------ PYTRENDS ------------------
    pytrends = TrendReq(hl="en-" + code, tz=360, timeout=(10, 25))

    def safe_related_queries(batch_keywords):
        try:
            pytrends.build_payload(batch_keywords, geo=code, timeframe="today 3-m")
            return pytrends.related_queries()
        except Exception:
            return None

    def extract_terms(queries_dict):
        terms = set()
        if isinstance(queries_dict, dict):
            for _, results in queries_dict.items():
                if results is None:
                    continue
                top_df = results.get("top")
                if top_df is not None and not top_df.empty:
                    for _, row in top_df.iterrows():
                        terms.add(row["query"].lower())
        return terms

    related_terms = set()

    for keyword_list in [trend_keywords, outfit_keywords]:
        for i in range(0, len(keyword_list), 5):
            batch = keyword_list[i:i + 5]
            res = safe_related_queries(batch)

            if res:
                related_terms.update(extract_terms(res))

    # ------------------ TREND SCORE ------------------
    if not related_terms:
        trend_score = 50.0
    else:
        weights = {
            "dress": 2,
            "fabric": 1.5,
            "length": 1,
            "default": 1
        }

        hot_fabrics = ["cotton", "linen", "chiffon", "rayon", "jersey", "silk"]
        cold_fabrics = ["wool", "velvet", "leather", "corduroy", "twill", "satin"]

        matched_score = 0

        for kw in outfit_keywords:
            best_score = max(
                [loose_overlap_score(kw, phrase) for phrase in related_terms],
                default=0
            )

            kw_lower = kw.lower()

            if "dress" in kw_lower:
                w = weights["dress"]
            elif kw_lower in hot_fabrics or kw_lower in cold_fabrics:
                w = weights["fabric"]
            elif kw_lower in ["short", "long", "midi", "mini", "maxi"]:
                w = weights["length"]
            else:
                w = weights["default"]

            matched_score += best_score * w

        total_weight = sum(weights.values()) + (len(outfit_keywords) - 3)

        if total_weight == 0:
            trend_score = 50.0
        else:
            trend_score = (matched_score / total_weight) * 100

        # Context adjustments
        if event and event.lower() in ["party", "wedding", "date"]:
            trend_score += 7

        if (temperature >= 24 and any(f in outfit_keywords for f in hot_fabrics)) or \
           (temperature <= 15 and any(f in outfit_keywords for f in cold_fabrics)):
            trend_score += 5

        for f in hot_fabrics:
            if f in outfit_keywords and temperature <= 10:
                trend_score -= 3

        for f in cold_fabrics:
            if f in outfit_keywords and temperature >= 28:
                trend_score -= 3

        trend_score = round(max(trend_score, 0), 1)

    # ------------------ ML SCORE ------------------
    try:
        ml_score = fallback(np.array(user_data_ml, dtype=object), entry_id)
    except Exception:
        ml_score = 50.0
    

    # ------------------ WEATHER PENALTY (SMART) ------------------

    very_bad_items = ["mini", "shorts", "tank", "sundress"]
    not_ideal_fabrics = ["cotton", "linen", "chiffon"]

    weather_penalty = 0

    # COLD WEATHER
    if temperature <= 15:
        if any(item in outfit_keywords for item in very_bad_items):
            weather_penalty -= 25   # very bad
        elif any(fab in outfit_keywords for fab in not_ideal_fabrics):
            weather_penalty -= 10   # mild penalty

    # HOT WEATHER
    elif temperature >= 25:
        if any(fab in outfit_keywords for fab in ["wool", "velvet", "leather","satin","curduroy"]):
            weather_penalty -= 20


    # ------------------ FINAL HYBRID SCORE ------------------

    final_score = (
        ml_score if trend_score == 50
        else 0.6 * ml_score + 0.4 * trend_score
    )

    # apply penalty
    final_score += weather_penalty


    # ------------------ LIGHT CLIMATE ADJUSTMENT (OPTIONAL BOOST ONLY) ------------------

    cold_fabrics = ["wool", "velvet", "corduroy", "leather"]

    if temperature <= 10:
        if any(f in outfit_keywords for f in cold_fabrics):
            final_score += 5   # small reward (no more punishment here)


    # ------------------ FINAL CLAMP ------------------

    final_score = round(max(0, min(100, final_score)), 1)

    return final_score, trend_score, ml_score, outfit_keywords, trend_keywords
    

def highest_match():
    conn = get_connection()
    curr = conn.cursor()
    curr.execute("""
        SELECT * FROM USER_ENTRIES
        WHERE MATCH_PERCENTAGE = (
            SELECT MAX(MATCH_PERCENTAGE) FROM USER_ENTRIES
        )
    """)
    row = curr.fetchone()
    conn.close()
    return row


# ------------------ K-MEANS CLUSTERING ------------------

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(len(X)):
        min_dist = float('inf')
        for j in range(K):
            dist = np.linalg.norm(X[i] - centroids[j])
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):
        points = X[idx == k]

        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
        else:
            centroids[k] = X[np.random.randint(0, m)]

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(X.shape[0], dtype=int)

    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx


# ------------------ CATEGORY ASSIGNMENT ------------------

def check(X, idx):
    categories = []
    cluster_means = {}

    for cluster_id in np.unique(idx):
        cluster_points = X[idx == cluster_id]
        cluster_means[cluster_id] = np.mean(cluster_points) if len(cluster_points) > 0 else 0

    # cluster with highest avg match % = Trendy
    trendy_cluster = max(cluster_means, key=cluster_means.get)

    for label in idx:
        categories.append("Trendy" if label == trendy_cluster else "Non Trendy")

    return categories


def assign_category(entry_id):
    conn = get_connection()
    curr = conn.cursor()

    # IMPROVEMENT 3: Cluster on style/context features so clusters represent
    # STYLE GROUPS rather than score buckets.
    curr.execute("""
        SELECT NO, WEATHER, MOOD, EVENT, DRESS_TOP, FABRIC, COLOUR, MATCH_PERCENTAGE
        FROM USER_ENTRIES
        WHERE MATCH_PERCENTAGE IS NOT NULL
    """)
    data = curr.fetchall()

    if len(data) < 3:
        update_entry_field(entry_id, "CATEGORY", "Neutral")
        conn.close()
        return

    ids        = [row[0] for row in data]
    match_pcts = np.array([row[7] for row in data], dtype=float)

    # Build a numeric feature matrix for clustering
    # WEATHER (numeric) + OHE of MOOD, EVENT, DRESS_TOP, FABRIC, COLOUR
    weathers   = np.array([row[1] if row[1] is not None else 20.0
                           for row in data], dtype=float).reshape(-1, 1)
    cat_fields = np.array([[row[2], row[3], row[4], row[5], row[6]]
                           for row in data], dtype=object)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = enc.fit_transform(cat_fields)

    # Normalise weather to 0-1 so it doesn't dominate
    w_min, w_max = weathers.min(), weathers.max()
    if w_max - w_min > 1e-6:
        weathers_norm = (weathers - w_min) / (w_max - w_min)
    else:
        weathers_norm = np.zeros_like(weathers)

    X = np.hstack([weathers_norm, cat_encoded])

    K = 3
    np.random.seed(42)
    initial_centroids = kMeans_init_centroids(X, K)
    centroids, idx = run_kMeans(X, initial_centroids)

    # Identify "Trendy" cluster = the one with the highest average MATCH_PERCENTAGE
    cluster_means = {}
    for cluster_id in np.unique(idx):
        scores = match_pcts[idx == cluster_id]
        cluster_means[cluster_id] = float(np.mean(scores)) if len(scores) > 0 else 0.0

    sorted_clusters = sorted(cluster_means, key=cluster_means.get, reverse=True)
    trendy_cluster     = sorted_clusters[0]
    mid_cluster        = sorted_clusters[1] if K >= 3 else None

    categories = []
    for label in idx:
        if label == trendy_cluster:
            categories.append("Trendy")
        elif mid_cluster is not None and label == mid_cluster:
            categories.append("Neutral")
        else:
            categories.append("Non Trendy")

    for i, row_id in enumerate(ids):
        curr.execute(
            "UPDATE USER_ENTRIES SET CATEGORY = ? WHERE NO = ?",
            (categories[i], row_id)
        )

    conn.commit()
    conn.close()


# ------------------ SUGGESTIONS BASED ON CATEGORY ------------------

def suggestions(entry_id):
    conn = get_connection()
    curr = conn.cursor()

    # Get current entry
    curr.execute("""
        SELECT NO, USER_ID, LOCATION, WEATHER, MOOD, EVENT, DRESS_TOP, TYPE, FABRIC,
               COLOUR, JEANS_SKIRT, LENGTH, TYPE_BOTTOM, FABRIC_BOTTOM,
               COLOUR_BOTTOM, EMAIL, MATCH_PERCENTAGE, CATEGORY
        FROM USER_ENTRIES
        WHERE NO = ?
    """, (entry_id,))

    data = curr.fetchone()
    if data is None:
        conn.close()
        return None

    (no, user_id, location, weather, mood, event, dress_top, types, fabric, colour,
     jeans_skirt, length, type_bottom, fabric_bottom, colour_bottom,
     email, match_percentage, category) = data

    # ------------------ WEATHER RANGE FILTER ------------------
    if weather <= 12:
        min_temp, max_temp = -5, 12
    elif weather <= 20:
        min_temp, max_temp = 10, 20
    else:
        min_temp, max_temp = 20, 50

    # ------------------ NON TRENDY ------------------
    if category == "Non Trendy":
        curr.execute("""
            SELECT TYPE, FABRIC, COLOUR, LENGTH, TYPE_BOTTOM, FABRIC_BOTTOM, COLOUR_BOTTOM, JEANS_SKIRT
            FROM USER_ENTRIES
            WHERE CATEGORY = 'Trendy'
              AND MOOD = ?
              AND EVENT = ?
              AND JEANS_SKIRT = ?
              AND LOCATION = ?
              AND DRESS_TOP = ?
              AND WEATHER BETWEEN ? AND ?
            ORDER BY MATCH_PERCENTAGE DESC
            LIMIT 1
        """, (mood, event, jeans_skirt, location, dress_top, min_temp, max_temp))

    # ------------------ TRENDY ------------------
    else:
        curr.execute("SELECT MAX(MATCH_PERCENTAGE) FROM USER_ENTRIES")
        highest = curr.fetchone()[0]

        if match_percentage >= highest:
            conn.close()
            return ["You're already wearing a top trendy outfit 🔥"]

        curr.execute("""
            SELECT TYPE, FABRIC, COLOUR, LENGTH, TYPE_BOTTOM, FABRIC_BOTTOM, COLOUR_BOTTOM, JEANS_SKIRT
            FROM USER_ENTRIES
            WHERE CATEGORY = 'Trendy'
              AND MOOD = ?
              AND EVENT = ?
              AND JEANS_SKIRT = ?
              AND LOCATION = ?
              AND DRESS_TOP = ?
              AND WEATHER BETWEEN ? AND ?
            ORDER BY MATCH_PERCENTAGE DESC
            LIMIT 1
        """, (mood, event, jeans_skirt, location, dress_top, min_temp, max_temp))

    trendy = curr.fetchone()
    conn.close()

    # ------------------ NO MATCH FOUND ------------------
    if trendy is None:
        return ["No similar weather-based suggestions yet. Try adding more outfits!"]

    types1, fabric1, colour1, length1, type_bottom1, fabric_bottom1, colour_bottom1, jeans_skirt1 = trendy

    swap = []

    # ------------------ TOP / DRESS ------------------
    if types1 != types:
        swap.append(f"Change your type of Dress/Top to {types1}")
    if fabric1 != fabric:
        swap.append(f"Change your fabric to {fabric1}")
    if colour1 != colour:
        swap.append(f"Change your colour to {colour1}")

    # ------------------ BOTTOM ------------------
    if jeans_skirt == "Skirt":
        if type_bottom1 != type_bottom:
            swap.append(f"Change your type of bottom to {type_bottom1}")
        if fabric_bottom1 != fabric_bottom:
            swap.append(f"Change your fabric of bottom to {fabric_bottom1}")
        if colour_bottom1 != colour_bottom:
            swap.append(f"Change your colour of bottom to {colour_bottom1}")
    else:
        if length1 != length:
            swap.append(f"Change your length of bottom to {length1}")
        if type_bottom1 != type_bottom:
            swap.append(f"Change your type of bottom to {type_bottom1}")
        if fabric_bottom1 != fabric_bottom:
            swap.append(f"Change your fabric of bottom to {fabric_bottom1}")
        if colour_bottom1 != colour_bottom:
            swap.append(f"Change your colour of bottom to {colour_bottom1}")

    return swap if swap else None
def email_summary(email, entry_id, temp, suggestion):
    highest = highest_match()
    conn = get_connection()
    curr = conn.cursor()

    curr.execute("SELECT MATCH_PERCENTAGE FROM USER_ENTRIES WHERE NO = ?", (entry_id,))
    row = curr.fetchone()
    conn.close()

    match_percent = row[0] if row else "N/A"

    text1 = "These Are The Suggestions For Higher Percentages:"
    if suggestion is None:
        text1 += "\nNo Suggestions Available."
        if highest:
            text1 += "\nTry our highest percentage entry:"
            text1 += f"\n{highest}"
    else:
        for i in suggestion:
            text1 += "\n-" + i

    body = (
        f"Temperature at Your Location: {temp}°C\n"
        f"Outfit match percentage: {match_percent}%\n"
        f"Thank You For Using TechThreads.\n"
        f"{text1}"
    )

    msg = MIMEMultipart()
    msg["From"] = "varsha6b@gmail.com"
    msg["To"] = email
    msg["Subject"] = "Your Fashion Match Summary"
    msg.attach(MIMEText(body, "plain"))

    smtpObj = smtplib.SMTP("smtp.gmail.com", 587)
    smtpObj.ehlo()
    smtpObj.starttls()
    
    smtpObj.login(os.getenv("EMAIL"), os.getenv("PASSWORD"))
    smtpObj.sendmail("varsha6b@gmail.com", email, msg.as_string())
    smtpObj.quit()
