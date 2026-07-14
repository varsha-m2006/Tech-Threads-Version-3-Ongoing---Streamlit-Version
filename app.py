import streamlit as st
import numpy as np
import base64
import os
import time
import hashlib
from PIL import Image
from st_img_selectbox import st_img_selectbox
import io

from backend import (
    init_db,
    migrate_db,
    get_or_create_user,
    create_user_entry,
    update_entry_field,
    api_temp,
    geocode,
    calculate_match_percentage,
    get_user_data_ml,
    fallback,
    email_summary,
    suggestions,
    assign_category,
    
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Tech Threads - Outfit Match", layout="centered")

# ------------------ LOAD BACKGROUND ------------------
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("background.jpg")

# ------------------ STYLING ------------------
st.markdown(f"""
<style>

/* BACKGROUND */
.stApp {{
    background: linear-gradient(
        rgba(255, 255, 255, 0.4),
        rgba(255, 255, 255, 0.4)
    ),
    url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* MAIN CONTAINER (GLASS EFFECT) */
.block-container {{
    background: rgba(255, 255, 255, 0.45);
    padding: 2rem;
    border-radius: 22px;
    backdrop-filter: blur(20px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.3);
}}

/* INPUT BOXES */
input, textarea {{
    background-color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    color: #111827 !important;
    padding: 0.5rem !important;
}}

div[data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,0.85) !important;
    border-radius: 12px !important;
}}

input::placeholder {{
    color: #6b7280 !important;
}}

/* LABELS */
label {{
    color: #1f2937 !important;
    font-weight: 600 !important;
}}

/* BUTTON */
div.stButton > button {{
    width: 100%;
    background: linear-gradient(90deg, #ff4b91, #7b61ff);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.8rem;
    font-size: 1rem;
    font-weight: 700;
    box-shadow: 0 6px 20px rgba(123, 97, 255, 0.45);
    transition: 0.3s ease;
}}

div.stButton > button:hover {{
    transform: translateY(-2px);
    opacity: 0.95;
}}

/* METRIC BOX */
div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, #fff1f2, #f3e8ff);
    padding: 1rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
}}

/* SUGGESTIONS */
.suggestion-box {{
    background: linear-gradient(135deg, #fdf2f8, #ede9fe);
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 0.8rem;
    color: #4b5563;
}}

</style>
""", unsafe_allow_html=True)
init_db()
migrate_db()

st.image("banner.png", use_container_width=True)   # logo size

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "entry_id" not in st.session_state:
    st.session_state.entry_id = None

# Detected colours stored in session state so selectboxes update correctly
for _key in ["detected_dress_colour", "detected_top_colour", "detected_pant_colour", "detected_skirt_colour"]:
    if _key not in st.session_state:
        st.session_state[_key] = None

location = st.text_input("Enter your location")
email = st.text_input("Enter your email for summary")

moods = [
    "Happy", "Relaxed", "Energetic", "Confident", "Romantic",
    "Casual", "Professional", "Tired", "Adventurous", "Reserved"
]

events = [
    "Casual", "Work / Office", "Formal", "Party",
    "Date", "Wedding", "Outdoor", "Interview"
]

dress_choice = st.radio("Choose outfit type", ["Dress", "Top+Bottoms"])
mood = st.selectbox("Select your mood", moods)
event = st.selectbox("Select your event", events)

# ------------------ COLOUR DETECTION FALLBACK ------------------
def detect_dominant_colour(image_bytes, colour_options):
    """Extract dominant colour from image using PIL and map to nearest option."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((80, 80))
    pixels = np.array(img).reshape(-1, 3).astype(float)

    # Only remove near-white (background) and near-black pixels
    # Keep light colours like yellow, beige, pink
    mask = ~(
        (pixels.min(axis=1) > 210) |  # near white
        (pixels.max(axis=1) < 25)      # near black
    )
    filtered = pixels[mask] if mask.sum() > 20 else pixels
    r, g, b = filtered.mean(axis=0)

    colour_map = {
        "Black":      (20,  20,  20),
        "White":      (240, 240, 240),
        "Grey":       (128, 128, 128),
        "Charcoal":   (60,  60,  65),
        "Red":        (200, 30,  30),
        "Blue":       (30,  80,  200),
        "Navy":       (20,  30,  100),
        "Light Blue": (130, 180, 230),
        "Green":      (30,  160, 30),
        "Olive":      (100, 110, 40),
        "Yellow":     (240, 220, 30),
        "Orange":     (220, 110, 30),
        "Pink":       (230, 130, 160),
        "Purple":     (120, 30,  160),
        "Beige":      (210, 190, 150),
        "Brown":      (120, 70,  30),
    }

    best_colour, best_dist = colour_options[0], float("inf")
    for name, (cr, cg, cb) in colour_map.items():
        if name not in colour_options:
            continue
        dist = ((r-cr)**2 + (g-cg)**2 + (b-cb)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_colour = name
    return best_colour


# ------------------ VISION HELPER ------------------
def analyse_outfit_image(image_bytes, item_type, options_map):
    """Try Gemini Vision first; fall back to PIL colour detection if quota exceeded."""
    import requests
    import json

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        options_text = "\n".join(f"- {k}: {v}" for k, v in options_map.items())
        prompt = f"""You are analysing a clothing image to extract outfit attributes.
The item is a {item_type}.
For each attribute below, pick EXACTLY one value from the provided list.
Return ONLY a valid JSON object with no extra text, no markdown, no explanation.
Attributes and allowed values:
{options_text}
Rules:
- Pick from allowed values only.
- If unsure, pick the closest match.
- Return only the JSON object."""

        payload = {"contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            {"text": prompt}
        ]}]}

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )

        for attempt in range(3):
            res = requests.post(url, json=payload, timeout=30)
            data = res.json()
            if res.status_code == 429 or "error" in data:
                print(f"Gemini attempt {attempt+1} failed: {data.get('error',{}).get('message','unknown')[:80]}")
                time.sleep(15)
                continue
            try:
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                print("Gemini SUCCESS:", raw[:100])
                return json.loads(raw), "gemini"
            except Exception as e:
                print("Gemini parse error:", e)
                break

    print("Using PIL fallback")

    # ---- PIL fallback: colour + shape-based type ----
    result = {}
    for key, val in options_map.items():
        options_list = [c.strip() for c in val.replace("Choose one of:", "").split(",")]
        if key == "colour":
            result[key] = detect_dominant_colour(image_bytes, options_list)
        elif key == "type":
            result[key] = detect_type_from_shape(image_bytes, item_type, options_list)
    return result, "fallback"


def detect_type_from_shape(image_bytes, item_type, type_options):
    """Guess clothing type from image aspect ratio."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    ratio = h / max(w, 1)

    if item_type == "dress":
        if ratio > 1.8:
            candidates = ["Maxi", "A-line", "Sheath"]
        elif ratio > 1.3:
            candidates = ["Wrap", "Bodycon", "A-line", "Midi"]
        else:
            candidates = ["Mini", "Cocktail", "Shift"]
    elif item_type == "top / shirt":
        if ratio > 1.4:
            candidates = ["Blouse", "Shirt", "Sweater", "Hoodie", "Cardigan"]
        elif ratio < 0.9:
            candidates = ["Crop Top", "Tube Top", "Tank Top"]
        else:
            candidates = ["T-shirt", "Bodysuit", "Blouse"]
    elif item_type == "pants / trousers":
        if ratio > 2.5:
            candidates = ["Straight", "Wide-Leg", "Flared", "Palazzo"]
        elif ratio > 1.5:
            candidates = ["Skinny", "Tapered", "Bootcut", "Joggers"]
        else:
            candidates = ["Cargo", "Straight"]
    elif item_type == "skirt":
        if ratio > 2.0:
            candidates = ["Maxi", "Midi", "Pleated"]
        elif ratio > 1.2:
            candidates = ["A-line", "Wrap", "Skater", "Pencil"]
        else:
            candidates = ["Mini", "Tulip", "Asymmetrical"]
    else:
        candidates = []

    for c in candidates:
        if c in type_options:
            return c
    return type_options[0]


def safe_index(options_list, value):
    """Return index of value in list, or 0 if not found."""
    try:
        return options_list.index(value)
    except ValueError:
        return 0


# ------------------ CACHED DETECTION HELPER ------------------
def run_detection_once(img_bytes, item_key, item_type, options_map):
    """
    Run analyse_outfit_image only if the image hasn't been analysed yet.
    Uses a hash of the image bytes as the cache key so that:
      - uploading a NEW image triggers fresh detection
      - reruns (button click, rating) skip detection entirely
    Returns (result_dict, mode_str).
    """
    img_hash = hashlib.md5(img_bytes).hexdigest()
    cache_key = f"_detection_cache_{item_key}"

    cached = st.session_state.get(cache_key)
    if cached and cached.get("hash") == img_hash:
        # Already detected for this exact image — return cached result
        return cached["result"], cached["mode"]

    # First time seeing this image — run detection
    result, mode = analyse_outfit_image(img_bytes, item_type, options_map)

    st.session_state[cache_key] = {
        "hash": img_hash,
        "result": result,
        "mode": mode,
    }
    return result, mode


# ------------------ OUTFIT SELECTION ------------------
type_ = "N/A"
fabric = "N/A"
colour = "N/A"
jeans_skirt = "N/A"
length = "N/A"
type_bottom = "N/A"
fabric_bottom = "N/A"
colour_bottom = "N/A"

DRESS_TYPES   = ["A-line", "Bodycon", "Maxi", "Mini", "Wrap", "Sheath", "Shift", "Ballgown", "Sundress", "Cocktail"]
TOP_TYPES     = ["T-shirt", "Blouse", "Tank Top", "Crop Top", "Shirt", "Sweater", "Hoodie", "Cardigan", "Bodysuit", "Tube Top"]
PANT_TYPES    = ["Straight", "Wide-Leg", "Skinny", "Bootcut", "Tapered", "Cargo", "Flared", "Joggers", "Palazzo"]
SKIRT_TYPES   = ["A-line", "Pencil", "Mini", "Midi", "Maxi", "Wrap", "Pleated", "Skater", "Asymmetrical", "Tulip"]
COLOURS       = ["Black", "White", "Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Beige", "Brown", "Grey", "Orange"]
PANT_COLOURS  = ["Blue", "Black", "Grey", "White", "Navy", "Light Blue", "Charcoal", "Beige", "Olive", "Brown"]
SKIRT_COLOURS = ["Black", "White", "Red", "Pink", "Blue", "Beige", "Brown", "Green", "Yellow", "Purple"]
PANT_LENGTHS  = ["Full Length", "Ankle Length", "Cropped", "Capri", "Knee Length", "Shorts"]

if dress_choice == "Dress":
    st.subheader("Upload your dress photo")
    st.caption("Colour will be auto-detected. Choose type below.")

    dress_img_file = st.file_uploader(
        "Upload dress image", type=["jpg", "jpeg", "png"], key="dress_upload"
    )

    if dress_img_file is not None:
        img_bytes = dress_img_file.read()
        st.image(img_bytes, width=220, caption="Your dress")

        # ---- Only detect if this image hasn't been seen before ----
        cache_key = "_detection_cache_dress"
        img_hash = hashlib.md5(img_bytes).hexdigest()
        already_cached = (
            st.session_state.get(cache_key, {}).get("hash") == img_hash
        )

        if not already_cached:
            with st.spinner("Detecting colour..."):
                try:
                    result, mode = run_detection_once(
                        img_bytes, "dress", "dress",
                        {
                            "type":   f"Choose one of: {', '.join(DRESS_TYPES)}",
                            "colour": f"Choose one of: {', '.join(COLOURS)}"
                        }
                    )
                    detected_colour = result.get("colour") or COLOURS[0]
                    detected_type   = result.get("type")   or DRESS_TYPES[0]
                    st.session_state["dress_colour_sel"]    = detected_colour
                    st.session_state["dress_type_detected"] = detected_type
                    if mode == "gemini":
                        st.success(f"✓ AI detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                    else:
                        st.info(f"✓ Detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                except Exception as e:
                    st.warning(f"Could not analyse image: {e}. Please select manually.")
        else:
            cached = st.session_state[cache_key]
            detected_colour = cached["result"].get("colour") or COLOURS[0]
            detected_type   = cached["result"].get("type")   or DRESS_TYPES[0]
            mode = cached["mode"]
            label = "✓ AI detected" if mode == "gemini" else "✓ Detected"
            st.info(f"{label}: **{detected_type}**, **{detected_colour}** — adjust below if needed.")

    st.subheader("Choose Type of Dress")
    options = []
    for dress in DRESS_TYPES:
        img_path = f"{dress}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            options.append({"image": img, "option": dress})
    if options:
        default_dress = st.session_state.get("dress_type_detected", DRESS_TYPES[0])
        selected_dress = st_img_selectbox(options=options, value=default_dress, height=140, fontsize=14, key="dress_imgbox")
        if isinstance(selected_dress, list):
            selected_dress = selected_dress[0]
        type_ = selected_dress

    colour = st.selectbox("Dress colour", COLOURS, key="dress_colour_sel")
    fabric = st.selectbox("Dress fabric", ["Cotton", "Silk", "Linen", "Polyester", "Wool", "Denim", "Chiffon", "Velvet", "Satin", "Leather"], key="dress_fabric_sel")

else:
    # ------------------ TOP ------------------
    st.subheader("Upload your top photo")
    st.caption("Colour will be auto-detected. Choose type below.")

    top_img_file = st.file_uploader("Upload top image", type=["jpg", "jpeg", "png"], key="top_upload")

    if top_img_file is not None:
        img_bytes = top_img_file.read()
        st.image(img_bytes, width=220, caption="Your top")

        cache_key = "_detection_cache_top"
        img_hash = hashlib.md5(img_bytes).hexdigest()
        already_cached = (
            st.session_state.get(cache_key, {}).get("hash") == img_hash
        )

        if not already_cached:
            with st.spinner("Detecting colour..."):
                try:
                    result, mode = run_detection_once(
                        img_bytes, "top", "top / shirt",
                        {
                            "type":   f"Choose one of: {', '.join(TOP_TYPES)}",
                            "colour": f"Choose one of: {', '.join(COLOURS)}"
                        }
                    )
                    detected_colour = result.get("colour") or COLOURS[0]
                    detected_type   = result.get("type")   or TOP_TYPES[0]
                    st.session_state["top_colour_sel"]    = detected_colour
                    st.session_state["top_type_detected"] = detected_type
                    if mode == "gemini":
                        st.success(f"✓ AI detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                    else:
                        st.info(f"✓ Detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                except Exception as e:
                    st.warning(f"Could not analyse image: {e}. Please select manually.")
        else:
            cached = st.session_state[cache_key]
            detected_colour = cached["result"].get("colour") or COLOURS[0]
            detected_type   = cached["result"].get("type")   or TOP_TYPES[0]
            mode = cached["mode"]
            label = "✓ AI detected" if mode == "gemini" else "✓ Detected"
            st.info(f"{label}: **{detected_type}**, **{detected_colour}** — adjust below if needed.")

    st.subheader("Choose Type of Top")
    options = []
    for top in TOP_TYPES:
        img_path = f"{top}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            options.append({"image": img, "option": top})
    if options:
        default_top = st.session_state.get("top_type_detected", TOP_TYPES[0])
        selected_top = st_img_selectbox(options=options, value=default_top, height=140, fontsize=14, key="top_imgbox")
        if isinstance(selected_top, list):
            selected_top = selected_top[0]
        type_ = selected_top

    colour = st.selectbox("Top colour", COLOURS, key="top_colour_sel")
    fabric = st.selectbox("Top fabric", ["Cotton", "Linen", "Silk", "Satin", "Chiffon", "Polyester", "Rayon", "Denim", "Wool", "Jersey"], key="top_fabric_sel")

    # ------------------ BOTTOM ------------------
    jeans_skirt = st.radio("Bottom type", ["Pants", "Skirt"])

    if jeans_skirt == "Pants":
        st.subheader("Upload your pants photo")
        st.caption("Colour will be auto-detected. Choose type below.")

        pant_img_file = st.file_uploader("Upload pants image", type=["jpg", "jpeg", "png"], key="pant_upload")

        if pant_img_file is not None:
            img_bytes = pant_img_file.read()
            st.image(img_bytes, width=220, caption="Your pants")

            cache_key = "_detection_cache_pant"
            img_hash = hashlib.md5(img_bytes).hexdigest()
            already_cached = (
                st.session_state.get(cache_key, {}).get("hash") == img_hash
            )

            if not already_cached:
                with st.spinner("Detecting colour..."):
                    try:
                        result, mode = run_detection_once(
                            img_bytes, "pant", "pants / trousers",
                            {
                                "type":   f"Choose one of: {', '.join(PANT_TYPES)}",
                                "colour": f"Choose one of: {', '.join(PANT_COLOURS)}"
                            }
                        )
                        detected_colour = result.get("colour") or PANT_COLOURS[0]
                        detected_type   = result.get("type")   or PANT_TYPES[0]
                        st.session_state["pant_colour_sel"]    = detected_colour
                        st.session_state["pant_type_detected"] = detected_type
                        if mode == "gemini":
                            st.success(f"✓ AI detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                        else:
                            st.info(f"✓ Detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                    except Exception as e:
                        st.warning(f"Could not analyse image: {e}. Please select manually.")
            else:
                cached = st.session_state[cache_key]
                detected_colour = cached["result"].get("colour") or PANT_COLOURS[0]
                detected_type   = cached["result"].get("type")   or PANT_TYPES[0]
                mode = cached["mode"]
                label = "✓ AI detected" if mode == "gemini" else "✓ Detected"
                st.info(f"{label}: **{detected_type}**, **{detected_colour}** — adjust below if needed.")

        st.subheader("Choose Pant Type")
        options = []
        for pant in PANT_TYPES:
            img_path = f"{pant}.png"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                options.append({"image": img, "option": pant})
        if options:
            default_pant = st.session_state.get("pant_type_detected", PANT_TYPES[0])
            selected_pant = st_img_selectbox(options=options, value=default_pant, height=140, fontsize=14, key="pant_imgbox")
            if isinstance(selected_pant, list):
                selected_pant = selected_pant[0]
            type_bottom = selected_pant

        colour_bottom = st.selectbox("Pant colour", PANT_COLOURS, key="pant_colour_sel")
        fabric_bottom = st.selectbox("Pant fabric", ["Denim", "Cotton Blend", "Stretch Denim", "Polyester Blend", "Corduroy", "Twill", "Linen Blend", "Raw Denim"], key="pant_fabric_sel")
        length = st.selectbox("Pant length", PANT_LENGTHS, key="pant_length_sel")

    else:  # Skirt
        length = "N/A"
        st.subheader("Upload your skirt photo")
        st.caption("Colour will be auto-detected. Choose type below.")

        skirt_img_file = st.file_uploader("Upload skirt image", type=["jpg", "jpeg", "png"], key="skirt_upload")

        if skirt_img_file is not None:
            img_bytes = skirt_img_file.read()
            st.image(img_bytes, width=220, caption="Your skirt")

            cache_key = "_detection_cache_skirt"
            img_hash = hashlib.md5(img_bytes).hexdigest()
            already_cached = (
                st.session_state.get(cache_key, {}).get("hash") == img_hash
            )

            if not already_cached:
                with st.spinner("Detecting colour..."):
                    try:
                        result, mode = run_detection_once(
                            img_bytes, "skirt", "skirt",
                            {
                                "type":   f"Choose one of: {', '.join(SKIRT_TYPES)}",
                                "colour": f"Choose one of: {', '.join(SKIRT_COLOURS)}"
                            }
                        )
                        detected_colour = result.get("colour") or SKIRT_COLOURS[0]
                        detected_type   = result.get("type")   or SKIRT_TYPES[0]
                        st.session_state["skirt_colour_sel"]    = detected_colour
                        st.session_state["skirt_type_detected"] = detected_type
                        if mode == "gemini":
                            st.success(f"✓ AI detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                        else:
                            st.info(f"✓ Detected: **{detected_type}**, **{detected_colour}** — adjust below if needed.")
                    except Exception as e:
                        st.warning(f"Could not analyse image: {e}. Please select manually.")
            else:
                cached = st.session_state[cache_key]
                detected_colour = cached["result"].get("colour") or SKIRT_COLOURS[0]
                detected_type   = cached["result"].get("type")   or SKIRT_TYPES[0]
                mode = cached["mode"]
                label = "✓ AI detected" if mode == "gemini" else "✓ Detected"
                st.info(f"{label}: **{detected_type}**, **{detected_colour}** — adjust below if needed.")

        st.subheader("Choose Skirt Type")
        skirt_folder = "skirt"
        options = []
        for skirt in SKIRT_TYPES:
            img_path = os.path.join(skirt_folder, f"{skirt}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                options.append({"image": img, "option": skirt})
        if options:
            default_skirt = st.session_state.get("skirt_type_detected", SKIRT_TYPES[0])
            selected_skirt = st_img_selectbox(options=options, value=default_skirt, height=140, fontsize=14, key="skirt_imgbox")
            if isinstance(selected_skirt, list):
                selected_skirt = selected_skirt[0]
            type_bottom = selected_skirt

        colour_bottom = st.selectbox("Skirt colour", SKIRT_COLOURS, key="skirt_colour_sel")
        fabric_bottom = st.selectbox("Skirt fabric", ["Cotton", "Denim", "Chiffon", "Silk", "Linen", "Wool", "Satin", "Polyester", "Corduroy", "Leather"], key="skirt_fabric_sel")

if st.button("Get Match Percentage"):
    try:
        if not location.strip():
            st.error("Please enter your location.")
        elif not email.strip():
            st.error("Please enter your email.")
        else:
            user_id = get_or_create_user(email, location)
            entry_id = create_user_entry(user_id)

            st.session_state.user_id = user_id
            st.session_state.entry_id = entry_id

            temp = api_temp(user_id)
            code = geocode(user_id)

            update_entry_field(entry_id, "WEATHER", temp)
            update_entry_field(entry_id, "MOOD", mood)
            update_entry_field(entry_id, "EVENT", event)
            update_entry_field(entry_id, "DRESS_TOP", dress_choice)
            update_entry_field(entry_id, "TYPE", type_)
            update_entry_field(entry_id, "FABRIC", fabric)
            update_entry_field(entry_id, "COLOUR", colour)
            update_entry_field(entry_id, "JEANS_SKIRT", jeans_skirt)
            update_entry_field(entry_id, "LENGTH", length)
            update_entry_field(entry_id, "TYPE_BOTTOM", type_bottom)
            update_entry_field(entry_id, "FABRIC_BOTTOM", fabric_bottom)
            update_entry_field(entry_id, "COLOUR_BOTTOM", colour_bottom)

            try:
                match_percent, trend_score, ml_score, matched, trend_keywords = calculate_match_percentage(entry_id, code)
            except Exception:
                # fallback safety
                match_percent = 50.0
                trend_score = 50.0
                ml_score = 50.0


            update_entry_field(entry_id, "MATCH_PERCENTAGE", match_percent)
            assign_category(entry_id)
            recs = suggestions(entry_id)

            st.success(f"Temperature in your location: {temp}°C")
            st.metric("Final Match Score", f"{match_percent}%")

            col1, col2 = st.columns(2)
            if(trend_score!=50):
                col1.metric("Trend Score", f"{trend_score}%")
            else:
                col1.info("Google PyTrends Currently Out of Service")
            col2.metric("ML Score", f"{ml_score}%")
            update_entry_field(entry_id, "TREND_SCORE", trend_score)
            update_entry_field(entry_id, "ML_SCORE", ml_score)

            st.subheader("Suggestions")
            
            if recs:
                for rec in recs:
                    st.write(f"- {rec}")
            else:
                st.write("No suggestions available.")

            try:
                email_summary(email, entry_id, temp, recs)
                st.info("Summary email sent successfully.")
            except Exception as email_error:
                st.warning(f"Match calculated, but email could not be sent: {email_error}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ------------------ STAR RATING (outside button block) ------------------
# Must live here so it survives Streamlit's rerun when the user clicks a star
if st.session_state.entry_id is not None:
    st.markdown("---")
    st.subheader("Rate this outfit match")
    st.caption("Your rating helps personalise future predictions.")

    star_rating = st.feedback("stars", key=f"rating_{st.session_state.entry_id}")

    if star_rating is not None:
        rating_value = star_rating + 1
        update_entry_field(st.session_state.entry_id, "USER_RATING", float(rating_value))
        st.success(f"Thanks! You rated this outfit {rating_value} ★")
