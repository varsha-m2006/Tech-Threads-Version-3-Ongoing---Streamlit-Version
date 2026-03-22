import streamlit as st
import numpy as np
import base64
from st_img_selectbox import st_img_selectbox
from PIL import Image
import os

from backend import (
    init_db,
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

st.image("banner.png", use_container_width=True)   # logo size

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "entry_id" not in st.session_state:
    st.session_state.entry_id = None

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

# ------------------ OUTFIT SELECTION ------------------
type_ = "N/A"
fabric = "N/A"
colour = "N/A"
jeans_skirt = "N/A"
length = "N/A"
type_bottom = "N/A"
fabric_bottom = "N/A"
colour_bottom = "N/A"

if dress_choice == "Dress":
    st.subheader("Choose Type of Dress")

    dress_types = [
        "A-line", "Bodycon", "Maxi", "Mini", "Wrap",
        "Sheath", "Shift", "Ballgown", "Sundress", "Cocktail"
    ]

    options = []
    for dress in dress_types:
        img_path = f"{dress}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            options.append({"image": img, "option": dress})
        else:
            st.warning(f"Image not found: {img_path}")

    selected_dress = st_img_selectbox(
        options=options,
        value=dress_types[0],
        height=140,
        fontsize=14,
        key="dress_imgbox"
    )

    if isinstance(selected_dress, list):
        selected_dress = selected_dress[0]

    st.success(f"You selected: {selected_dress}")
    type_ = selected_dress

    fabric = st.selectbox(
        "Dress fabric",
        ["Cotton", "Silk", "Linen", "Polyester", "Wool", "Denim", "Chiffon", "Velvet", "Satin", "Leather"]
    )
    colour = st.selectbox(
        "Dress colour",
        ["Black", "White", "Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Beige", "Brown", "Grey", "Orange"]
    )

else:
    st.subheader("Choose Type of Top")

    top_types = [
        "T-shirt", "Blouse", "Tank Top", "Crop Top", "Shirt",
        "Sweater", "Hoodie", "Cardigan", "Bodysuit", "Tube Top"
    ]

    options = []
    for top in top_types:
        img_path = f"{top}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            options.append({"image": img, "option": top})
        else:
            st.warning(f"Image not found: {img_path}")

    selected_top = st_img_selectbox(
        options=options,
        value=top_types[0],
        height=140,
        fontsize=14,
        key="top_imgbox"
    )

    if isinstance(selected_top, list):
        selected_top = selected_top[0]

    st.success(f"You selected: {selected_top}")
    type_ = selected_top

    fabric = st.selectbox(
        "Top fabric",
        ["Cotton", "Linen", "Silk", "Satin", "Chiffon", "Polyester", "Rayon", "Denim", "Wool", "Jersey"]
    )
    colour = st.selectbox(
        "Top colour",
        ["Black", "White", "Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Beige", "Brown", "Grey", "Orange"]
    )

    jeans_skirt = st.radio("Bottom type", ["Pants", "Skirt"])

    if jeans_skirt == "Pants":
        length = st.selectbox(
            "Pant length",
            ["Full Length", "Ankle Length", "Cropped", "Capri", "Knee Length", "Shorts"]
        )
        st.subheader("Choose Pant Type")

        pant_types = [
            "Straight", "Wide-Leg", "Skinny", "Bootcut", "Tapered",
            "Cargo", "Flared", "Joggers", "Palazzo"
        ]

        options = []
        for pant in pant_types:
            img_path = f"{pant}.png"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                options.append({"image": img, "option": pant})
            else:
                st.warning(f"Image not found: {img_path}")

        selected_pant = st_img_selectbox(
            options=options,
            value=pant_types[0],
            height=140,
            fontsize=14,
            key="pant_imgbox"
        )

        if isinstance(selected_pant, list):
            selected_pant = selected_pant[0]

        st.success(f"You selected: {selected_pant}")
        type_bottom = selected_pant

        fabric_bottom = st.selectbox(
            "Pant fabric",
            ["Denim", "Cotton Blend", "Stretch Denim", "Polyester Blend", "Corduroy", "Twill", "Linen Blend", "Raw Denim"]
        )
        colour_bottom = st.selectbox(
            "Pant colour",
            ["Blue", "Black", "Grey", "White", "Navy", "Light Blue", "Charcoal", "Beige", "Olive", "Brown"]
        )

    else:  # Skirt
        length = "N/A"
        st.subheader("Choose Skirt Type")

        type_bottom_list = ["A-line", "Pencil", "Mini", "Midi", "Maxi", "Wrap", "Pleated", "Skater", "Asymmetrical", "Tulip"]
        skirt_folder = "skirt"  # folder containing skirt images

        options = []
        for skirt in type_bottom_list:
            img_path = os.path.join(skirt_folder, f"{skirt}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                options.append({"image": img, "option": skirt})
            else:
                st.warning(f"Image not found: {img_path}")

        if options:
            selected_skirt = st_img_selectbox(
                options=options,
                value=type_bottom_list[0],
                height=140,
                fontsize=14,
                key="skirt_imgbox"
            )

            if isinstance(selected_skirt, list):
                selected_skirt = selected_skirt[0]

            st.success(f"You selected: {selected_skirt}")
            type_bottom = selected_skirt

        fabric_bottom = st.selectbox(
            "Skirt fabric",
            ["Cotton", "Denim", "Chiffon", "Silk", "Linen", "Wool", "Satin", "Polyester", "Corduroy", "Leather"]
        )
        colour_bottom = st.selectbox(
            "Skirt colour",
            ["Black", "White", "Red", "Pink", "Blue", "Beige", "Brown", "Green", "Yellow", "Purple"]
        )

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