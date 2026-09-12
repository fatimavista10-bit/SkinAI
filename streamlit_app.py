
import os
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from groq import Groq


# =========================================================
# CONFIG
# =========================================================

MODEL_ID = "SopanhaZinII/facial-condition-model"
MODEL_FILENAME = "finetuned_best.h5"
CSV_PATH = "SkinAI_Product_Database.csv"

CONDITIONS = [
    "Acne",
    "Dry Skin",
    "Oily Skin",
    "Dark Spots",
    "Wrinkles"
]


# =========================================================
# LOAD SKIN MODEL
# =========================================================

print("Loading SkinAI model...")

model_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename=MODEL_FILENAME
)

skin_model = tf.keras.models.load_model(model_path)

print("Skin model loaded!")


# =========================================================
# LOAD PRODUCT DATABASE
# =========================================================

print("Loading product database...")

product_df = pd.read_csv(CSV_PATH)

print(f"Products loaded: {len(product_df)}")


# =========================================================
# GROQ
# =========================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# =========================================================
# LOCATION SEARCH
# =========================================================

def find_nearby_care(location):

    if not location or not location.strip():
        return "⚠️ Please enter a city or area."

    location = location.strip()

    searches = {
        "🩺 Dermatologists": f"dermatologist near {location}",
        "🏥 Skin Clinics": f"skin clinic near {location}",
        "💆 Beauty Salons": f"beauty salon near {location}"
    }

    result = f"""
## 📍 Nearby Skincare Care

**Searching around:** `{location}`

"""

    for title, query in searches.items():

        url = (
            "https://www.google.com/maps/search/?api=1&query="
            + quote_plus(query)
        )

        result += f"- [{title}]({url})\n"

    result += """

### ⚠️ Important

These are location-based search results, not medical endorsements.
Check qualifications, services, and availability before visiting.
"""

    return result


# =========================================================
# MAIN SKINAI FUNCTION
# =========================================================

def analyze_skin(face_image, user_concern, user_budget):

    if face_image is None:
        return "⚠️ Please upload a face photo."

    if not user_concern or not user_concern.strip():
        return "⚠️ Please enter your main skin concern."

    if user_budget is None or user_budget <= 0:
        return "⚠️ Please enter a valid budget."

    user_budget = float(user_budget)

    # -----------------------------------------------------
    # IMAGE PREPROCESSING
    # -----------------------------------------------------

    image = face_image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------------------------------
    # SKIN PREDICTION
    # -----------------------------------------------------

    predictions = skin_model.predict(
        img_array,
        verbose=0
    )[0]

    skin_profile = {}

    for condition, score in zip(CONDITIONS, predictions):

        skin_profile[condition] = round(
            float(score) * 100,
            2
        )

    main_characteristic = CONDITIONS[
        int(np.argmax(predictions))
    ]

    # -----------------------------------------------------
    # MAP MODEL RESULT TO DATABASE
    # -----------------------------------------------------

    skin_type_mapping = {
        "Oily Skin": "Oily",
        "Dry Skin": "Dry, Sensitive",
        "Combination Skin": "Combination",
        "Dark Spots": "Combination",
        "Wrinkles": "Combination"
    }

    database_skin_type = skin_type_mapping.get(
        main_characteristic,
        main_characteristic
    )

    # -----------------------------------------------------
    # BUDGET FILTER
    # -----------------------------------------------------

    affordable_products = product_df[
        product_df["Price_PKR"] <= user_budget
    ].copy()

    # -----------------------------------------------------
    # SKIN PROFILE FILTER
    # -----------------------------------------------------

    matched_products = affordable_products[
        affordable_products["Skin_Profile"].str.contains(
            database_skin_type,
            case=False,
            na=False
        )
    ].copy()

    # -----------------------------------------------------
    # CONCERN MATCHING
    # -----------------------------------------------------

    concern_text = user_concern.lower()

    if "oil" in concern_text or "oily" in concern_text:

        concern_keywords = [
            "Oil Control",
            "Acne, Oil Control"
        ]

    elif "acne" in concern_text or "pimple" in concern_text:

        concern_keywords = [
            "Acne, Oil Control"
        ]

    elif "dry" in concern_text or "dehydrated" in concern_text:

        concern_keywords = [
            "Dryness",
            "Hydration",
            "Dryness, Hydration"
        ]

    elif "dark" in concern_text or "spot" in concern_text:

        concern_keywords = [
            "Dark Spots, Dullness"
        ]

    elif "dull" in concern_text:

        concern_keywords = [
            "Dullness",
            "Dullness, Hydration",
            "Hydration, Dullness"
        ]

    else:

        concern_keywords = []

    if concern_keywords:

        concern_mask = matched_products["Concern"].apply(
            lambda x: any(
                keyword.lower() in str(x).lower()
                for keyword in concern_keywords
            )
        )

        final_products = matched_products[
            concern_mask
        ].copy()

        # If no products match the exact concern,
        # fall back to products matching the AI-estimated
        # skin profile and budget.
        if final_products.empty:
            final_products = matched_products.copy()

    else:

        final_products = matched_products.copy()

    # -----------------------------------------------------
    # ONE PRODUCT PER CATEGORY
    # -----------------------------------------------------

    if not final_products.empty:

        recommended_products = (
            final_products
            .sort_values("Price_PKR")
            .groupby("Category", as_index=False)
            .first()
        )

    else:

        recommended_products = pd.DataFrame(
            columns=product_df.columns
        )

    # -----------------------------------------------------
    # NO MATCH
    # -----------------------------------------------------

    if recommended_products.empty:

        return f"""
## 🧴 SkinAI Results

**AI-estimated visible characteristic:** {main_characteristic}

Unfortunately, SkinAI could not find products in the database
that match your selected concern and budget.

Try increasing your budget or describing your concern differently.

⚠️ This is an AI-based skincare recommendation, not a medical diagnosis.
"""

    # -----------------------------------------------------
    # TOTAL
    # -----------------------------------------------------

    total = int(
        recommended_products["Price_PKR"].sum()
    )

    remaining = int(
        user_budget - total
    )

    # -----------------------------------------------------
    # PRODUCT TEXT FOR GROQ
    # -----------------------------------------------------

    products_text = "\n".join(
        f"- {row['Product']} ({row['Category']}) — Rs. {row['Price_PKR']}"
        for _, row in recommended_products.iterrows()
    )

    # -----------------------------------------------------
    # GROQ EXPLANATION
    # -----------------------------------------------------

    if client:

        prompt = f"""
You are SkinAI, an AI skincare assistant.

AI-estimated visible characteristic:
{main_characteristic}

User's stated concern:
{user_concern}

Budget:
Rs. {int(user_budget)}

FINAL PRODUCTS SELECTED BY THE RULE-BASED ENGINE:
{products_text}

Total cost:
Rs. {total}

Remaining budget:
Rs. {remaining}

Write a concise personalized recommendation using ONLY the information provided above.

STRICT RULES:
1. Mention ONLY the products listed under FINAL PRODUCTS.
2. Do not invent ingredients, benefits, prices, brands, or product properties.
3. Do not make medical claims.
4. Do not diagnose the user.
5. Say "AI-estimated visible characteristic", not "you have".
6. Explain the recommendation only using the product category and the user's stated concern.
7. Mention total cost and remaining budget.
8. Do not encourage buying additional products.
9. End with exactly:
"This is an AI-based skincare recommendation, not a medical diagnosis."
10. Maximum 100 words.

Format:

### 🧴 SkinAI Recommendation

**AI-estimated characteristic:** [characteristic]

**Routine:**
- [Product] — [Category] — Rs. [Price]
- [Product] — [Category] — Rs. [Price]

**Total:** Rs. [total]
**Remaining:** Rs. [remaining]

[One short explanation]

This is an AI-based skincare recommendation, not a medical diagnosis.
"""

        try:

            response = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=180,
                extra_body={
                    "reasoning_effort": "none"
                }
            )

            groq_result = response.choices[0].message.content

            return groq_result

        except Exception as e:

            print("Groq error:", e)

    # -----------------------------------------------------
    # FALLBACK IF GROQ IS UNAVAILABLE
    # -----------------------------------------------------

    routine = "\n".join(
        f"- {row['Product']} — {row['Category']} — Rs. {row['Price_PKR']}"
        for _, row in recommended_products.iterrows()
    )

    return f"""
## 🧴 SkinAI Recommendation

**AI-estimated characteristic:** {main_characteristic}

**Routine:**
{routine}

**Total:** Rs. {total}
**Remaining:** Rs. {remaining}

The routine was selected based on your stated concern,
available products, skin profile matching, and budget.

This is an AI-based skincare recommendation, not a medical diagnosis.
"""



# =========================================================
# STREAMLIT UI
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="SkinAI - Smart Skincare Assistant",
    page_icon="🧴",
    layout="wide"
)

st.title("🧴 SkinAI")
st.subheader("Your AI-powered personal skincare assistant")

st.markdown("""
SkinAI analyzes **visible skin characteristics**, considers your
skincare concern and budget, and suggests a personalized routine.

⚠️ *This is an AI-based skincare recommendation, not a medical diagnosis.*
""")

st.markdown("## 🔍 1. Analyze Your Skin")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📸 Upload Face Photo",
        type=["jpg", "jpeg", "png", "webp"]
    )

with col2:
    concern_input = st.text_area(
        "💬 Main Skin Concern",
        placeholder="Example: My skin feels dry"
    )

    budget_input = st.number_input(
        "💰 Budget (PKR)",
        min_value=1,
        value=4000,
        step=100
    )

if st.button("🔍 Analyze My Skin", type="primary"):
    if uploaded_file is None:
        st.warning("⚠️ Please upload a face photo.")
    elif not concern_input.strip():
        st.warning("⚠️ Please enter your main skin concern.")
    elif budget_input <= 0:
        st.warning("⚠️ Please enter a valid budget.")
    else:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyzing your skin..."):
            result = analyze_skin(
                image,
                concern_input,
                budget_input
            )

        st.markdown("## 📊 Your SkinAI Results")
        st.markdown(result)

st.markdown("---")

st.markdown("## 📍 2. Find Nearby Skincare Care")

st.markdown(
    "Enter a city, area, or neighborhood to search for "
    "dermatologists, skin clinics, and beauty salons."
)

location_input = st.text_input(
    "📍 City / Area",
    placeholder="Example: Lahore, Punjab, Pakistan"
)

if st.button("🔎 Search Nearby"):
    st.markdown(find_nearby_care(location_input))

st.markdown("""
---
**SkinAI** • AI-powered skincare assistance

*Always consult a qualified dermatologist for persistent or serious skin concerns.*
""")
