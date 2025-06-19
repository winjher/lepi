import streamlit as st
import datetime
import random
import pandas as pd
import csv
from PIL import Image
import io
import base64

# --- Configuration ---
CSV_FILE = 'butterfly_purchases.csv'

# Sample data for butterfly items
ITEMS = {
    1: {"name": "Clipper", "price": 23},
    2: {"name": "Common Jay", "price": 35},
    3: {"name": "Common Lime", "price": 43},
    4: {"name": "Common Mime", "price": 65},
    5: {"name": "Common Mormon", "price": 48},
    6: {"name": "Emerald Swallowtail", "price": 65},
    7: {"name": "Gray Glassy Tiger", "price": 78},
    8: {"name": "Great Eggfly", "price": 89},
    9: {"name": "Great Yellow Mormon", "price": 71},
    10: {"name": "Golden Birdwing", "price": 73},
    11: {"name": "Paper Kite", "price": 81},
    12: {"name": "Pink Rose", "price": 34},
    13: {"name": "Plain Tiger", "price": 39},
    14: {"name": "Red Lacewing", "price": 100},
    15: {"name": "Scarlet Mormon", "price": 85},
    16: {"name": "Tailed Jay", "price": 45},
    17: {"name": "Atlas Moth", "price": 75},
    18: {"name": "Giant Silk Moth", "price": 80},
}

# --- Helper Functions ---

def initialize_csv():
    """Ensures the CSV file exists with the correct headers."""
    try:
        with open(CSV_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Date', 'OR', 'Image_Filename', 'Quantity', 'Classification_Code',
                'Amount', 'Subtotal'
            ])
    except FileExistsError:
        pass # File already exists

def generate_order_number():
    """Generates a random order number."""
    return random.randint(100000, 999999)

def add_item_to_session_order(item_id, quantity):
    """Adds an item to the current order in Streamlit's session state."""
    if item_id not in ITEMS:
        st.error("Invalid item ID.")
        return

    item_info = ITEMS[item_id]
    st.session_state.current_order.append({
        "item_id": item_id,
        "name": item_info["name"],
        "price": item_info["price"],
        "quantity": quantity,
        "subtotal": item_info["price"] * quantity
    })

def calculate_order_total(order):
    """Calculates the total for a given order."""
    return sum(item["subtotal"] for item in order)

def save_purchase_data(row_data):
    """Saves purchase data to the CSV file."""
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def add_glassmorphism_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1547743519-c0c169b1b4a3?fit=crop&w=1920&q=80');
            background-size: cover;
            background-attachment: fixed;
        }
        .stSidebar > div:first-child {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .main .block-container {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-top: 20px;
        }
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stSelectbox, .stNumberInput, .stFileUploader, .stTextInput label, .stSelectbox label, .stNumberInput label {
            color: white !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        }
        .stTable, .dataframe {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stTable th, .stTable td, .dataframe th, .dataframe td {
            color: white;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stButton button {
            background-color: rgba(69, 170, 242, 0.7);
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: rgba(69, 170, 242, 1);
        }
        .stTextInput > div > div > input, .stSelectbox > div > div > select, .stNumberInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }
        .stInfo, .stSuccess, .stWarning {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
            padding: 10px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def generate_receipt_html(order_details, img_path, order_number, total_amount, date_time):
    """Generates an HTML string for a customized receipt."""
    receipt_items_html = ""
    for item in order_details:
        receipt_items_html += f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>{item['quantity']}x {item['name']}</span>
            <span>${item['subtotal']:.2f}</span>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Purchase Receipt</title>
        <style>
            body {{
                font-family: 'Courier New', Courier, monospace;
                font-size: 12px;
                width: 80mm;
                margin: 0 auto;
                padding: 10px;
                box-sizing: border-box;
                color: #333;
            }}
            .receipt-container {{
                border: 1px dashed #ccc;
                padding: 10px;
            }}
            h3 {{
                text-align: center;
                margin-bottom: 5px;
                font-size: 14px;
            }}
            .header-info, .footer-info {{
                text-align: center;
                margin-bottom: 10px;
            }}
            .item-list {{
                margin-top: 15px;
                border-top: 1px dashed #ccc;
                padding-top: 10px;
                margin-bottom: 15px;
                border-bottom: 1px dashed #ccc;
                padding-bottom: 10px;
            }}
            .total {{
                display: flex;
                justify-content: space-between;
                font-size: 14px;
                font-weight: bold;
                margin-top: 10px;
            }}
            .thank-you {{
                text-align: center;
                margin-top: 20px;
                font-style: italic;
            }}
            .receipt-image {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 10px auto;
            }}
        </style>
    </head>
    <body>
        <div class="receipt-container">
            <h3>Butterfly Haven</h3>
            <div class="header-info">
                <span>Date: {date_time}</span><br>
                <span>Order No: #{order_number}</span>
            </div>
            {'<img src="' + img_path + '" class="receipt-image">' if img_path and img_path != "N/A" else ''}
            <div class="item-list">
                {receipt_items_html}
            </div>
            <div class="total">
                <span>TOTAL:</span>
                <span>${total_amount:.2f}</span>
            </div>
            <div class="thank-you">
                Thank you for your purchase!
            </div>
        </div>
        <script>
            window.onload = function() {{
                window.print();
            }};
        </script>
    </body>
    </html>
    """
    return html_content

# --- Streamlit App Layout ---

st.set_page_config(page_title="Butterfly Purchase System", layout="centered")

add_glassmorphism_style()

st.title("🦋 Butterfly Purchase System")

# Initialize session state for the current order
if 'current_order' not in st.session_state:
    st.session_state.current_order = []
if 'order_number' not in st.session_state:
    st.session_state.order_number = generate_order_number()
if 'last_purchase_details' not in st.session_state:
    st.session_state.last_purchase_details = None

initialize_csv()

st.sidebar.header("Product Catalog")
st.sidebar.markdown("---")
for item_id, item_info in ITEMS.items():
    st.sidebar.write(f"**{item_id}. {item_info['name']}** - ${item_info['price']:.2f}")

st.header(f"Purchase Order: #{st.session_state.order_number}")
current_date_time = datetime.datetime.now()
st.write(f"Date: {current_date_time.strftime('%A, %B %d, %Y')}")

st.subheader("Add Items to Order")

col1, col2 = st.columns(2)
with col1:
    selected_item_id = st.selectbox(
        "Select Butterfly Classification Code",
        options=list(ITEMS.keys()),
        format_func=lambda x: f"{x} - {ITEMS[x]['name']}"
    )
with col2:
    quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

if st.button("Add to Cart"):
    add_item_to_session_order(selected_item_id, quantity)
    st.success(f"Added {quantity} x {ITEMS[selected_item_id]['name']} to cart!")

st.markdown("---")

st.subheader("Your Current Order")

if st.session_state.current_order:
    order_df = pd.DataFrame(st.session_state.current_order)
    order_df = order_df.rename(columns={
        "name": "Butterfly",
        "price": "Unit Price",
        "quantity": "Quantity",
        "subtotal": "Subtotal"
    })
    st.table(order_df[["Butterfly", "Unit Price", "Quantity", "Subtotal"]])
    st.markdown(f"### Total Amount: **${calculate_order_total(st.session_state.current_order):.2f}**")
else:
    st.info("Your cart is empty. Add some items above!")

st.markdown("---")

st.subheader("Upload Butterfly Image (Optional)")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_filename = None
image_base64 = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    buffered = io.BytesIO()
    image_format = image.format if image.format else 'PNG'
    image.save(buffered, format=image_format)
    image_base64 = f"data:image/{image_format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode()}"
    image_filename = uploaded_file.name
    st.info(f"Image '{image_filename}' uploaded. (Note: Image is not permanently stored in this demo.)")

st.markdown("---")

if st.button("Complete Purchase", type="primary"):
    if st.session_state.current_order:
        total_order_amount = calculate_order_total(st.session_state.current_order)
        purchase_date_time = datetime.datetime.now()
        for item in st.session_state.current_order:
            predicted_percent = random.randint(70, 99)
            row_data = [
                purchase_date_time.strftime('%Y-%m-%d %H:%M:%S'), # Date
                st.session_state.order_number,                     # OR (Order number)
                image_filename if image_filename else "N/A",       # Image_Filename
                item["quantity"],                                  # Quantity
                item["item_id"],                                   # Classification_Code
                # item["price"],                                     # Predicted_Price
                #item["subtotal"],                                  # Amount
                total_order_amount                                 # Subtotal (Total for order)
            ]
            save_purchase_data(row_data)
        st.balloons()
        st.success(f"🎉 Purchase complete for Order #{st.session_state.order_number}! Thank you!")
        st.session_state.last_purchase_details = {
            "order": st.session_state.current_order,
            "order_number": st.session_state.order_number,
            "total_amount": total_order_amount,
            "date_time": purchase_date_time.strftime('%Y-%m-%d %H:%M:%S'),
            "image_base64": image_base64
        }
        st.session_state.current_order = []
        st.session_state.order_number = generate_order_number()
        st.rerun()
    else:
        st.warning("Please add items to your cart before completing the purchase.")

if st.session_state.last_purchase_details:
    st.markdown("---")
    st.subheader("Receipt Options")
    if st.button("🖨️ Print Last Receipt"):
        receipt_html = generate_receipt_html(
            st.session_state.last_purchase_details["order"],
            st.session_state.last_purchase_details["image_base64"],
            st.session_state.last_purchase_details["order_number"],
            st.session_state.last_purchase_details["total_amount"],
            st.session_state.last_purchase_details["date_time"]
        )
        st.components.v1.html(receipt_html, height=1, width=1, scrolling=False)
        st.info("A print dialog should appear. If not, check your browser's pop-up settings or print manually (Ctrl/Cmd + P).")

st.markdown("---")
st.subheader("Purchase History")
try:
    history_df = pd.read_csv(CSV_FILE)
    st.dataframe(history_df)
except FileNotFoundError:
    st.info("No purchase history yet.")