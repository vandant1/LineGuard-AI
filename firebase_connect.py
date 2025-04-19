import firebase_admin
from firebase_admin import credentials, db
import logging
from typing import Any, Dict, Optional

# -------------------- Configuration --------------------
FIREBASE_CRED_FILE = "firebase_config.json"  # Make sure this file exists in your project root
FIREBASE_DB_URL = "https://lineguard-ai-default-rtdb.firebaseio.com/"  # Cleaned Realtime DB URL

# -------------------- Initialize Firebase --------------------
def initialize_firebase():
    try:
        if not firebase_admin._apps:  # Prevent reinitialization
            cred = credentials.Certificate(FIREBASE_CRED_FILE)
            firebase_admin.initialize_app(cred, {
                "databaseURL": FIREBASE_DB_URL
            })
            logging.info("✅ Firebase initialized successfully.")
    except Exception as e:
        logging.error(f"❌ Firebase initialization failed: {e}")
        raise

# -------------------- Fetch Real-Time Data --------------------
def get_realtime_data(path: str = "/") -> Dict[str, Any]:
    """
    Fetches real-time data from the specified Firebase path.
    """
    try:
        initialize_firebase()
        ref = db.reference(path)
        data = ref.get()
        logging.info("📥 Data fetched successfully from Firebase.")
        return data if data else {}
    except Exception as e:
        logging.error(f"❌ Failed to fetch data from Firebase: {e}")
        return {}

# -------------------- Push Data to Firebase --------------------
def send_data(path: str, data: Dict[str, Any]) -> bool:
    """
    Sends data to a specified path in Firebase Realtime Database.
    """
    try:
        initialize_firebase()
        ref = db.reference(path)
        ref.set(data)
        logging.info(f"📤 Data sent to Firebase at '{path}'.")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to send data to Firebase: {e}")
        return False

# -------------------- Append New Record to a List Node --------------------
def append_data(path: str, data: Dict[str, Any]) -> Optional[str]:
    """
    Appends data under the given Firebase path using push().
    Useful for adding logs, sensor readings, etc.
    """
    try:
        initialize_firebase()
        ref = db.reference(path)
        new_ref = ref.push(data)
        logging.info(f"➕ Data appended to Firebase at '{path}', new key: {new_ref.key}")
        return new_ref.key
    except Exception as e:
        logging.error(f"❌ Failed to append data to Firebase: {e}")
        return None
