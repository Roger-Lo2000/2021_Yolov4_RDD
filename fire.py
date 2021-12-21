import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("/home/firedfirej/roaddamagedetect-firebase-adminsdk-1tbww-02f570b657.json")
firebase_admin.initialize_app(cred)

