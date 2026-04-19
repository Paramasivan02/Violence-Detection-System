

import os
import certifi
# --- START: FORCEFUL SSL CERTIFICATE FIX ---
# This block MUST be at the very top of the script
os.environ['SSL_CERT_FILE'] = certifi.where()
# --- END: FORCEFUL SSL CERTIFICATE FIX ---

from dotenv import load_dotenv
import base64
from datetime import datetime
import streamlit as st
import cv2
import mediapipe as mp
import pickle
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# --- Load Environment Variables ---
# This line loads the .env file
load_dotenv()

# --- Configuration ---
# Get credentials from .env file
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '').strip()
SENDER_APP_PASSWORD = os.environ.get('SENDER_APP_PASSWORD', '').strip()
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/rf_model.pkl')
ALERT_COOLDOWN_SECONDS = int(os.environ.get('ALERT_COOLDOWN_SECONDS', '10'))

is_email_configured = bool(SENDER_EMAIL and SENDER_APP_PASSWORD)

# --- Global Variables & Model Loading ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- THIS IS THE FIX ---
# Create the list of 66 column names that the model expects
# (e.g., x_0, x_1, ..., x_32, y_0, y_1, ..., y_32)
#landmark_names = [f'{coord}_{i}' for coord in ['x', 'y'] for i in range(33)]
# --- END OF FIX ---

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f'Failed to load model at {MODEL_PATH}: {e}')
    st.stop()

if not os.path.exists('alerts'):
    os.makedirs('alerts')

# --- Helper Functions (No Changes) ---

def send_alert_email(receiver_email: str, image_path: str) -> bool:
    """Send an email using Gmail (returns True on success)."""
    if not is_email_configured:
        st.warning('Gmail not configured; set SENDER_EMAIL and SENDER_APP_PASSWORD in .env file.')
        return False
    if not receiver_email or receiver_email == "example@domain.com":
        st.warning('Receiver email empty or default; skipping email.')
        return False

    try:
        # Set up the email
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = '🚨 Anomaly Detected'
        
        body = f'<strong>An anomaly was detected at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.</strong>'
        msg.attach(MIMEText(body, 'html'))
        
        # Attach the image
        with open(image_path, 'rb') as f:
            img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
        
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.quit()
        
        st.success(f'Email sent to {receiver_email}')
        return True
    
    except Exception as e:
        st.error(f'Error sending email: {e}')
        return False


def send_test_email(receiver_email: str) -> None:
    """Send a small test email without attachments."""
    if not is_email_configured:
        st.warning('Gmail not configured; set SENDER_EMAIL and SENDER_APP_PASSWORD in .env file.')
        return
    if not receiver_email or receiver_email == "example@domain.com":
        st.warning('Please enter a valid receiver email address.')
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = 'Test email from Anomaly Detection App'
        msg.attach(MIMEText('This is a test email to confirm your configuration is working.', 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.quit()
        
        st.success(f'Test email sent to {receiver_email}')
    except Exception as e:
        st.error(f'Error sending test email: {e}')


def process_video(cap, video_container, alerts_container, receiver_email: str):
    """Main video processing loop."""
    last_alert_time = None
    alert_log_placeholder = alerts_container.empty() # Placeholder for alerts

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write('Video ended')
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status_text = 'NORMAL'
        box_color = (0, 255, 0)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Create the data row
                row = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
                
                # Create the DataFrame with the correct column names
                X = pd.DataFrame([row])

                proba = model.predict_proba(X)[0]
                pred = int(model.predict(X)[0])

                if pred == 1:
                    status_text = f'ABNORMAL ({proba[1]:.2f})'
                    box_color = (0, 0, 255)
                    now = datetime.now()
                    # Cooldown check
                    if last_alert_time is None or (now - last_alert_time).total_seconds() > ALERT_COOLDOWN_SECONDS:
                        last_alert_time = now
                        ts = now.strftime('%Y%m%d_%H%M%S')
                        alert_path = os.path.join('alerts', f'alert_{ts}.jpg')
                        cv2.imwrite(alert_path, frame)
                        
                        # Update alerts display
                        with alert_log_placeholder.container():
                            st.image(alert_path, caption=f'Alert at {ts}', use_container_width=True)
                            st.write("---") # Add a separator

                        # Send the email
                        send_alert_email(receiver_email, alert_path)
                else:
                    status_text = f'NORMAL ({proba[0]:.2f})'
                    box_color = (0, 255, 0)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        except Exception as e:
            # Display processing errors on the dashboard
            st.error(f'Processing error: {e}')
            break

        # Draw status text on the frame
        cv2.rectangle(image, (0, 0), (400, 60), (255, 255, 255), -1)
        cv2.putText(image, 'STATUS', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        
        # Display the frame
        video_container.image(image, channels='BGR', use_container_width=True)

    cap.release()


def main():
    """Main Streamlit application function."""
    st.set_page_config(page_title="Anomaly Detection",layout='wide')
    st.title('Anomaly Detection')
    st.sidebar.header('Configuration')
    
    # --- CHANGE: REMOVED THE RADIO BUTTON ---
    # source = st.sidebar.radio('Video source', ('Recorded Video', 'Real-time Webcam'))
    # --- END CHANGE ---
    
    receiver = st.sidebar.text_input("Receiver email for alerts", value='example@domain.com')

    # Test email button
    st.sidebar.markdown('---')
    if st.sidebar.button('Send test email'):
        send_test_email(receiver)

    if is_email_configured:
        st.sidebar.success("Email is configured.")
    else:
        st.sidebar.warning('Emails are disabled: set SENDER_EMAIL and SENDER_APP_PASSWORD as env vars.')
    
    st.sidebar.markdown('---')
    st.sidebar.header('Upload Video')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header('Video Feed')
        video_container = st.empty()
    with col2:
        st.header('Alerts')
        alerts_container = st.empty() # This will hold the log of alerts

    cap = None
    run = False
    
    # --- CHANGE: REMOVED THE 'if source == ...' and 'else: ...' ---
    # This is now the only logic
    uploaded = st.sidebar.file_uploader('Drag and drop file here', type=['mp4', 'avi', 'mov', 'mpeg4'], label_visibility="collapsed")
    
    if uploaded is not None:
        # Button to start processing the *uploaded* file
        if st.sidebar.button('Start'):
            tmp = 'temp_video.mp4'
            with open(tmp, 'wb') as f:
                f.write(uploaded.getbuffer())
            
            try:
                cap = cv2.VideoCapture(tmp)
                if not cap.isOpened():
                    st.error("Failed to open the uploaded video file. It might be corrupted or in an unsupported format.")
                else:
                    run = True
            except Exception as e:
                st.error(f"Error opening video file: {e}")
    # --- END CHANGE ---

    # --- Main Processing (No Change) ---
    if run and cap is not None:
        # Clear the alerts container for a new run
        alerts_container.empty()
        process_video(cap, video_container, alerts_container, receiver)
        
        # Clean up the temp file if it exists
        if os.path.exists('temp_video.mp4'):
            try:
                os.remove('temp_video.mp4')
            except Exception as e:
                print(f"Warning: could not remove temp file: {e}") # Non-critical error
    else:
        video_container.info("Select a video source and click 'Start' to begin processing.")

if __name__ == '__main__':
    main()
