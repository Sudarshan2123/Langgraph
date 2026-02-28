
from google.oauth2 import service_account
from google.auth import default, exceptions as auth_exceptions
from google.cloud import resourcemanager_v3
from typing import Optional
import configparser
import base64
import json
import logging
import time
import os

MAX_RETRIES = 3  # You can adjust the number of retries
RETRY_DELAY_SECONDS = 2  # You can adjust the delay between retries

def load_gcp_credentials() -> Optional[service_account.Credentials]:
    """Loads and authenticates GCP credentials automatically or from environment variables."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Attempt to load default credentials (for GCP environments)
            credentials, project_id = default()
            if credentials and project_id:
                # Attempt to validate credentials by fetching project information
                try:
                    project_client = resourcemanager_v3.ProjectsClient(credentials=credentials)
                    request = resourcemanager_v3.GetProjectRequest(name=f"projects/{project_id}")
                    project_info = project_client.get_project(request=request)
                    logging.info(f"Attempt {attempt}: GCP default credentials loaded and project '{project_info.project_id}' obtained.")
                    return credentials
                except auth_exceptions.GoogleAuthError as auth_error:
                    logging.error(f"Attempt {attempt}: Authentication error with default credentials: {auth_error}")
                    if attempt == MAX_RETRIES:
                        return None
                    time.sleep(RETRY_DELAY_SECONDS)
                except Exception as e:
                    logging.error(f"Attempt {attempt}: Error fetching project info with default credentials: {e}")
                    if attempt == MAX_RETRIES:
                        return None
                    time.sleep(RETRY_DELAY_SECONDS)
                return credentials  # Return if default credentials work

            # If default credentials fail, attempt to load from environment variables
            if "GOOGLE_APPLICATION_CREDENTIALS_CONTENT" in os.environ:
                try:
                    creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"])
                    credentials = service_account.Credentials.from_service_account_info(creds_info)
                    logging.info(f"Attempt {attempt}: GCP credentials loaded from GOOGLE_APPLICATION_CREDENTIALS_CONTENT.")
                    return credentials
                except json.JSONDecodeError:
                    logging.error(f"Attempt {attempt}: Error decoding GOOGLE_APPLICATION_CREDENTIALS_CONTENT.")
                    if attempt == MAX_RETRIES:
                        return None
                    time.sleep(RETRY_DELAY_SECONDS)
                except Exception as e:
                    logging.error(f"Attempt {attempt}: Error creating credentials from content: {e}")
                    if attempt == MAX_RETRIES:
                        return None
                    time.sleep(RETRY_DELAY_SECONDS)
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                    )
                    logging.info(f"Attempt {attempt}: GCP credentials loaded from GOOGLE_APPLICATION_CREDENTIALS file.")
                    return credentials
                except FileNotFoundError:
                    logging.error(f"Attempt {attempt}: Service account key file not found: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
                    if attempt == MAX_RETRIES:
                        return None
                    time.sleep(RETRY_DELAY_SECONDS)
                except Exception as e:
                    logging.error(f"Attempt {attempt}: Error creating credentials from file: {e}")
                    if attempt == MAX_RETRIES:
                        return None
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.warning(f"Attempt {attempt}: Neither default credentials nor environment variables found.")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(RETRY_DELAY_SECONDS)

        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error loading GCP credentials: {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY_SECONDS)

    logging.error("ERROR: Failed to load and validate GCP credentials after multiple retries.")
    return credentials
