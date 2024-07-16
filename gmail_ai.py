import os
import pickle
import base64
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import webbrowser
import html
from email import message_from_bytes
import quopri
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
import torch
import time

# Spécifiez le modèle et la révision
# Utiliser le premier GPU disponible
device = 0  # Utiliser -1 pour CPU
nbr_email = 4 # Nombre d'emails à traiter

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    device = 0
else:
    print("GPU is not available.")
    device = -1

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            # Utiliser un port fixe, par exemple 58152
            creds = flow.run_local_server(port=58152)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

def get_emails(service, max_results=1):
    # results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    results = service.users().messages().list(userId='me', maxResults=max_results, q='is:inbox').execute()
    
    messages = results.get('messages', [])

    emails = []
    if not messages:
        print('No messages found.')
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            body = get_email_body(service, 'me', message['id'])

            # Trouver la valeur du sujet dans les en-têtes de l'email
            subject_value = "No Subject"
            for header in msg['payload']['headers']:
                if header['name'] == 'Subject':
                    subject_value = header['value']
                    break

            email_data = {
                'id': msg['id'],
                'subject': subject_value,
                'body': body,
                'snippet': msg['snippet']
            }
            # print(f"Email ID: {email_data['id']}")
            # print(f"Title: {email_data['subject']}")
            # print(f"Snippet: {email_data['snippet']}")
            # print(f"Body: {body}")
            # print("\n")
            
            emails.append(email_data)
    return emails

def get_email_body(service, user_id, email_id):
    # Récupérer l'email complet via l'API Gmail
    email = service.users().messages().get(userId=user_id, id=email_id, format='raw').execute()

    # Décoder l'email depuis sa forme base64url
    msg_raw = base64.urlsafe_b64decode(email['raw'].encode('ASCII'))
    msg = message_from_bytes(msg_raw)

    # Trouver le corps de l'email
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                # Décoder le texte si nécessaire (quoted-printable)
                body = part.get_payload(decode=True)
                try:
                    body = quopri.decodestring(body).decode()
                except:
                    pass
                return body
    else:
        # Pour les emails non multipart, le corps est directement accessible
        body = msg.get_payload(decode=True)
        try:
            body = quopri.decodestring(body).decode()
        except:
            pass
        return body

    return ""

# def summarize_email(body):
#     # print("Begin the symmarization process")
#     # print(f"Body: {body}")
#     # Decode the byte string to a standard string if it's not already
#     if isinstance(body, bytes):
#         body = body.decode('utf-8')

#     # Convert HTML encoded characters
#     body = html.unescape(body)
    
#     # Now call the summarizer with the properly formatted string
#     summary = summarizer(body, max_length=100, min_length=25, do_sample=False)
#     return summary

def summarize_email_bart(body, max_length, min_length):
    """
    Summarizes the given email body using the BART model.

    Args:
        body (str): The email body to be summarized.
        max_length (int): The maximum length of the generated summary.
        min_length (int): The minimum length of the generated summary.

    Returns:
        str: The generated summary of the email body.
    """
    # Initialize the tokenizer and the model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')  # Use 'facebook/bart-large-cnn-cnn' for longer max_length

    # Prepare the summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

    # Ensure that the body is a string
    if isinstance(body, bytes):
        body = body.decode('utf-8', 'ignore')
    else :
        print("The body is not a byte string")
        # body = str(body)

    # Convert encoded characters to HTML
    body = html.unescape(body)
    
    # Check the input length and truncate if necessary
    inputs = tokenizer.encode("summarize: " + body, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Generate the summar
    print("Begin the symmarization process")
    start_time = time.time()
    # Le summary_ids est une variable qui contient les identifiants des tokens générés par le modèle de résumé BART. 
    # Ces identifiants représentent la séquence de tokens qui constitue le résumé généré.
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    # print(f"Summary_ids: {summary_ids}")

    print("Begin the tokenizer process")
    start_time = time.time()

    # Convert the summary_ids to text
    summaries = [tokenizer.decode(summary_id, skip_special_tokens=True) for summary_id in summary_ids]
    summary = " ".join(summaries)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    # print(f"Summary: {summary}")

    summary = html.unescape(summary)

    return summary


if __name__ == '__main__':

    # # Charger le tokenizer et le modèle
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # # Obtenir la longueur maximale du modèles
    # max_length = tokenizer.model_max_length

    # print(f"La longueur maximale de séquence que le modèle peut gérer est de {max_length} tokens.")


    service = authenticate_gmail()
    emails = get_emails(service, nbr_email)
    all_summaries = ""  # Initialize the variable all_summaries
    for email in emails:
        # print(f"Email ID: {email['id']}")
        # print(f"Snippet: {email['snippet']}")
        # summary = summarize_email(email['body'])
        print(f"Titre: {email['subject']}")
        summary = summarize_email_bart(email['body'], 1024, 100)
        summary.replace('<!doctype html>', '')
        # print(f"Summary: {summary}")
        # print("\n")
        
        # # Ajouter le résumé à la variable all_summaries (with summarize_email)
        # all_summaries += f"Titre: {html.unescape(email['subject'])}\n\n<br>Summary: {summary[0]['summary_text']}\n\n<br><br>"
        all_summaries += f"Titre: {html.unescape(email['subject'])}\n\n<br>Summary: {summary.replace('. ', '.<br>')}\n\n<br><br>"
    
    # Ouvrir une seule page avec tous les résumés
    # print(f"all_summaries: {all_summaries}")
    webbrowser.open_new_tab(f"data:text/html;charset=utf-8,<meta charset='UTF-8'>{all_summaries}")



