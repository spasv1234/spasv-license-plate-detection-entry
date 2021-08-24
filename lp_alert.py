import config
import imghdr
import smtplib
from email.message import EmailMessage

def send_lp_alert(license_plate_number,img_file=None):
    with smtplib.SMTP('smtp.gmail.com',587) as smtp:
        #Logins to gmail
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(config.email_login_username,config.email_login_password)

        #Compose the content of the email
        msg = EmailMessage()
        msg['Subject'] = 'SP ASV AI - License Plate Detection'
        msg['From'] = "spasv1234@gmail.com"
        msg['To'] = "spasv1234@gmail.com"
        msg.set_content("License plate detected: " + str(license_plate_number))

        #Attach image if exist
        if img_file is not None:
            with open(img_file,'rb') as f:
                file_data = f.read()
            msg.add_attachment(file_data,maintype='image',subtype='',filename=img_file)

        #Send the email
        smtp.send_message(msg)


