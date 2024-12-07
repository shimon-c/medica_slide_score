# Python code to illustrate Sending mail with attachments
# from your Gmail account

# libraries to be imported
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import datetime

fromaddr = "EMAIL address of the sender"
toaddr = "EMAIL address of the receiver"


import smtplib
from email.mime.text import MIMEText

subject = "Email Subject"
body = "This is the body of the text message"
sender = "sender@gmail.com"
recipients = ["recipient1@gmail.com", "recipient2@gmail.com"]
password = "password"


def send_email(subject, body, sender, recipients, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")

subject = "test"
body = "results"
sender = "shimon.cohen1958@gmail.com"
recipients=["shimon.cohen1958@gmail.com"]
password=""
send_email(subject, body, sender, recipients, password)



class SendEmail:
    def __init__(self, fromaddr="shimon.cohen1958@gmail.com", pwd="Lno.04.2023",
                 toaddr="shimon.cohen1958@gmail.com",
                 filename="/mnt/medica/medica_data/for_test_folder_out/slidemgr_results.txt"):
        self.fromaddr = fromaddr
        self.toaddr = toaddr


        # string to store the body of the mail
        cur_time = datetime.datetime.now()
        self.body = f"results on algorithm:\n{cur_time.day}/{cur_time.month}/{cur_time.year}\t{cur_time.hour}:{cur_time.minute}\n"

        result_file = self.create_message(filename)
        self.body = f'{self.body}\n------------------------------------------------\n{result_file}'
        # instance of MIMEMultipart
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security
        s.starttls()
        # Authentication
        s.login(user=fromaddr, password=pwd)


        # attach the body with the msg instance

        # sending the mail
        s.sendmail(fromaddr, toaddr, self.body)

        # terminating the session
        s.quit()

    def create_message(self,filename):
        msg = ''
        file = open(filename, "r")
        for line in file:
            msg = f'{msg}\n{line.strip()}'
        return msg



if __name__ == '__main__':
    sm = SendEmail()
    print(f'message sent chechk your email')
    sys.exit(0)