import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def send_teacher_email_inform(student_question):
    try:
        # E-posta ayarları
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        teacher_email = os.getenv("TEACHER_EMAIL")
        
        # E-posta içeriği
        subject = "📚 Öğrencimizden Gelen Anlama Problemi Hakkında | Destek Talebi"

        message = f"""
        🎓 Saygıdeğer Öğretmenimiz,

        Umarız gününüz güzel geçiyordur. 📅 Bugün sizlerle, öğrencimizin yaşadığı bir öğrenme güçlüğünü paylaşmak ve destek talebinde bulunmak istiyoruz. 

        🔍 Aşağıda yer alan soruyu öğrencimiz yöneltmiştir, ancak yanıtı tam olarak anlayamadığını içtenlikle ifade etmiştir:

        ❓ **Öğrenci Sorusu:** {student_question}

        Bu noktada öğrencimizin daha iyi anlayabilmesi ve konuyu kavrayabilmesi adına sizden açıklayıcı, yönlendirici ve örneklerle zenginleştirilmiş bir geri bildirim talep ediyoruz. 🧠💡

        Eğitime olan özveriniz ve katkılarınız için en içten teşekkürlerimizi sunarız. Her bir açıklamanız, bir öğrencinin daha aydınlanmasına vesile olmaktadır. 🌟

        📬 Geri dönüşünüzü sabırsızlıkla bekliyoruz.

        Saygılarımızla ve iyi çalışmalar dileriz,
        🤖 **AI Asistan**
        """
        
        # E-posta oluşturma
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = teacher_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        # SMTP sunucusuna bağlanma ve e-posta gönderme
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, teacher_email, text)
        server.quit()
        
        print("✅ Öğretmene bilgilendirme e-postası gönderildi.")
        return True
        
    except Exception as e:
        print(f"❌ E-posta gönderimi sırasında hata oluştu: {str(e)}")
        return False
