import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def send_teacher_email_inform_quiz(student_question,quiz_result):
    try:
        # E-posta ayarları
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        teacher_email = os.getenv("TEACHER_EMAIL")
        
        # E-posta içeriği
        subject = "🎯 Öğrenci Quiz Değerlendirmesi | Quiz Sonucu ve Bilgilendirme"

        message = f"""
         🎓 Saygıdeğer Öğretmenimiz,

        Umarız gününüz güzel geçiyordur. 📅 Bugün sizlere, öğrencimizin {student_question} konusuyla ilgili olarak daha önce yaşadığı anlama güçlüğünün ardından uygulanan kısa quizin sonucunu memnuniyetle paylaşmak istiyoruz.

        ✨ Öğrencimiz, yönlendirmeleriniz ve gösterdiğiniz özverili rehberlik sayesinde ilgili konuyu başarıyla kavramış ve quizde etkileyici bir performans sergilemiştir. Bu gelişme, sizin kıymetli katkılarınızın ne kadar etkili ve yol gösterici olduğunu bir kez daha göstermiştir.

        ❓ **Öğrenci Puanı:** {quiz_result}

        🌟 Bu başarı, yalnızca öğrencimizin değil, aynı zamanda siz değerli öğretmenimizin de başarısıdır. Her bir açıklamanız, sadece soruları değil, öğrenme sürecini de aydınlatıyor. 🌟

        📬 Bu güzel gelişmeyi sizlerle paylaşmak istedik ve emekleriniz için en içten teşekkürlerimizi sunarız.

        🤖 **AI Asistan**

        Saygılarımızla ve iyi çalışmalar dileriz,  
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
        
        print("✅ Öğretmene quiz bilgilendirme e-postası gönderildi.")
        return True
        
    except Exception as e:
        print(f"❌ E-posta gönderimi sırasında hata oluştu: {str(e)}")
        return False