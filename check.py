import smtplib
import subprocess
import time
from email.mime.text import MIMEText
from email.utils import formataddr


# QQ 邮箱配置
SMTP_SERVER = "smtp.qq.com"  # QQ 邮箱的 SMTP 服务器地址
SMTP_PORT = 465              # QQ 邮箱的 SMTP 端口（SSL）
EMAIL_FROM = "820863776@qq.com"  # 发件人邮箱
AUTH_CODE = "emnoygnsaohgbcgh"   # QQ 邮箱的授权码

def send_email(subject, body):
    receivers = ['820863776@qq.com','1169591397@qq.com']
    # 创建邮件对象
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = formataddr(("Sender", EMAIL_FROM))  # 使用标准格式
    msg["Subject"] = subject
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_FROM, AUTH_CODE)
            for receiver in receivers:
                msg["To"] = formataddr(("Recipient", receiver))  # 使用标准格式
                server.sendmail(EMAIL_FROM, receiver, msg.as_string())
        print("Email sent successfully to all receivers!")
    except Exception as e:
        pass


def get_gpu_utilization():
    """
    使用 nvidia-smi 获取 GPU 利用率
    """
    try:
        # 调用 nvidia-smi 获取 GPU 利用率
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print("Error running nvidia-smi:", result.stderr)
            return None

        # 解析 GPU 利用率
        utilization = int(result.stdout.strip())
        return utilization
    except Exception as e:
        print("Error getting GPU utilization:", e)
        return None

def monitor_gpu(interval=300, threshold=5):
    """
    监控 GPU 利用率
    :param interval: 监控间隔时间（秒），默认为 300 秒（5 分钟）
    :param threshold: 利用率阈值，默认为 5%
    """
    while True:
        utilization = get_gpu_utilization()
        if utilization is not None:
            print(f"Current GPU utilization: {utilization}%")
            if utilization < threshold:
                send_email(f'A100 gpu free', f'gpu utilization: {utilization}%')
                break  # 退出循环
        else:
            print("Failed to get GPU utilization. Retrying...")

        # 等待指定时间
        time.sleep(interval)

if __name__ == "__main__":
    monitor_gpu()

