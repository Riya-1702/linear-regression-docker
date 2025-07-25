FROM python:3.9-slim-buster
WORKDIR /apps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY advertising_sales_data.csv .
CMD ["python", "-u", "app.py"]                             
