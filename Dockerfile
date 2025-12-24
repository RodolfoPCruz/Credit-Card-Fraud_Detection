FROM python:3.11-slim

# Defining the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY . /app

# Install the dependencies listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose the 8888 port (default port of jupyter)
EXPOSE 8888

#Initialize jupyter notebook inside the container
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

