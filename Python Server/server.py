from flask import Flask

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return "This is an API for prediction."

if __name__ == "__main__":
    app.run(port=5000, debug=True)