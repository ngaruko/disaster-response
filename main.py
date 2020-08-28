from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello google"

@app.route("/docs")
def docs():
    return 'ok deal'

@app.route("/about")
def about():
    return render_template("page.html", title="about page")

if __name__ == "__main__":
    app.run(debug=True)