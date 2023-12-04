import json

import requests
from flask import Flask, redirect
from flask_login import LoginManager, login_required, UserMixin

from blueprints.auth import auth_blueprint
from blueprints.form import form_blueprint
from blueprints.scan import scan_blueprint

app = Flask(__name__)
app.secret_key = b"yM'f+[UwhYqVz7|sO)w)S:oJXF7Ajm"
app.debug = True

login_manager = LoginManager()
login_manager.init_app(app)

app.register_blueprint(auth_blueprint)
app.register_blueprint(form_blueprint)
app.register_blueprint(scan_blueprint)


def setup_session(sessionid: str) -> requests.Session:
    request = requests.get(url="https://schools.by/login/")
    csrftoken: str = request.cookies["csrftoken"]

    session = requests.session()
    session.cookies["csrftoken"] = csrftoken
    session.cookies["sessionid"] = sessionid
    return session


@login_manager.user_loader
def load_user(session_id):
    user = UserMixin()
    user.id = session_id
    user.session = setup_session(session_id)
    with open("db.json", "r", encoding="utf-8") as file:
        db = json.load(file)
        user.school = db[session_id]
    return user


@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect("/auth")


@app.route("/")
@login_required
def index():
    return redirect("/form")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
