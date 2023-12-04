import json

import requests
from bs4 import BeautifulSoup
from flask import Blueprint, render_template, request, redirect
from flask_login import login_user, UserMixin

auth_blueprint = Blueprint("auth", __name__)


def try_auth(username: str, password: str) -> tuple[str, str]:
    session = requests.session()

    response = session.get(url="https://schools.by/login/")
    soup = BeautifulSoup(response.text, "html.parser")

    csrfmiddlewaretoken: str = soup.find(
        "input",
        {"name": "csrfmiddlewaretoken"}
    ).attrs["value"]

    response = session.post(
        url="https://schools.by/login/",
        data={
            "csrfmiddlewaretoken": csrfmiddlewaretoken,
            "username": username,
            "password": password,
            "|123": "|123"
        },
        headers={
            "Referer": "https://schools.by/login/"
        }
    )
    school: str = response.url.split("://")[1].split("/")[0].split(".")[0]

    sessionid = session.cookies["sessionid"]
    return school, sessionid


@auth_blueprint.route("/auth", methods=["GET", "POST"])
def auth():
    if request.method == "GET":
        return render_template("auth.html")

    username: str = request.form["username"]
    password: str = request.form["password"]

    school, sessionid = try_auth(username, password)

    db: dict
    with open("db.json", "r", encoding="utf-8") as file:
        db = json.load(file)
        db[sessionid] = school
    with open("db.json", "w", encoding="utf-8") as file:
        json.dump(db, file)

    user = UserMixin()
    user.id = sessionid
    user.school = school
    login_user(user, remember=True)

    return redirect("/")
