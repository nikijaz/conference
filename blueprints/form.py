import re

import requests
from bs4 import BeautifulSoup
from flask import Blueprint, render_template
from flask_login import current_user, login_required

form_blueprint = Blueprint("form", __name__)


def get_forms(school: str, session: requests.Session):
    response = session.get(
        url=f"https://{school}.schools.by/classes"
    )
    soup = BeautifulSoup(response.text, "html.parser")

    raw_forms = soup.find(
        attrs={"class": "classes_main_box"}
    ).findAll(
        "span", {"class": "class"}
    )

    forms: dict[str, str] = {}
    for raw_form in raw_forms:
        forms[re.findall("\d+ \".\"", raw_form.a.text)[0]] = f"/scan/{raw_form.a['href'].split('/')[-1]}"

    return forms


@form_blueprint.route("/form", methods=["GET"])
@login_required
def form():
    return render_template("form.html", forms=get_forms(current_user.school, current_user.session))
