import numpy as np
import requests
from PIL import Image
from bs4 import BeautifulSoup
from flask import Blueprint, render_template, request, redirect
from flask_login import current_user, login_required

import extractor

scan_blueprint = Blueprint("scan", __name__)


def get_journals(school: str, form: str, session: requests.Session):
    response = session.get(
        url=f"https://{school}.schools.by/class/{form}/journals"
    )
    soup = BeautifulSoup(response.text, "html.parser")

    raw_journals = soup.findAll(
        "a", {"class": "jour3"}
    )

    journals: dict[str, str] = {}
    for raw_journal in raw_journals:
        if raw_journal.span is None:
            continue
        journals[raw_journal.span.text] = raw_journal["href"].split("/")[-1]

    return journals


def get_students(school: str, journal: str, session: requests.Session):
    response = session.get(
        url=f"https://{school}.schools.by/journal/{journal}/quarter/80"
    )
    soup = BeautifulSoup(response.text, "html.parser")

    raw_students = soup.findAll(
        "a", {"class": "user_type_1"}
    )

    students: list[tuple[str, str]] = []
    for i, raw_student in enumerate(raw_students):
        student = raw_student.text.split(" ")
        students.append((f"{i + 1}. {student[0]} {student[1][0]}.", raw_student["href"].split("/")[-1]))

    return students


def get_dates(school: str, journal: str, session: requests.Session):
    dates: list[tuple[str, str, str, str]] = []
    for i in range(2):
        response = session.get(
            url=f"https://{school}.schools.by/journal/{journal}/quarter/{80 + i}"
        )
        soup = BeautifulSoup(response.text, "html.parser")

        raw_dates = soup.findAll(
            "tr", {"class": "lesson_dates"}
        )[1].findAll(
            "td", {"class": "lesson_date"}
        )

        if not raw_dates:
            continue

        for raw_date in raw_dates:
            date = raw_date["day"].split("-")
            dates.append((date[2], date[1], raw_date["lesson_id"], raw_date["day"]))

        dates.append((["1Ч", "2Ч", "3Ч", "4Ч"][i], " ", "", ""))

    return dates


def place_mark(school: str, journal: str, session: requests.Session, lesson_id, date, student_id, mark):
    print(session.post(
        url=f"https://{school}.schools.by/marks/class-subject:{journal}/set",
        data={
            "id": "",
            "m": mark,
            "note": "",
            "lesson_id": lesson_id,
            "lesson_date": date,
            "pupil_id": student_id
        }
    ).text)


def get_marks(school: str, journal: str, session: requests.Session, marks: list[list]):
    indexj = -1
    for n in range(2):
        response = session.get(
            url=f"https://{school}.schools.by/journal/{journal}/quarter/{80 + n}"
        )
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.findAll(
            "tbody"
        )[-1].findAll(
            "tr"
        )

        j: int
        for i, row in enumerate(rows):
            j = indexj
            for raw_mark in row.findAll("td", {"class": "mark"}):
                j += 1
                if not raw_mark.get("m_id"):
                    continue

                marks[i][j] = raw_mark.b.text

        indexj = j + 1

    return marks


@scan_blueprint.route("/scan/<form_id>", methods=["GET", "POST"])
@login_required
def scan(form_id):
    if request.method == "GET":
        return render_template("scan.html", journals=get_journals(current_user.school, form_id, current_user.session))

    if request.files.get("file") is not None:
        students = get_students(current_user.school, request.form["journal"], current_user.session)
        dates = get_dates(current_user.school, request.form["journal"], current_user.session)
        marks = get_marks(current_user.school, request.form["journal"], current_user.session,
                          [[None for _ in range(len(dates))] for _ in range(len(students))])

        mat = Image.open(request.files["file"])
        mat = np.array(mat)
        scanned_marks = extractor.extract(mat)

        return render_template(
            "journal.html",
            students=students,
            dates=dates,
            marks=marks,
            scanned_marks=scanned_marks,
            journal=request.form["journal"]
        )

    journal = request.form["journal"]
    for raw_entry in request.form:
        if raw_entry == "journal":
            continue

        entry = raw_entry.split("_")
        lesson_id = entry[0]
        date = entry[1]
        student_id = entry[2]
        mark = request.form[raw_entry]

        if not lesson_id:
            continue

        place_mark(current_user.school, journal, current_user.session, lesson_id, date, student_id, mark)

    return redirect("/")
