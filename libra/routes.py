from flask import Flask, request, render_template, redirect, flash, url_for
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.local import LocalProxy

import numpy as np

from libra.models import User, UserGame
from libra.forms import RegistrationForm, LoginForm
from libra import app, db, bcrypt

from libra.core.game import Game


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    LocalProxy
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pass = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        user = User(username=form.username.data, password=hashed_pass)
        db.session.add(user)
        db.session.commit()
        flash(
            f"Welcome to Libra, {form.username.data}! You can now sign in.", "success"
        )
        return redirect(url_for("login"))

    return render_template("register.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get("next", "home")
            return redirect(next_page)
        flash("Failed to Sign In.", "failure")
    return render_template("login.html", form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/games")
@login_required
def games():
    return "Error 418"


TTT_WINNING_SHAPES = list(
    map(
        np.array,
        [
            [
                [1, 1, 1],
            ],
            [
                [1],
                [1],
                [1],
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
        ],
    )
)


CURRENT_GAME = {}


@app.route("/create", methods=["GET", "POST"])
@login_required
def create():
    if request.method == "GET":
        return """
                <p>Enter the grid size</p>
                 <form action="" method="POST">
                 <input type="number" name="size"></input>
                </form>
               """
    return render_template("create.html", value=int(request.form["size"]))


@app.route("/train")
@login_required
def train():
    size = int(request.args.get("size", "3"))
    shapes = request.args.get("array", "0" * size * size)
    shapes = list(map(int, shapes))
    shapes = list(
        map(
            lambda x: x.reshape((size, size)),
            np.array_split(shapes, len(shapes) // (size * size)),
        )
    )

    g = Game(size, shapes, iters=1)
    g.train()

    """

    g = Game(3, TTT_WINNING_SHAPES, savefile="tictactoe-nn")
    """

    CURRENT_GAME[current_user.get_id()] = g
    return redirect(url_for("play"))


@app.route("/play", methods=["GET", "POST"])
@login_required
def play():
    uid = current_user.get_id()
    if uid not in CURRENT_GAME:
        redirect(url_for("home"))

    g = CURRENT_GAME[uid]

    if request.method == "POST" and g.result() is None:
        g.move(int(request.form["choice"]))

    if g.current_player == 1 and g.result() is None:
        g.move(g.predict())

    return render_template(
        "game.html", size=g.size, board=g.current_board(), winner=g.result()
    )
