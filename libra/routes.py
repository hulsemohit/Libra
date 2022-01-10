from datetime import datetime
import pickle
import random
import string

from flask import Flask, request, render_template, redirect, flash, url_for
from flask_login import login_user, current_user, logout_user, login_required
import numpy as np

from libra.models import User, UserGame
from libra.forms import RegistrationForm, LoginForm
from libra import app, db, bcrypt

from libra.core import utils
from libra.core.game import Game


CURRENT_GAME = {}


@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
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
        flash(
            "Failed to sign in. Please check your credentials and try again.", "warning"
        )
    return render_template("login.html", form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/create", methods=["GET", "POST"])
@login_required
def create():
    if request.method == "GET":
        return render_template("create-size.html")
    return render_template("create.html", size=int(request.form["size"]))


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

    name = str(current_user.get_id()) + str(datetime.now())
    salts = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(10)
    )

    user_game = UserGame(path=name + salts, user_id=current_user.get_id(), size=size)
    db.session.add(user_game)
    db.session.commit()

    game = Game(size, shapes, iters=1)
    game.train()

    game.save_model("models/" + name + salts)
    with open("shapes/" + name + salts, "wb") as shapefile:
        pickle.dump(shapes, shapefile)

    CURRENT_GAME[current_user.get_id()] = game
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


@app.route("/games")
@login_required
def games():
    user_games = UserGame.query.filter_by(user_id=current_user.get_id()).all()
    return render_template("my_games.html", game_list=user_games)


@app.route("/replay", methods=["GET"])
@login_required
def replay():
    game_id = int(request.args.get("id", "1"))
    uid = current_user.get_id()
    user_game = UserGame.query.filter_by(id=game_id, user_id=uid).all()
    if not user_game:
        flash("Game not found.", "warning")
        return redirect(url_for("games"))
    user_game = user_game[0]
    with open("shapes/" + user_game.path, "rb") as f:
        shapes = pickle.load(f)
    game = Game(user_game.size, shapes, savefile="models/" + user_game.path)
    CURRENT_GAME[uid] = game
    return redirect(url_for("play"))
