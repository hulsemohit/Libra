from datetime import datetime

from flask_login import UserMixin

from libra import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    games = db.relationship("UserGame", backref="creator", lazy=True)

    def __repr__(self):
        return f"User({self.username})"


class UserGame(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    shapes_path = db.Column(db.String(50), unique=True, nullable=False)
    model_path = db.Column(db.String(50), unique=True, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def __repr__(self):
        return f"Game('{self.shapes_path}', '{self.date_posted}')"
