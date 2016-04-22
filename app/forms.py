from flask.ext.wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired

class AdditionForm(Form):
    first = StringField('number1', validators=[DataRequired()])
    second = StringField('number2', validators=[DataRequired()])

class TextGenForm(Form):
    primer = StringField('primer_text', validators=[DataRequired()])